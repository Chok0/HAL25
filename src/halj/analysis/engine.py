"""Orchestration de l'analyse : audio brut -> etat musical par hop.

L'engine est volontairement synchrone et sans thread : on lui pousse des blocs
d'echantillons, il rend une liste d'`AnalysisFrame`. Toute la gestion du temps
reel (callback audio, threads, horloge) vit dans `halj.runtime`, ce qui rend
l'analyse rejouable a l'identique sur un fichier ou un signal de test.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import Config
from ..notes import Key
from .chroma import ChromaExtractor
from .energy import EnergyTracker, rms
from .key import KeyTracker
from .onset import Onset, create_onset_detector
from .selfmask import SelfListen


@dataclass(frozen=True)
class AnalysisFrame:
    """Etat musical a l'instant d'un hop."""

    time_s: float
    rms: float  # niveau mesure au micro, tel quel
    level: float  # niveau de jeu lisse, [0, 1]
    density: float  # densite rythmique, [0, 1]
    onset: Onset | None
    key: Key | None
    key_confidence: float
    chroma: np.ndarray | None  # renseigne uniquement sur les trames chroma
    # Ce qui reste du micro une fois retire ce que l'appli joue elle-meme.
    # Egal a `rms` tant que l'appli ne produit aucun son.
    play_rms: float = 0.0
    # Part de l'energie spectrale attribuee a l'auto-ecoute, [0, 1].
    self_removed: float = 0.0


class AnalysisEngine:
    """Chaine complete : chroma + tonalite + onsets + energie."""

    def __init__(
        self,
        config: Config,
        chroma_backend: str = "numpy",
        onset_backend: str = "numpy",
    ):
        self.config = config
        self.chroma = ChromaExtractor.create(config.audio, config.key, chroma_backend)
        self.onset_detector = create_onset_detector(
            config.audio, config.onset, onset_backend
        )
        self.key_tracker = KeyTracker(config.key, config.hop_s)
        self.energy = EnergyTracker(config.energy, config.hop_s)
        # L'appli sait ce qu'elle joue : elle peut donc le retirer de ce
        # qu'elle entend, au lieu de se prendre elle-meme pour un musicien.
        self.self_listen = SelfListen(
            config.self_listen,
            config.audio.samplerate,
            config.key.frame_size,
            config.key.fmin_hz,
            config.key.fmax_hz,
        )
        self._self_mask_ok = getattr(self.chroma, "supports_self_mask", False)

        self._hop = config.audio.hop_size
        # Historique glissant pour le chroma, qui a besoin d'une fenetre longue.
        # Le detecteur d'attaques, lui, entretient la sienne.
        self._history = np.zeros(config.key.frame_size, dtype=np.float64)
        self._pending = np.zeros(0, dtype=np.float64)
        self._hop_index = 0

    @property
    def hop_index(self) -> int:
        return self._hop_index

    def reset(self) -> None:
        self._history[:] = 0.0
        self._pending = np.zeros(0, dtype=np.float64)
        self._hop_index = 0
        self.onset_detector.reset()
        self.key_tracker.reset()
        self.energy.reset()
        self.self_listen.reset()

    # -- ce que l'appli joue, declare par l'appelant -------------------

    def set_self_drone(self, root_hz: float, intervals=(0,)) -> None:
        """Declare l'accord tenu par le drone, pour pouvoir le soustraire."""
        self.self_listen.set_drone(root_hz, intervals)

    def note_self_hit(self, time_s: float) -> None:
        """Declare un impact rythmique emis, pour ne pas le compter en attaque."""
        self.self_listen.note_hit(time_s)

    def process_block(self, block: np.ndarray) -> list[AnalysisFrame]:
        """Consomme un bloc d'echantillons mono, rend une trame par hop complet.

        Un bloc de taille quelconque est accepte : le reliquat est conserve
        jusqu'au prochain appel.
        """
        block = np.asarray(block, dtype=np.float64).reshape(-1)
        self._pending = (
            block if self._pending.size == 0 else np.concatenate([self._pending, block])
        )

        frames: list[AnalysisFrame] = []
        while self._pending.size >= self._hop:
            hop, self._pending = self._pending[: self._hop], self._pending[self._hop :]
            frames.append(self._process_hop(hop))
        return frames

    def _process_hop(self, hop: np.ndarray) -> AnalysisFrame:
        self._history = np.concatenate([self._history[self._hop :], hop])
        time_s = self._hop_index * self.config.hop_s
        self._hop_index += 1

        frame_rms = rms(hop)
        hop_s = self.config.hop_s
        play_rms = self.self_listen.clean_rms(frame_rms)

        # Le detecteur d'attaques entretient sa propre fenetre (il filtre en
        # amont) : on lui passe le hop brut, pas une tranche de l'historique.
        onset = self.onset_detector.process(
            hop,
            time_s,
            threshold_scale=self.self_listen.threshold_scale(time_s),
        )
        if onset is not None and not self.self_listen.playing:
            # Le niveau dit que personne ne joue : ce qui a franchi le seuil est
            # une crete du battement du drone, pas une attaque. Sans ce
            # garde-fou, l'appli s'excite sur son propre rythme.
            onset = None
        if onset is not None:
            self.self_listen.note_onset(onset.time_s)

        density = self.energy.update(play_rms, onset_count=1 if onset else 0)

        chroma: np.ndarray | None = None
        if self._hop_index % self.config.key.decimation == 0:
            frame = self._history[-self.config.key.frame_size :]
            if self._self_mask_ok:
                spectrum = self.chroma.spectrum(frame)
                spectrum = self.self_listen.clean_spectrum(
                    spectrum, hop_s * self.config.key.decimation
                )
                chroma = self.chroma.from_spectrum(spectrum)
            else:
                chroma = self.chroma.process(frame)
            # Le tracker recoit le niveau *joue* : si le micro n'entend que le
            # drone de l'appli, il n'y a rien a analyser et le silence doit
            # etre reconnu comme tel. Et tant que l'auto-ecoute se cale, on ne
            # lui donne rien : sinon l'appli s'accorderait sur elle-meme dans
            # la demi-seconde qui suit le demarrage, et n'en bougerait plus.
            if self.self_listen.playing and not self.self_listen.warming:
                self.key_tracker.update(chroma, play_rms)

        return AnalysisFrame(
            time_s=time_s,
            rms=frame_rms,
            level=self.energy.level,
            density=density,
            onset=onset,
            key=self.key_tracker.key,
            key_confidence=self.key_tracker.confidence,
            chroma=chroma,
            play_rms=play_rms,
            self_removed=self.self_listen.removed_ratio,
        )
