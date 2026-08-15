"""Boucle temps reel : source audio -> analyse -> OSC + affichage.

Le temps de reference est l'horloge audio (nombre d'echantillons consommes),
pas l'horloge murale. Consequence : sur une source fichier ou synthetique, la
session se rejoue a l'identique, aussi vite que le CPU le permet, et les dates
des evenements restent exactes.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .analysis.engine import AnalysisEngine, AnalysisFrame
from .audio.sources import AudioSource
from .bridge.osc import JamBridge
from .config import Config
from .generative.scheduler import RhythmScheduler
from .ui.terminal import TerminalDisplay


@dataclass
class SessionStats:
    """De quoi juger une session apres coup (et calibrer les seuils)."""

    frames: int = 0
    onsets: int = 0
    key_changes: int = 0
    hits: dict[str, int] = field(default_factory=lambda: {"kick": 0, "perc": 0})
    duration_s: float = 0.0
    wall_time_s: float = 0.0

    @property
    def realtime_factor(self) -> float:
        """Duree audio / temps CPU. > 1 : on tient le temps reel."""
        if self.wall_time_s <= 0:
            return float("inf")
        return self.duration_s / self.wall_time_s

    def summary(self) -> str:
        return (
            f"{self.duration_s:.1f}s analysees en {self.wall_time_s:.1f}s "
            f"(x{self.realtime_factor:.1f} temps reel) | "
            f"{self.onsets} attaques | {self.key_changes} changements de tonalite | "
            f"{self.hits['kick']} kicks, {self.hits['perc']} percus"
        )


class JamSession:
    """Assemble analyse, generation et sortie pour une session de jeu."""

    def __init__(
        self,
        config: Config,
        source: AudioSource,
        bridge: JamBridge | None = None,
        display: TerminalDisplay | None = None,
        enable_rhythm: bool = True,
        enable_drone: bool = True,
        pace_realtime: bool | None = None,
        chroma_backend: str = "numpy",
        onset_backend: str = "numpy",
    ):
        self.config = config
        self.source = source
        self.bridge = bridge
        self.display = display
        self.enable_rhythm = enable_rhythm
        self.enable_drone = enable_drone
        # Une source non temps reel (fichier, synthese) defile aussi vite que
        # le CPU le permet ; on la cadence seulement si on veut l'ecouter.
        self.pace_realtime = (
            (not source.realtime) if pace_realtime is None else pace_realtime
        )

        self.engine = AnalysisEngine(
            config, chroma_backend=chroma_backend, onset_backend=onset_backend
        )
        self.scheduler = RhythmScheduler(config.rhythm)
        self.stats = SessionStats()
        self._stop = False
        self._last_key = None

    def stop(self) -> None:
        """Demande l'arret ; la boucle sort a la fin du bloc courant."""
        self._stop = True

    def run(self, max_duration_s: float | None = None) -> SessionStats:
        """Boucle principale. Rend les statistiques de la session."""
        wall_start = time.perf_counter()
        if self.bridge and self.enable_drone:
            self.bridge.start_drone()

        try:
            for block in self.source.iter_blocks():
                if self._stop:
                    break
                for frame in self.engine.process_block(np.asarray(block)):
                    self._handle_frame(frame)
                    if self.pace_realtime:
                        self._pace(frame.time_s, wall_start)
                if max_duration_s is not None and self.stats.duration_s >= max_duration_s:
                    break
        except KeyboardInterrupt:
            pass
        finally:
            self.stats.wall_time_s = time.perf_counter() - wall_start
            self._teardown()
        return self.stats

    # -- traitement d'une trame ---------------------------------------

    def _handle_frame(self, frame: AnalysisFrame) -> None:
        self.stats.frames += 1
        self.stats.duration_s = frame.time_s

        if frame.onset is not None:
            self.stats.onsets += 1
            if self.bridge:
                self.bridge.send_onset(frame.onset.strength)

        # Compte tenu cote analyse, pas cote OSC : en mode analyse seule il
        # n'y a pas de bridge actif, et le decompte doit rester juste.
        if frame.key is not None and frame.key != self._last_key:
            self.stats.key_changes += 1
            self._last_key = frame.key

        if self.bridge:
            if frame.key is not None and self.enable_drone:
                self.bridge.send_key(frame.key, frame.key_confidence)
            self.bridge.send_energy(frame.level, frame.density)

        if self.enable_rhythm:
            self.scheduler.set_density(frame.density)
            for event in self.scheduler.advance(frame.time_s):
                self.stats.hits[event.voice] += 1
                if self.bridge:
                    self.bridge.send_hit(event.voice, event.velocity, event.step)

        if self.display:
            self.display.render(frame, self.scheduler if self.enable_rhythm else None)

    def _pace(self, audio_time_s: float, wall_start: float) -> None:
        """Freine pour coller au temps reel quand la source ne le fait pas."""
        ahead = audio_time_s - (time.perf_counter() - wall_start)
        if ahead > 0.002:
            time.sleep(ahead)

    def _teardown(self) -> None:
        if self.bridge:
            self.bridge.stop()
            self.bridge.close()
        if self.display:
            self.display.close()
        self.source.close()


def analyze_offline(
    config: Config, samples: np.ndarray, bridge: JamBridge | None = None
) -> list[AnalysisFrame]:
    """Analyse un signal deja en memoire, sans temps reel ni source.

    Utilise par les tests et par `halj analyze --input` pour rejouer une prise.
    """
    engine = AnalysisEngine(config)
    frames = engine.process_block(np.asarray(samples))
    if bridge:
        for frame in frames:
            if frame.onset is not None:
                bridge.send_onset(frame.onset.strength)
            if frame.key is not None:
                bridge.send_key(frame.key, frame.key_confidence)
            bridge.send_energy(frame.level, frame.density)
    return frames
