"""Configuration centrale du pipeline.

Un seul objet passe de la capture audio jusqu'au scheduler : ca evite les
constantes eparpillees et rend les tests reproductibles (on instancie un
Config, on l'injecte, on assert).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AudioConfig:
    samplerate: int = 44100
    # hop court = onsets reactifs. 256 @ 44.1k -> 5.8 ms de grain temporel.
    hop_size: int = 256
    channels: int = 1
    device: int | str | None = None
    # taille du bloc rendu par le driver ; multiple de hop_size.
    block_size: int = 512


@dataclass(frozen=True)
class KeyConfig:
    """Detection de tonalite : chroma -> Krumhansl-Schmuckler -> lissage."""

    # Fenetre longue : a 44.1k, 8192 echantillons donnent ~5.4 Hz de resolution,
    # necessaire pour separer les notes graves de la guitare (E2 82 / F2 87 Hz).
    frame_size: int = 8192
    # Le chroma n'a pas besoin du grain temporel de l'onset : 1 trame sur 8
    # (~46 ms) suffit et divise d'autant le cout CPU.
    decimation: int = 8
    # fenetre glissante d'accumulation du chroma, en secondes.
    window_s: float = 2.0
    # constante de temps du lissage exponentiel du vecteur chroma.
    chroma_tau_s: float = 0.35
    # bornes de la bande utile guitare (E2 ~82 Hz -> harmoniques aigues).
    fmin_hz: float = 73.4  # D2, une case sous le Mi grave en drop
    fmax_hz: float = 2093.0  # C7
    bins_per_octave: int = 36  # 3 bins/demi-ton : tolere un accordage flottant
    # Pente de compensation spectrale, en 1/f^tilt. C'est le reglage le plus
    # sensible de toute la detection : une corde de guitare a un 5e harmonique
    # qui est une tierce majeure au-dessus de la fondamentale, si bien qu'un
    # chroma non compense entend un do majeur comme un mi mineur. Mesure sur
    # un banc de 24 accords + 24 lignes monodiques : 8/48 sans compensation,
    # 47/48 a partir de 1.5. Le palier s'etend jusqu'a 3.0.
    spectral_tilt: float = 2.0
    # Largeur des gaussiennes de projection, en multiples de l'ecart entre
    # bins. Au-dela de ~1.5 les classes voisines commencent a se melanger.
    bin_width_scale: float = 1.3
    # Hysterese : la nouvelle tonalite doit battre la courante de cette marge
    # (en score K-S normalise) pendant `hold_frames` trames consecutives.
    switch_margin: float = 0.06
    hold_frames: int = 12
    # En dessous de ce score, on considere la tonalite comme non fiable.
    min_confidence: float = 0.55
    # En dessous de ce RMS, on ne nourrit pas le chroma (silence / souffle).
    silence_rms: float = 0.004


@dataclass(frozen=True)
class OnsetConfig:
    """Detection d'attaques par flux spectral a seuil adaptatif."""

    # Fenetre courte : on veut la reactivite, pas la resolution frequentielle.
    frame_size: int = 1024
    # mediane glissante servant de plancher au seuil, en secondes.
    median_window_s: float = 0.35
    # Facteur multiplicatif au-dessus de la mediane locale. Le flux etant
    # normalise dans [0, 1], ces seuils sont absolus et transposables : une
    # attaque vaut 0.6 a 1.0, le sustain d'un accord tourne autour de 0.10.
    # Le comptage d'attaques est identique entre delta 4.5 et 6 sur le banc
    # de signaux de test — signe que le reglage n'est pas sur une arete.
    delta: float = 5.0
    # Plancher absolu : c'est lui qui tranche pour une attaque isolee dans le
    # silence, ou la mediane locale vaut 0 et ne dit rien.
    floor: float = 0.25
    # Seuil propre au backend aubio, dont l'echelle n'a rien a voir avec
    # `delta` (fonction de detection differente, deja normalisee).
    aubio_threshold: float = 0.4
    # temps mort apres une attaque (bruit de mediator, double-attaque).
    min_interval_s: float = 0.045
    # passe-haut avant flux spectral : coupe les rumbles et le bruit de manche.
    highpass_hz: float = 120.0


@dataclass(frozen=True)
class EnergyConfig:
    """Energie de jeu -> densite rythmique."""

    # constantes de temps asymetriques : monte vite, redescend lentement.
    attack_tau_s: float = 0.12
    release_tau_s: float = 1.6
    # RMS mappe sur [0, 1] entre ces deux bornes (echelle dB).
    floor_db: float = -48.0
    ceil_db: float = -12.0
    # part de la densite venant du debit d'attaques (le reste vient du RMS).
    onset_rate_weight: float = 0.45
    # debit d'attaques considere comme "plein regime", en attaques/seconde.
    onset_rate_full: float = 6.0


@dataclass(frozen=True)
class RhythmConfig:
    """Grille euclidienne pilotee par la densite."""

    bpm: float = 84.0
    steps: int = 16
    # nombre d'impacts kick interpole entre ces bornes selon la densite.
    kick_pulses_min: int = 2
    kick_pulses_max: int = 7
    perc_pulses_min: int = 0
    perc_pulses_max: int = 11
    # rotations fixes : decale les motifs pour eviter kick et perc a l'unisson.
    kick_rotation: int = 0
    perc_rotation: int = 3
    # en dessous de cette densite, la couche rythmique se tait.
    gate_density: float = 0.08


@dataclass(frozen=True)
class DroneConfig:
    amp: float = 0.35
    # glissando entre deux tonalites, en secondes (pas de saut brutal).
    glide_s: float = 1.8
    # octave de la fondamentale du drone (2 = E2/C2 territoire basse).
    octave: int = 2


@dataclass(frozen=True)
class OscConfig:
    host: str = "127.0.0.1"
    port: int = 57120  # sclang, pas scsynth (57110)
    prefix: str = "/jam"


@dataclass(frozen=True)
class Config:
    audio: AudioConfig = field(default_factory=AudioConfig)
    key: KeyConfig = field(default_factory=KeyConfig)
    onset: OnsetConfig = field(default_factory=OnsetConfig)
    energy: EnergyConfig = field(default_factory=EnergyConfig)
    rhythm: RhythmConfig = field(default_factory=RhythmConfig)
    drone: DroneConfig = field(default_factory=DroneConfig)
    osc: OscConfig = field(default_factory=OscConfig)

    @property
    def hop_s(self) -> float:
        """Duree d'un hop en secondes : l'unite de temps de tout le pipeline."""
        return self.audio.hop_size / self.audio.samplerate

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        sections = {
            "audio": AudioConfig,
            "key": KeyConfig,
            "onset": OnsetConfig,
            "energy": EnergyConfig,
            "rhythm": RhythmConfig,
            "drone": DroneConfig,
            "osc": OscConfig,
        }
        kwargs: dict[str, Any] = {}
        for name, klass in sections.items():
            if name in data:
                kwargs[name] = klass(**data[name])
        unknown = set(data) - set(sections)
        if unknown:
            raise ValueError(f"sections inconnues dans la config : {sorted(unknown)}")
        return cls(**kwargs)

    @classmethod
    def load(cls, path: str | Path | None) -> "Config":
        if path is None:
            return cls()
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def with_overrides(self, **sections: Any) -> "Config":
        """Retourne une copie avec des sections remplacees (Config est frozen)."""
        return replace(self, **sections)
