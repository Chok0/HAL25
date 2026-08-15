"""Energie de jeu -> densite rythmique.

La densite est la grandeur qui pilote la couche rythmique : elle melange
"combien ca sonne fort" (RMS lisse) et "a quel rythme ca attaque" (debit
d'onsets). Un accord plaque tenu et un tremolo au mediator ont la meme energie
brute mais ne doivent pas produire la meme densite.
"""

from __future__ import annotations

import math

import numpy as np

from ..config import EnergyConfig


def rms(frame: np.ndarray) -> float:
    if frame.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(frame.astype(np.float64)))))


def _to_db(value: float) -> float:
    return 20.0 * math.log10(max(value, 1e-9))


class EnergyTracker:
    """Suit le niveau de jeu et en derive une densite dans [0, 1].

    Les constantes de temps sont asymetriques : l'attaque monte vite (le rythme
    doit repondre a la relance) et la retombee est lente (une phrase qui respire
    ne doit pas couper la couche rythmique).
    """

    def __init__(self, config: EnergyConfig, hop_s: float):
        self.config = config
        self._hop_s = hop_s
        self._attack = 1.0 - math.exp(-hop_s / max(config.attack_tau_s, 1e-6))
        self._release = 1.0 - math.exp(-hop_s / max(config.release_tau_s, 1e-6))
        self.level: float = 0.0  # niveau lisse, dans [0, 1]
        self.onset_rate: float = 0.0  # attaques/seconde, lisse
        self._rate_decay = math.exp(-hop_s / max(config.release_tau_s, 1e-6))

    def reset(self) -> None:
        self.level = 0.0
        self.onset_rate = 0.0

    def update(self, frame_rms: float, onset_count: int = 0) -> float:
        """Integre un hop. Retourne la densite courante."""
        target = self._normalize(frame_rms)
        coeff = self._attack if target > self.level else self._release
        self.level += coeff * (target - self.level)

        # Debit d'attaques : chaque onset injecte 1/hop_s "attaque par seconde",
        # amorti ensuite exponentiellement.
        self.onset_rate *= self._rate_decay
        if onset_count:
            self.onset_rate += onset_count * (1.0 - self._rate_decay) / self._hop_s

        return self.density

    def _normalize(self, frame_rms: float) -> float:
        db = _to_db(frame_rms)
        span = self.config.ceil_db - self.config.floor_db
        if span <= 0:
            raise ValueError("ceil_db doit etre superieur a floor_db")
        return float(np.clip((db - self.config.floor_db) / span, 0.0, 1.0))

    @property
    def density(self) -> float:
        rate_component = min(
            1.0, self.onset_rate / max(self.config.onset_rate_full, 1e-9)
        )
        weight = self.config.onset_rate_weight
        return float(
            np.clip((1.0 - weight) * self.level + weight * rate_component, 0.0, 1.0)
        )
