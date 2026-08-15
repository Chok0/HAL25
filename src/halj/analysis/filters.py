"""Filtres a etat, appliques en continu sur le flux de hops.

Un filtrage par trame remettrait l'etat a zero a chaque fenetre et
produirait une discontinuite a chaque bord ; ces filtres gardent donc leur
etat entre les appels et se nourrissent des hops, pas des trames.
"""

from __future__ import annotations

import math

import numpy as np

try:  # chemin rapide : recurrence IIR compilee
    from scipy.signal import lfilter

    _HAS_SCIPY = True
except ImportError:  # pragma: no cover - depend de l'environnement
    _HAS_SCIPY = False


class Biquad:
    """Cellule biquad Direct Form I, coefficients normalises sur a0."""

    def __init__(self, b: tuple[float, float, float], a: tuple[float, float, float]):
        a0 = a[0]
        self.b = np.array(b, dtype=np.float64) / a0
        self.a = np.array(a, dtype=np.float64) / a0
        self._x1 = self._x2 = 0.0
        self._y1 = self._y2 = 0.0
        # Etat au format scipy (2 elements) quand le chemin rapide est actif.
        self._zi = np.zeros(2, dtype=np.float64)

    @classmethod
    def highpass(cls, cutoff_hz: float, samplerate: int, q: float = 0.7071) -> "Biquad":
        """Passe-haut Butterworth (Audio EQ Cookbook)."""
        if not 0 < cutoff_hz < samplerate / 2:
            raise ValueError(
                f"frequence de coupure hors bande : {cutoff_hz} Hz "
                f"pour {samplerate} Hz d'echantillonnage"
            )
        w0 = 2.0 * math.pi * cutoff_hz / samplerate
        cos_w0, sin_w0 = math.cos(w0), math.sin(w0)
        alpha = sin_w0 / (2.0 * q)
        b = ((1 + cos_w0) / 2, -(1 + cos_w0), (1 + cos_w0) / 2)
        a = (1 + alpha, -2 * cos_w0, 1 - alpha)
        return cls(b, a)

    def reset(self) -> None:
        self._x1 = self._x2 = self._y1 = self._y2 = 0.0
        self._zi = np.zeros(2, dtype=np.float64)

    def process(self, block: np.ndarray) -> np.ndarray:
        """Filtre un bloc en conservant l'etat pour le bloc suivant."""
        block = np.asarray(block, dtype=np.float64)
        if block.size == 0:
            return block.copy()
        if _HAS_SCIPY:
            out, self._zi = lfilter(self.b, self.a, block, zi=self._zi)
            return out
        out = np.empty_like(block)
        b0, b1, b2 = self.b
        _, a1, a2 = self.a
        x1, x2, y1, y2 = self._x1, self._x2, self._y1, self._y2
        for index, x0 in enumerate(block):
            y0 = b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
            out[index] = y0
            x2, x1 = x1, x0
            y2, y1 = y1, y0
        self._x1, self._x2, self._y1, self._y2 = x1, x2, y1, y2
        return out


class CascadedHighpass:
    """Deux biquads en cascade : pente de 24 dB/octave.

    Une seule cellule (12 dB/oct) laisse encore passer assez de grave pour que
    la fuite spectrale d'un rumble a 40 Hz reveille le flux spectral. La
    cascade descend ce residu sous le plancher de detection.
    """

    def __init__(self, cutoff_hz: float, samplerate: int):
        # Q de Butterworth d'ordre 4 : 0.5412 puis 1.3066.
        self._stages = [
            Biquad.highpass(cutoff_hz, samplerate, q=0.5412),
            Biquad.highpass(cutoff_hz, samplerate, q=1.3066),
        ]

    def reset(self) -> None:
        for stage in self._stages:
            stage.reset()

    def process(self, block: np.ndarray) -> np.ndarray:
        for stage in self._stages:
            block = stage.process(block)
        return block
