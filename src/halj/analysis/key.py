"""Detection de tonalite par Krumhansl-Schmuckler, avec lissage temporel.

Le lissage n'est pas une option de confort : sans lui la tonalite "flicke" a
chaque trame et le drone devient inecoutable. Deux mecanismes se cumulent ici :

1. lissage du chroma (EMA + fenetre glissante) -> stabilise l'entree ;
2. hysterese sur la decision -> une nouvelle tonalite doit battre la tonalite
   courante d'une marge donnee, et tenir plusieurs trames de suite, pour
   declencher un changement.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from ..config import KeyConfig
from ..notes import Key

# Profils Krumhansl-Kessler (1982), moyennes des jugements de stabilite
# tonale, indexes en demi-tons depuis la fondamentale.
KS_MAJOR = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
)
KS_MINOR = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
)


def _zscore(vector: np.ndarray) -> np.ndarray:
    centered = vector - vector.mean()
    norm = np.linalg.norm(centered)
    if norm <= 1e-12:
        return np.zeros_like(centered)
    return centered / norm


# Precalcul : 24 profils centres-normes (12 majeurs puis 12 mineurs).
# Avec des vecteurs centres et de norme 1, la correlation de Pearson se reduit
# a un simple produit scalaire.
_PROFILE_MATRIX = np.stack(
    [_zscore(np.roll(KS_MAJOR, tonic)) for tonic in range(12)]
    + [_zscore(np.roll(KS_MINOR, tonic)) for tonic in range(12)]
)


@dataclass(frozen=True)
class KeyEstimate:
    """Resultat brut d'une trame, avant hysterese."""

    key: Key
    confidence: float  # correlation de Pearson, clippee dans [0, 1]
    margin: float  # ecart avec la meilleure tonalite concurrente
    scores: np.ndarray  # les 24 correlations, pour comparer a une tonalite donnee

    def score_of(self, key: Key) -> float:
        """Correlation obtenue par une tonalite precise (pas forcement la best)."""
        return float(self.scores[_key_to_index(key)])


def correlate_profiles(chroma: np.ndarray) -> np.ndarray:
    """Correlations (24,) du chroma avec les 24 profils K-S."""
    if chroma.shape != (12,):
        raise ValueError(f"chroma de forme {chroma.shape}, (12,) attendu")
    return _PROFILE_MATRIX @ _zscore(np.asarray(chroma, dtype=np.float64))


def estimate_key(chroma: np.ndarray) -> KeyEstimate | None:
    """Meilleure tonalite pour ce chroma, ou None si le chroma est vide."""
    scores = correlate_profiles(chroma)
    if not np.any(scores):
        return None
    best = int(np.argmax(scores))
    others = np.delete(scores, best)
    return KeyEstimate(
        key=_index_to_key(best),
        confidence=float(np.clip(scores[best], 0.0, 1.0)),
        margin=float(scores[best] - others.max()),
        scores=scores,
    )


def _index_to_key(index: int) -> Key:
    return Key(tonic=index % 12, mode="maj" if index < 12 else "min")


def _key_to_index(key: Key) -> int:
    return key.tonic + (0 if key.mode == "maj" else 12)


class KeyTracker:
    """Suit la tonalite dans le temps : lissage du chroma + hysterese.

    Etat expose apres chaque `update` : `key` (tonalite retenue, None tant
    qu'aucune n'est fiable) et `confidence`.
    """

    def __init__(self, config: KeyConfig, hop_s: float):
        self.config = config
        # Une "trame" ici = une trame chroma, donc un hop decime.
        self._frame_s = hop_s * config.decimation
        window_frames = max(1, int(round(config.window_s / self._frame_s)))
        self._window: deque[np.ndarray] = deque(maxlen=window_frames)
        self._ema: np.ndarray | None = None
        self._alpha = 1.0 - np.exp(-self._frame_s / max(config.chroma_tau_s, 1e-6))

        self.key: Key | None = None
        self.confidence: float = 0.0
        self._candidate: Key | None = None
        self._candidate_frames: int = 0

    @property
    def window_frames(self) -> int:
        return self._window.maxlen or 1

    def reset(self) -> None:
        self._window.clear()
        self._ema = None
        self.key = None
        self.confidence = 0.0
        self._candidate = None
        self._candidate_frames = 0

    def update(self, chroma: np.ndarray, rms: float) -> Key | None:
        """Integre un chroma. Retourne la tonalite retenue (eventuellement None).

        Sous le seuil de silence, on ne nourrit pas la fenetre : un blanc entre
        deux phrases ne doit pas diluer le contexte harmonique accumule.
        """
        if rms < self.config.silence_rms or not np.any(chroma):
            return self.key

        chroma = np.asarray(chroma, dtype=np.float64)
        if self._ema is None:
            self._ema = chroma.copy()
        else:
            self._ema += self._alpha * (chroma - self._ema)
        self._window.append(self._ema.copy())

        accumulated = np.mean(self._window, axis=0)
        estimate = estimate_key(accumulated)
        if estimate is None:
            return self.key

        self._apply_hysteresis(estimate)
        return self.key

    def _apply_hysteresis(self, estimate: KeyEstimate) -> None:
        if estimate.confidence < self.config.min_confidence:
            # Trop incertain pour changer quoi que ce soit : on garde la
            # tonalite courante et on casse le compteur du candidat.
            self._candidate = None
            self._candidate_frames = 0
            return

        if self.key is None:
            self.key = estimate.key
            self.confidence = estimate.confidence
            self._candidate = None
            self._candidate_frames = 0
            return

        if estimate.key == self.key:
            self.confidence = estimate.confidence
            self._candidate = None
            self._candidate_frames = 0
            return

        # Le candidat doit battre la tonalite en place d'une marge nette — et
        # non pas seulement arriver en tete du classement. C'est ce qui evite
        # les bascules relatif majeur / relatif mineur, dont les profils K-S
        # sont tres proches.
        if estimate.score_of(estimate.key) - estimate.score_of(self.key) < (
            self.config.switch_margin
        ):
            self._candidate = None
            self._candidate_frames = 0
            return

        if estimate.key == self._candidate:
            self._candidate_frames += 1
        else:
            self._candidate = estimate.key
            self._candidate_frames = 1

        if self._candidate_frames >= self.config.hold_frames:
            self.key = estimate.key
            self.confidence = estimate.confidence
            self._candidate = None
            self._candidate_frames = 0
