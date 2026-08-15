"""Generateur de guitare synthetique (Karplus-Strong).

Sert a deux choses : alimenter le mode `--source synth` quand aucune guitare
n'est branchee, et donner aux tests un signal deterministe dont on connait la
tonalite et la position exacte des attaques.

Karplus-Strong plutot que des sinusoides : le spectre produit a des partiels
qui decroissent et une attaque bruitee, donc il exerce reellement le chroma et
le flux spectral, la ou une sinusoide pure les rendrait trivialement faciles.
"""

from __future__ import annotations

import numpy as np

from ..notes import midi_to_hz

# Voicings guitare usuels, en demi-tons relatifs a la fondamentale.
MAJOR_TRIAD = (0, 4, 7, 12, 16)
MINOR_TRIAD = (0, 3, 7, 12, 15)


def karplus_strong(
    freq: float,
    duration_s: float,
    samplerate: int,
    decay: float = 0.995,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Une corde pincee a `freq`, sur `duration_s` secondes.

    Le filtre en peigne est applique periode par periode (une passe vectorisee
    par periode) plutot qu'echantillon par echantillon : c'est le meme
    principe, des centaines de fois plus rapide en Python. La difference porte
    sur un echantillon au point de bouclage, sans incidence sur ce qu'on
    mesure ici. `decay` s'applique donc par periode, pas par echantillon.
    """
    if freq <= 0:
        raise ValueError("frequence positive attendue")
    rng = rng or np.random.default_rng(0)
    n_samples = max(0, int(duration_s * samplerate))
    if n_samples == 0:
        return np.zeros(0, dtype=np.float64)
    delay = max(2, int(round(samplerate / freq)))

    buffer = rng.uniform(-1.0, 1.0, delay)
    # Le filtre moyenneur conserve la moyenne du buffer : une excitation a
    # moyenne non nulle laisserait une composante continue qui ne decroit
    # jamais, et qui domine le spectre. On la retire a la source.
    buffer -= buffer.mean()
    n_periods = int(np.ceil(n_samples / delay))
    output = np.empty(n_periods * delay, dtype=np.float64)
    for period in range(n_periods):
        output[period * delay : (period + 1) * delay] = buffer
        # Filtre moyenneur : c'est lui qui mange les aigus au fil du temps.
        buffer = 0.5 * (buffer + np.roll(buffer, -1)) * decay
    return output[:n_samples]


def strum(
    root_midi: int,
    intervals: tuple[int, ...],
    duration_s: float,
    samplerate: int,
    spread_s: float = 0.018,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Un accord gratte : les cordes entrent decalees de `spread_s`."""
    rng = rng or np.random.default_rng(0)
    n_samples = int(duration_s * samplerate)
    out = np.zeros(n_samples, dtype=np.float64)
    for index, interval in enumerate(intervals):
        offset = int(index * spread_s * samplerate)
        if offset >= n_samples:
            break
        voice = karplus_strong(
            midi_to_hz(root_midi + interval),
            duration_s - offset / samplerate,
            samplerate,
            rng=rng,
        )
        out[offset : offset + voice.size] += voice / len(intervals)
    return out


def chord_progression(
    roots_midi: list[int],
    samplerate: int,
    modes: list[str] | None = None,
    chord_duration_s: float = 2.0,
    seed: int = 0,
) -> np.ndarray:
    """Enchaine des accords grattes, un par entree de `roots_midi`."""
    rng = np.random.default_rng(seed)
    modes = modes or ["maj"] * len(roots_midi)
    if len(modes) != len(roots_midi):
        raise ValueError("modes et roots_midi doivent avoir la meme longueur")
    chunks = [
        strum(
            root,
            MAJOR_TRIAD if mode == "maj" else MINOR_TRIAD,
            chord_duration_s,
            samplerate,
            rng=rng,
        )
        for root, mode in zip(roots_midi, modes)
    ]
    return np.concatenate(chunks) if chunks else np.zeros(0)
