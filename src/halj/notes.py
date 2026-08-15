"""Utilitaires notes / tonalites, partages entre analyse et generation."""

from __future__ import annotations

from dataclasses import dataclass

NOTE_NAMES: tuple[str, ...] = (
    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
)

A4_HZ = 440.0
A4_MIDI = 69


def midi_to_hz(midi: float) -> float:
    return A4_HZ * (2.0 ** ((midi - A4_MIDI) / 12.0))


def hz_to_midi(hz: float) -> float:
    if hz <= 0:
        raise ValueError("frequence positive attendue")
    from math import log2

    return 12.0 * log2(hz / A4_HZ) + A4_MIDI


def pitch_class_to_hz(pitch_class: int, octave: int) -> float:
    """Frequence de la note `pitch_class` (0 = C) dans l'octave donnee.

    Convention scientifique : C4 = 261.6 Hz (MIDI 60).
    """
    midi = 12 * (octave + 1) + (pitch_class % 12)
    return midi_to_hz(midi)


@dataclass(frozen=True)
class Key:
    """Une tonalite : fondamentale (0 = C) + mode."""

    tonic: int
    mode: str  # "maj" | "min"

    def __post_init__(self) -> None:
        if not 0 <= self.tonic < 12:
            raise ValueError(f"tonic hors [0,12) : {self.tonic}")
        if self.mode not in ("maj", "min"):
            raise ValueError(f"mode inconnu : {self.mode}")

    @property
    def name(self) -> str:
        return f"{NOTE_NAMES[self.tonic]} {'maj' if self.mode == 'maj' else 'min'}"

    def root_hz(self, octave: int = 2) -> float:
        return pitch_class_to_hz(self.tonic, octave)

    def __str__(self) -> str:  # pragma: no cover - confort d'affichage
        return self.name
