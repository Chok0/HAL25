"""Sequenceur de la couche rythmique.

Modele "pull" : l'appelant fait avancer l'horloge (`advance(now)`) et recupere
les evenements echus. Aucun thread, aucun sleep interne — la meme sequence est
donc rejouable a l'identique dans un test avec une horloge virtuelle.

Le motif n'est pas un etat mais une fonction pure de la densite : a chaque pas,
on redemande le motif euclidien correspondant a la densite du moment. Deux
motifs euclidiens de k voisins partagent l'essentiel de leurs impulsions, donc
la variation de densite s'entend comme une respiration, pas comme une rupture.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import RhythmConfig
from .euclidean import euclidean_pattern, pulses_for_density


@dataclass(frozen=True)
class RhythmEvent:
    """Un impact a jouer."""

    time_s: float  # date theorique du pas, sur l'horloge du scheduler
    voice: str  # "kick" | "perc"
    velocity: float  # [0, 1]
    step: int  # position dans la grille, [0, steps)


class RhythmScheduler:
    """Grille euclidienne cadencee par une horloge interne.

    Le tempo est un parametre fixe : le beat tracking causal est hors scope v0.
    L'horloge interne sert de trame regulière, c'est la densite — donc le jeu —
    qui fait vivre le motif.
    """

    def __init__(self, config: RhythmConfig, start_time: float = 0.0):
        self.config = config
        self._start_time = start_time
        self._next_step = 0
        self.density = 0.0

    @property
    def step_duration(self) -> float:
        """Duree d'un pas, en secondes (grille de `steps` pas sur 4 temps)."""
        beat = 60.0 / self.config.bpm
        return beat * 4.0 / self.config.steps

    @property
    def next_step(self) -> int:
        return self._next_step

    def reset(self, start_time: float = 0.0) -> None:
        self._start_time = start_time
        self._next_step = 0

    def set_density(self, density: float) -> None:
        self.density = min(1.0, max(0.0, density))

    def pattern(self, voice: str) -> list[bool]:
        """Motif courant d'une voix, pour la densite actuelle."""
        cfg = self.config
        if voice == "kick":
            pulses = pulses_for_density(
                self.density, cfg.kick_pulses_min, cfg.kick_pulses_max
            )
            rotation = cfg.kick_rotation
        elif voice == "perc":
            pulses = pulses_for_density(
                self.density, cfg.perc_pulses_min, cfg.perc_pulses_max
            )
            rotation = cfg.perc_rotation
        else:
            raise ValueError(f"voix inconnue : {voice}")
        return euclidean_pattern(cfg.steps, min(pulses, cfg.steps), rotation)

    def advance(self, now: float) -> list[RhythmEvent]:
        """Fait avancer l'horloge jusqu'a `now`, rend les evenements echus."""
        events: list[RhythmEvent] = []
        gated = self.density < self.config.gate_density

        # Garde-fou : apres une pause du process (ou un gros hoquet systeme),
        # on ne rejoue pas le retard accumule — on se recale sur le present.
        late_steps = int((now - self._step_time(self._next_step)) / self.step_duration)
        if late_steps > self.config.steps:
            self._resync(now)

        while self._step_time(self._next_step) <= now:
            step = self._next_step
            slot = step % self.config.steps
            if not gated:
                events.extend(self._events_for_step(step, slot))
            self._next_step += 1

        return events

    def _events_for_step(self, step: int, slot: int) -> list[RhythmEvent]:
        events: list[RhythmEvent] = []
        time_s = self._step_time(step)
        for voice in ("kick", "perc"):
            if not self.pattern(voice)[slot]:
                continue
            events.append(
                RhythmEvent(
                    time_s=time_s,
                    voice=voice,
                    velocity=self._velocity(slot),
                    step=slot,
                )
            )
        return events

    def _velocity(self, slot: int) -> float:
        # Accent sur les temps forts : sans lui, une grille euclidienne dense
        # s'aplatit en nappe de clics sans hierarchie.
        accent = 1.0 if slot % 4 == 0 else 0.78
        return round(min(1.0, 0.35 + 0.65 * self.density) * accent, 4)

    def _step_time(self, step: int) -> float:
        return self._start_time + step * self.step_duration

    def _resync(self, now: float) -> None:
        self._start_time = now
        self._next_step = 0
