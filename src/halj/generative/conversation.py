"""Qui mene, a cet instant — et ce que l'appli fait quand c'est son tour.

Dans une jam, la question n'est jamais tranchee une fois pour toutes : le lead
circule, et il circule surtout par la place qu'on laisse. Celui qui s'arrete
invite l'autre a prendre la parole ; celui qui relance la reprend.

Un accompagnement a `agency` fixe ne fait qu'une moitie du chemin. Il suit
bien, ou il mene bien, mais il ne *repond* pas : on joue devant lui, pas avec
lui. Ce module ajoute la moitie manquante — l'appli mesure la place que prend
l'instrumentiste et se decide en consequence :

* il occupe le terrain -> elle accompagne, se range derriere sa grille, et
  garde juste assez de rythme pour tenir le fond ;
* il laisse un blanc -> elle prend la main : elle tient sa grille, releve la
  densite au lieu de la laisser retomber, et **repond** par une phrase.

Trois details font la difference entre un partenaire et un metronome :

* **la reprise est plus rapide que la prise.** L'appli met quelques secondes a
  s'autoriser a mener, et rend la main en moins d'une : un partenaire qui
  n'ecoute pas est pire qu'un partenaire muet.
* **la densite ne retombe jamais a zero.** La couche rythmique est censee
  motiver le jeu ; si elle s'eteint des qu'on respire, elle motive un silence.
  Elle a donc un plancher — bas quand l'appli accompagne, franchement plus haut
  quand elle mene.
* **quand elle parle, elle n'ecoute pas.** Une phrase de reponse tombe dans la
  bande analysee, transitoire comme du jeu : rien ne permettrait de la
  distinguer d'un instrument. L'appli met donc son analyse en pause pendant
  qu'elle joue ses propres notes, exactement comme on ne s'ecoute pas parler.

`agency` reste le seul reglage, mais il devient une *disposition* plutot qu'un
etat : a 0 l'appli n'ose jamais prendre la main, a 1 elle ne la rend jamais, et
entre les deux elle decide au cas par cas — d'autant plus vite qu'il est haut.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp

from ..config import ConversationConfig
from .harmony import Chord

PLAYER = "player"
APP = "app"


def _lerp(low: float, high: float, amount: float) -> float:
    return low + (high - low) * min(1.0, max(0.0, amount))


# Contours de reponse, en degres de l'accord : 0 fondamentale, 1 tierce,
# 2 quinte, 3 octave, 4 neuvieme (la tierce a l'octave). Quatre contours et
# quatre placements suffisent a ne pas se repeter sur une grille de quatre
# accords, tout en restant deterministes — donc testables, et identiques entre
# les deux portages.
RESPONSE_CONTOURS: tuple[tuple[int, ...], ...] = (
    (0, 1, 2, 3),
    (3, 2, 1, 0),
    (2, 3, 1, 2),
    (0, 2, 1, 4),
)
RESPONSE_POSITIONS: tuple[tuple[int, ...], ...] = (
    (0, 4, 8, 12),
    (0, 3, 6, 11),
    (2, 4, 8, 11),
    (0, 4, 7, 12),
)


@dataclass(frozen=True)
class ResponseNote:
    """Une note de la phrase de reponse, placee sur la grille."""

    step: int  # position dans la mesure
    midi: int
    velocity: float


def response_phrase(
    chord: Chord, bar_index: int, octave: int = 4
) -> list[ResponseNote]:
    """Phrase de reponse sur l'accord en cours.

    Registre median : c'est celui qu'un petit haut-parleur de telephone
    reproduit reellement, alors qu'il ne rend a peu pres rien du drone. La
    reponse est donc ce que l'instrumentiste entend en premier — ce qui tombe
    bien, puisque c'est ce qui doit lui donner envie de repondre a son tour.
    """
    contour = RESPONSE_CONTOURS[bar_index % len(RESPONSE_CONTOURS)]
    positions = RESPONSE_POSITIONS[bar_index % len(RESPONSE_POSITIONS)]
    third, fifth = chord.intervals[1], chord.intervals[2]
    ladder = (0, third, fifth, 12, 12 + third)
    base = 12 * (octave + 1) + chord.root
    return [
        ResponseNote(
            step=step,
            midi=base + ladder[degree],
            # La premiere note pose la phrase, les suivantes la prolongent.
            velocity=0.55 if index == 0 else 0.4,
        )
        for index, (degree, step) in enumerate(zip(contour, positions))
    ]


class Conversation:
    """Suit la place prise par l'instrumentiste, et decide a qui est le tour.

    Modele "pull" comme le reste de la generation : l'appelant pousse ce qu'il
    mesure, lit ce qui en decoule. Aucune horloge interne, donc une session se
    rejoue a l'identique dans un test.
    """

    def __init__(self, config: ConversationConfig, agency: float):
        self.config = config
        self.base_agency = min(1.0, max(0.0, agency))
        self.presence = 0.0
        # Presence typique de l'instrumentiste : l'echelle a laquelle se lisent
        # les seuils. Monte au rythme du jeu, redescend en une minute.
        self.reference = 0.0
        self.lead = PLAYER
        self.since_s = 0.0
        self._elapsed_s = 0.0
        self._pending_since: float | None = None
        self._bars_led = 0

    # -- disposition, derivee du seul reglage ---------------------------

    @property
    def typical(self) -> float:
        """La presence de jeu typique, servant d'echelle a tout le reste."""
        return max(self.reference, self.config.reference_floor)

    @property
    def free_below(self) -> float:
        """Presence en dessous de laquelle l'appli considere la place libre."""
        ratio = _lerp(
            self.config.free_ratio_min, self.config.free_ratio_max, self.base_agency
        )
        return self.typical * ratio

    @property
    def busy_above(self) -> float:
        """...et au-dessus de laquelle elle rend la main. La bande entre les
        deux est l'hysterese : sans elle, le lead clignoterait a chaque note."""
        ratio = _lerp(
            self.config.free_ratio_min, self.config.free_ratio_max, self.base_agency
        )
        return self.typical * (ratio + self.config.hysteresis_ratio)

    @property
    def take_s(self) -> float:
        return _lerp(self.config.take_s_max, self.config.take_s_min, self.base_agency)

    @property
    def give_s(self) -> float:
        return _lerp(self.config.give_s_min, self.config.give_s_max, self.base_agency)

    @property
    def leads(self) -> bool:
        return self.lead == APP

    # -- ce que l'appli mesure ------------------------------------------

    def observe(self, density: float, time_s: float) -> str | None:
        """Integre la densite de jeu. Rend le nouveau meneur s'il a change.

        La densite est deja la bonne grandeur : elle melange le niveau et le
        debit d'attaques, et l'auto-ecoute lui a retire ce que l'appli joue
        elle-meme. Reste a la regarder sur une fenetre assez longue pour qu'un
        silence entre deux phrases ne passe pas pour un abandon.
        """
        dt = max(0.0, time_s - self.since_s) if self.since_s else 0.0
        cible = min(1.0, max(0.0, density))
        tau = (
            self.config.presence_attack_s
            if cible > self.presence
            else self.config.presence_release_s
        )
        self.presence += (1.0 - exp(-dt / max(tau, 1e-6))) * (cible - self.presence)
        tau_ref = (
            self.config.presence_attack_s
            if self.presence > self.reference
            else self.config.reference_release_s
        )
        self.reference += (1.0 - exp(-dt / max(tau_ref, 1e-6))) * (
            self.presence - self.reference
        )
        self.since_s = time_s
        self._elapsed_s += dt

        if not self.config.enabled:
            return None
        # Tant que la mesure ne veut rien dire, on accompagne.
        if self._elapsed_s < self.config.engage_s:
            return None
        if self.base_agency <= self.config.never_leads_below:
            return self._settle(PLAYER, time_s)
        if self.base_agency >= self.config.always_leads_above:
            return self._settle(APP, time_s)

        if self.lead == PLAYER:
            vise, delai = APP, self.take_s
            atteint = self.presence < self.free_below
        else:
            vise, delai = PLAYER, self.give_s
            atteint = self.presence > self.busy_above

        if not atteint:
            self._pending_since = None
            return None
        if self._pending_since is None:
            self._pending_since = time_s
            return None
        if time_s - self._pending_since < delai:
            return None
        return self._settle(vise, time_s)

    def _settle(self, lead: str, time_s: float) -> str | None:
        self._pending_since = None
        if lead == self.lead:
            return None
        self.lead = lead
        self._bars_led = 0
        return lead

    # -- ce qui en decoule ----------------------------------------------

    def agency(self) -> float:
        """L'initiative harmonique du moment, pas celle du curseur.

        Quand l'appli mene, elle tient sa grille ; quand elle accompagne, elle
        se laisse detourner beaucoup plus facilement.
        """
        if not self.config.enabled:
            return self.base_agency
        if self.leads:
            return max(self.base_agency, self.config.lead_agency)
        return min(self.base_agency, self.config.follow_agency)

    def density(self, measured: float) -> float:
        """La densite reellement jouee : celle mesuree, mais avec un plancher.

        C'est ici que se joue "la percu ne doit pas abandonner". Le plancher
        d'accompagnement garde un fond audible quand l'instrumentiste respire ;
        celui de meneur releve franchement la couche rythmique, parce qu'une
        relance qui n'insiste pas n'est pas une relance.
        """
        if not self.config.enabled:
            return measured
        plancher = (
            _lerp(self.config.lead_density_min, self.config.lead_density_max,
                  self.base_agency)
            if self.leads
            else self.config.follow_density
        )
        return min(1.0, max(measured, plancher))

    def wants_response(self) -> bool:
        """L'appli repond-elle a cette mesure ?

        Seulement quand elle mene, et pas a toutes les mesures : une phrase a
        chaque mesure n'est plus une reponse, c'est un bavardage. Une mesure sur
        deux laisse a l'instrumentiste le temps de repondre a son tour.
        """
        if not (self.config.enabled and self.config.respond and self.leads):
            return False
        self._bars_led += 1
        return self._bars_led % max(1, self.config.respond_every_bars) == 1
