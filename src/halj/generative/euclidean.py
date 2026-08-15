"""Rythmes euclidiens : le moteur de variete de la couche rythmique.

Un motif euclidien repartit k impulsions aussi uniformement que possible sur n
pas. C'est le generateur retenu au cadrage : une seule grandeur continue (la
densite de jeu) suffit a se promener dans une famille de motifs qui sonnent
tous "musicaux", sans table de patterns ecrite a la main.
"""

from __future__ import annotations

import math


def euclidean_pattern(steps: int, pulses: int, rotation: int = 0) -> list[bool]:
    """Motif de Bjorklund : `pulses` impulsions reparties sur `steps` pas.

    E(3, 8) -> le tresillo, E(5, 8) -> le cinquillo, etc.
    `rotation` decale le motif vers la gauche (rotation=1 : le motif commence
    un pas plus tard dans la sequence).
    """
    if steps <= 0:
        raise ValueError("steps doit etre strictement positif")
    if not 0 <= pulses <= steps:
        raise ValueError(f"pulses hors [0, {steps}] : {pulses}")

    if pulses == 0:
        pattern = [False] * steps
    elif pulses == steps:
        pattern = [True] * steps
    else:
        pattern = _bjorklund(steps, pulses)

    if rotation:
        offset = rotation % steps
        pattern = pattern[offset:] + pattern[:offset]
    return pattern


def _bjorklund(steps: int, pulses: int) -> list[bool]:
    """Algorithme de Bjorklund, formule "plis successifs".

    On part de `pulses` groupes [1] et `steps - pulses` groupes [0], puis on
    replie les restes tant qu'il y a plus d'un reste : ce qui subsiste est la
    repartition la plus uniforme possible.
    """
    groups = [[True] for _ in range(pulses)]
    remainders = [[False] for _ in range(steps - pulses)]

    while len(remainders) > 1:
        pairs = min(len(groups), len(remainders))
        merged = [groups[i] + remainders[i] for i in range(pairs)]
        if len(groups) > pairs:
            leftover = groups[pairs:]
        else:
            leftover = remainders[pairs:]
        groups, remainders = merged, leftover

    return [step for group in groups + remainders for step in group]


def pulses_for_density(density: float, minimum: int, maximum: int) -> int:
    """Interpole un nombre d'impulsions a partir d'une densite dans [0, 1].

    L'arrondi est explicitement "au superieur a mi-chemin" (0.5 -> 1) et non
    l'arrondi bancaire de `round()` en Python (0.5 -> 0). C'est le meme que
    `Math.round` en JavaScript : les deux implementations doivent produire
    exactement le meme motif pour la meme densite, sans quoi elles derivent
    silencieusement l'une de l'autre.
    """
    if minimum > maximum:
        raise ValueError("minimum doit etre <= maximum")
    density = min(1.0, max(0.0, density))
    return int(math.floor(minimum + density * (maximum - minimum) + 0.5))
