"""Grille d'accords : l'appli propose, au lieu de seulement suivre.

Un accompagnement qui ne fait que suivre la tonalite detectee finit par tourner
en rond — et, sur haut-parleur, par se suivre lui-meme. Le directeur harmonique
renverse le rapport : il tient une grille, l'avance a la mesure, et ne s'en
detourne que si le jeu insiste vraiment.

C'est un seul reglage, `agency` :

    0.0   suiveur pur — l'accord est la triade de la tonalite detectee, comme
          avant l'introduction de ce module ;
    0.5   conversation — l'appli deroule sa grille, mais cede des que le jeu
          designe clairement un autre accord ;
    1.0   meneur — la grille tient bon, le jeu doit vraiment insister.

Le vote du joueur est lu dans le chroma accumule depuis le dernier changement :
la masse tombant dans l'accord, plus un bonus sur la fondamentale. Ce bonus
n'est pas cosmetique — La mineur et Do majeur partagent deux notes sur trois,
c'est la fondamentale qui les separe.

Les progressions sont ecrites en degres, donc transposables telles quelles dans
la tonalite detectee, et parcourues dans un ordre deterministe : deux sessions
identiques donnent la meme grille, ce qui rend le comportement testable.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log2

from ..config import HarmonyConfig
from ..notes import NOTE_NAMES, Key, pitch_class_to_hz

# Intervalles de chaque qualite d'accord, en demi-tons depuis la fondamentale.
CHORD_INTERVALS: dict[str, tuple[int, ...]] = {
    "maj": (0, 4, 7),
    "min": (0, 3, 7),
    "sus4": (0, 5, 7),
}

# Degres disponibles par mode : nom -> (demi-tons depuis la tonique, qualite).
DEGREES: dict[str, dict[str, tuple[int, str]]] = {
    "min": {
        "i": (0, "min"),
        "III": (3, "maj"),
        "iv": (5, "min"),
        "v": (7, "min"),
        "V": (7, "maj"),
        "VI": (8, "maj"),
        "VII": (10, "maj"),
    },
    "maj": {
        "I": (0, "maj"),
        "ii": (2, "min"),
        "iii": (4, "min"),
        "IV": (5, "maj"),
        "V": (7, "maj"),
        "vi": (9, "min"),
    },
}

# Grilles proposees, en degres. Toutes commencent sur la tonique : quel que
# soit le moment ou l'ancre change, la grille repart d'un point stable.
PROGRESSIONS: dict[str, tuple[tuple[str, ...], ...]] = {
    "min": (
        ("i", "VI", "III", "VII"),
        ("i", "iv", "VII", "III"),
        ("i", "VII", "VI", "V"),
        ("i", "iv", "v", "i"),
    ),
    "maj": (
        ("I", "V", "vi", "IV"),
        ("I", "IV", "V", "IV"),
        ("I", "vi", "ii", "V"),
        ("I", "IV", "vi", "V"),
    ),
}


@dataclass(frozen=True)
class Chord:
    """Un accord : fondamentale (0 = do), qualite, et son degre d'origine."""

    root: int
    quality: str
    degree: str = ""

    def __post_init__(self) -> None:
        if not 0 <= self.root < 12:
            raise ValueError(f"root hors [0,12) : {self.root}")
        if self.quality not in CHORD_INTERVALS:
            raise ValueError(f"qualite inconnue : {self.quality}")

    @property
    def intervals(self) -> tuple[int, ...]:
        return CHORD_INTERVALS[self.quality]

    @property
    def pitch_classes(self) -> tuple[int, ...]:
        return tuple((self.root + i) % 12 for i in self.intervals)

    @property
    def name(self) -> str:
        suffix = {"maj": "", "min": "m", "sus4": "sus4"}[self.quality]
        return f"{NOTE_NAMES[self.root]}{suffix}"

    def root_hz(self, octave: int = 2) -> float:
        return pitch_class_to_hz(self.root, octave)

    def __str__(self) -> str:  # pragma: no cover - confort d'affichage
        return self.name


def chord_from_degree(key: Key, degree: str) -> Chord:
    """Transpose un degre dans la tonalite donnee."""
    table = DEGREES[key.mode]
    if degree not in table:
        raise ValueError(f"degre inconnu en {key.mode} : {degree}")
    semitones, quality = table[degree]
    return Chord((key.tonic + semitones) % 12, quality, degree)


def progression_for(key: Key, index: int) -> list[Chord]:
    """La `index`-ieme grille du mode, transposee dans la tonalite."""
    grids = PROGRESSIONS[key.mode]
    return [chord_from_degree(key, d) for d in grids[index % len(grids)]]


def chord_score(chord: Chord, chroma, root_weight: float = 0.5) -> float:
    """Masse du chroma tombant dans l'accord, fondamentale sur-ponderee.

    Le chroma est normalise en L1, donc la masse est directement comparable
    d'un accord a l'autre : pas de normalisation supplementaire a faire.
    """
    total = sum(float(v) for v in chroma)
    if total <= 1e-12:
        return 0.0
    inside = sum(float(chroma[pc]) for pc in chord.pitch_classes)
    return (inside + root_weight * float(chroma[chord.root])) / total


# Rappel vers le registre nominal, en octaves de penalite par octave d'ecart.
# Sans lui, "prendre l'octave la plus proche" a chaque accord est un cliquet :
# une grille descendante entraine le drone toujours plus bas, et il finit sous
# le seuil d'audition d'un haut-parleur de telephone. Mesure : Am-F-C-G perdait
# une octave et demie en deux tours.
REGISTER_PULL = 0.5


def voiced_root_hz(
    pitch_class: int, octave: int, previous_hz: float | None = None
) -> float:
    """Fondamentale du drone : plus petit deplacement, sans quitter le registre.

    Le drone glisse d'un accord au suivant — un saut de septieme s'entend comme
    un plongeon la ou son renversement passe inapercu. On accepte donc l'octave
    voisine quand elle rapproche vraiment, mais sous le rappel ci-dessus, qui
    borne l'ecart au registre nominal a une octave et interdit la derive.
    """
    base = pitch_class_to_hz(pitch_class, octave)
    if not previous_hz or previous_hz <= 0:
        return base
    best, best_cost = base, None
    for shift in (-1, 0, 1):
        candidate = base * (2.0**shift)
        cost = abs(log2(candidate / previous_hz)) + REGISTER_PULL * abs(shift)
        if best_cost is None or cost < best_cost:
            best, best_cost = candidate, cost
    return best


@dataclass(frozen=True)
class ChordChange:
    """Ce qui vient d'etre decide a une frontiere de la grille."""

    chord: Chord
    decision: str  # "lead" | "follow" | "anchor"
    step: int


class HarmonyDirector:
    """Tient la grille, l'avance a la mesure, ecoute si le jeu propose autre chose.

    Modele "pull", comme le sequenceur rythmique : l'appelant fait avancer les
    pas et recupere les changements. Aucun thread, aucune horloge interne, donc
    une session se rejoue a l'identique dans un test.
    """

    def __init__(self, config: HarmonyConfig, steps_per_bar: int = 16):
        self.config = config
        self.steps_per_bar = steps_per_bar
        # Initiative *du moment* : `generative/conversation.py` la fait varier
        # selon a qui est le tour. Sans lui, elle reste celle du reglage.
        self.agency = config.agency
        self.key: Key | None = None
        self.progression: list[Chord] = []
        self.position = 0
        self.chord: Chord | None = None
        self.last_decision = "anchor"
        self._grid_index = 0
        self._loops = 0
        self._vote = [0.0] * 12
        self._next_change_step = 0
        self.agency = self.config.agency

    @property
    def steps_per_chord(self) -> int:
        return max(1, self.steps_per_bar * self.config.bars_per_chord)

    @property
    def next_chord(self) -> Chord | None:
        """L'accord que la grille propose ensuite — affichable, donc jouable."""
        if not self.progression:
            return None
        return self.progression[(self.position + 1) % len(self.progression)]

    def steps_to_change(self, step_index: int) -> int:
        return max(0, self._next_change_step - step_index)

    def reset(self) -> None:
        self.key = None
        self.progression = []
        self.position = 0
        self.chord = None
        self.last_decision = "anchor"
        self._grid_index = 0
        self._loops = 0
        self._vote = [0.0] * 12
        self._next_change_step = 0

    # -- ce que l'appli entend ----------------------------------------

    def explains(self, key: Key) -> bool:
        """La tonalite detectee est-elle deja un degre de la grille en place ?

        C'est la question qui evite le tic le plus penible : ancre en La
        mineur, le joueur passe sur Fa, le detecteur annonce "Fa majeur" — mais
        Fa est le VI degre, donc rien n'a bouge, c'est la grille qui fonctionne.
        Ne re-ancrer que sur une tonalite reellement etrangere.
        """
        if self.key is None:
            return False
        for degree in DEGREES[self.key.mode]:
            chord = chord_from_degree(self.key, degree)
            if chord.root != key.tonic:
                continue
            if (chord.quality == "min") == (key.mode == "min"):
                return True
        return False

    def observe_key(self, key: Key | None, step_index: int = 0) -> ChordChange | None:
        """Ancre la grille sur la tonalite detectee, si elle sort vraiment du cadre.

        Le KeyTracker a deja son hysterese, mais elle se compte en fractions de
        seconde : elle stabilise le drone, pas une grille d'accords. Le filtre
        qui compte ici est harmonique, pas temporel — cf. `explains`.
        """
        if key is None or key == self.key or self.explains(key):
            return None
        self.key = key
        # Une grille differente a chaque ancrage : reprendre la meme suite
        # d'accords a chaque changement de tonalite s'entendrait comme un tic.
        self.progression = progression_for(key, self._grid_index)
        self._grid_index += 1
        self._loops = 0
        self.position = 0
        self.chord = self.progression[0]
        self.last_decision = "anchor"
        self._vote = [0.0] * 12
        self._next_change_step = step_index + self.steps_per_chord
        return ChordChange(self.chord, "anchor", step_index)

    def observe_chroma(self, chroma) -> None:
        """Accumule le vote du joueur depuis le dernier changement d'accord."""
        for pc in range(12):
            self._vote[pc] += float(chroma[pc])

    # -- avancee de la grille ------------------------------------------

    def on_step(self, step_index: int) -> ChordChange | None:
        """Fait avancer la grille jusqu'a ce pas ; rend le changement s'il y a lieu."""
        if self.chord is None or self.key is None:
            return None
        # Le sequenceur se recale apres une pause du process : on se recale avec
        # lui plutot que de deverser tous les changements en retard.
        if step_index < self._next_change_step - 2 * self.steps_per_chord:
            self._next_change_step = step_index + self.steps_per_chord
            return None
        if step_index < self._next_change_step:
            return None

        self._next_change_step = step_index + self.steps_per_chord
        change = self._decide(step_index)
        self._vote = [0.0] * 12
        return change

    def _decide(self, step_index: int) -> ChordChange:
        agency = min(1.0, max(0.0, self.agency))
        if agency <= 0.0 or not self.progression:
            # Suiveur pur : la triade de la tonalite, rien d'autre.
            self.chord = chord_from_degree(self.key, self._tonic_degree())
            self.last_decision = "follow"
            return ChordChange(self.chord, "follow", step_index)

        planned = self.progression[(self.position + 1) % len(self.progression)]
        candidate = self._player_choice()
        margin = self.config.follow_margin_min + agency * (
            self.config.follow_margin_max - self.config.follow_margin_min
        )

        if candidate is not None:
            score, chord = candidate
            if score >= self.config.min_vote and score - chord_score(
                planned, self._vote, self.config.root_weight
            ) > margin:
                # Le jeu designe clairement un autre accord : on le rejoint, et
                # la grille reprend a partir de la s'il en fait partie.
                self.chord = chord
                self.last_decision = "follow"
                for index, item in enumerate(self.progression):
                    if item.root == chord.root and item.quality == chord.quality:
                        self.position = index
                        break
                return ChordChange(chord, "follow", step_index)

        self._advance_position()
        self.chord = self.progression[self.position]
        self.last_decision = "lead"
        return ChordChange(self.chord, "lead", step_index)

    def _advance_position(self) -> None:
        self.position = (self.position + 1) % len(self.progression)
        if self.position != 0:
            return
        self._loops += 1
        if self._loops >= max(1, self.config.renew_after_loops):
            self._loops = 0
            self._grid_index += 1
            self.progression = progression_for(self.key, self._grid_index)

    def _player_choice(self) -> tuple[float, Chord] | None:
        """Meilleur accord au sens du vote accumule, parmi les degres du mode."""
        if sum(self._vote) <= 1e-12 or self.key is None:
            return None
        best: tuple[float, Chord] | None = None
        for degree in DEGREES[self.key.mode]:
            chord = chord_from_degree(self.key, degree)
            score = chord_score(chord, self._vote, self.config.root_weight)
            if best is None or score > best[0]:
                best = (score, chord)
        return best

    def _tonic_degree(self) -> str:
        return "i" if self.key.mode == "min" else "I"
