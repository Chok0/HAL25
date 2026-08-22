"""La grille d'accords : ce que l'appli propose de son propre chef.

Le directeur harmonique est la deuxieme moitie de la reponse au probleme du
haut-parleur : une appli qui tient sa grille ne depend plus d'entendre juste a
chaque instant. Encore faut-il qu'elle sache aussi ceder.
"""

import pytest

from halj.config import HarmonyConfig
from halj.generative.harmony import (
    DEGREES,
    PROGRESSIONS,
    Chord,
    HarmonyDirector,
    chord_from_degree,
    chord_score,
    progression_for,
    voiced_root_hz,
)
from halj.notes import Key, pitch_class_to_hz

LA_MINEUR = Key(9, "min")
DO_MAJEUR = Key(0, "maj")


def chroma_de(*classes, poids=1.0):
    """Chroma normalise portant du poids sur les classes donnees."""
    vecteur = [0.0] * 12
    for pitch_class in classes:
        vecteur[pitch_class % 12] += poids
    total = sum(vecteur)
    return [v / total for v in vecteur]


# --- accords ----------------------------------------------------------


def test_les_qualites_donnent_les_bonnes_notes():
    assert Chord(9, "min").pitch_classes == (9, 0, 4)  # la do mi
    assert Chord(0, "maj").pitch_classes == (0, 4, 7)  # do mi sol
    assert Chord(7, "sus4").pitch_classes == (7, 0, 2)  # sol do re


def test_les_noms_sont_lisibles():
    assert Chord(9, "min").name == "Am"
    assert Chord(5, "maj").name == "F"
    assert Chord(7, "sus4").name == "Gsus4"


def test_une_qualite_inconnue_est_refusee():
    with pytest.raises(ValueError):
        Chord(0, "aug")


def test_les_degres_se_transposent():
    assert chord_from_degree(LA_MINEUR, "VI") == Chord(5, "maj", "VI")
    assert chord_from_degree(LA_MINEUR, "i") == Chord(9, "min", "i")
    assert chord_from_degree(DO_MAJEUR, "V") == Chord(7, "maj", "V")


def test_la_premiere_grille_est_celle_qu_on_attend():
    assert [c.name for c in progression_for(LA_MINEUR, 0)] == ["Am", "F", "C", "G"]
    assert [c.name for c in progression_for(DO_MAJEUR, 0)] == ["C", "G", "Am", "F"]


def test_toutes_les_grilles_partent_de_la_tonique():
    """Quel que soit le moment ou l'ancre change, la grille repart d'un point stable."""
    for mode, grilles in PROGRESSIONS.items():
        tonique = "i" if mode == "min" else "I"
        for grille in grilles:
            assert grille[0] == tonique
        for grille in grilles:
            for degre in grille:
                assert degre in DEGREES[mode]


# --- le vote du joueur -------------------------------------------------


def test_le_vote_distingue_la_mineur_de_do_majeur():
    """Deux notes sur trois en commun : c'est la fondamentale qui tranche."""
    joue_la_mineur = chroma_de(9, 9, 0, 4)  # la insiste
    assert chord_score(Chord(9, "min"), joue_la_mineur) > chord_score(
        Chord(0, "maj"), joue_la_mineur
    )
    joue_do_majeur = chroma_de(0, 0, 4, 7)
    assert chord_score(Chord(0, "maj"), joue_do_majeur) > chord_score(
        Chord(9, "min"), joue_do_majeur
    )


def test_un_chroma_vide_ne_vote_pas():
    assert chord_score(Chord(9, "min"), [0.0] * 12) == 0.0


# --- conduite des voix -------------------------------------------------


def test_la_fondamentale_reste_dans_son_registre():
    """Prendre l'octave la plus proche a chaque accord serait un cliquet."""
    precedente = None
    for _ in range(4):
        for pitch_class in (9, 5, 0, 7):  # la, fa, do, sol
            precedente = voiced_root_hz(pitch_class, 2, precedente)
            nominale = pitch_class_to_hz(pitch_class, 2)
            assert nominale / 2 <= precedente <= nominale * 2


def test_la_grille_boucle_sur_les_memes_hauteurs():
    """Deux tours de la meme grille doivent redonner les memes frequences."""
    hauteurs, precedente = [], None
    for _ in range(3):
        for pitch_class in (9, 5, 0, 7):
            precedente = voiced_root_hz(pitch_class, 2, precedente)
            hauteurs.append(round(precedente, 6))
    assert hauteurs[0:4] == hauteurs[4:8] == hauteurs[8:12]


def test_sans_precedent_on_prend_le_registre_nominal():
    assert voiced_root_hz(9, 2) == pytest.approx(110.0)


# --- le directeur ------------------------------------------------------


def directeur(**reglages):
    return HarmonyDirector(HarmonyConfig(**reglages), steps_per_bar=16)


def test_la_grille_s_ancre_sur_la_tonalite_detectee():
    d = directeur()
    changement = d.observe_key(LA_MINEUR, step_index=0)
    assert changement is not None
    assert changement.decision == "anchor"
    assert d.chord.name == "Am"
    assert d.next_chord.name == "F"
    assert d.steps_to_change(0) == 16


def test_la_grille_avance_toute_seule():
    """Personne ne joue : l'appli deroule quand meme, c'est tout l'interet."""
    d = directeur(agency=1.0)
    d.observe_key(LA_MINEUR)
    joues = [d.on_step(step).chord.name for step in (16, 32, 48, 64)]
    assert joues == ["F", "C", "G", "Am"]
    assert all(d.on_step(step) is None for step in (17, 20, 31))


def test_l_accord_ne_change_qu_a_la_mesure():
    d = directeur()
    d.observe_key(LA_MINEUR)
    for step in range(1, 16):
        assert d.on_step(step) is None
    assert d.on_step(16) is not None


def test_l_appli_cede_quand_le_joueur_insiste():
    """Le joueur plaque un accord qui n'est pas celui prevu : la grille le rejoint."""
    d = directeur(agency=0.5)
    d.observe_key(LA_MINEUR)  # grille Am - F - C - G, prochain accord prevu : F
    for _ in range(40):
        d.observe_chroma(chroma_de(7, 7, 11, 2))  # sol majeur, appuye
    changement = d.on_step(16)
    assert changement.decision == "follow"
    assert changement.chord.name == "G"
    # La grille reprend a partir de la, elle ne recommence pas au debut.
    assert d.next_chord.name == "Am"


def test_un_meneur_ne_cede_pas_pour_si_peu():
    """Meme insistance, agency au maximum : l'appli tient sa proposition."""
    d = directeur(agency=1.0)
    d.observe_key(LA_MINEUR)
    for _ in range(40):
        d.observe_chroma(chroma_de(7, 11, 2))
    changement = d.on_step(16)
    assert changement.decision == "lead"
    assert changement.chord.name == "F"


def test_agency_nulle_rend_le_suiveur_d_avant():
    """A zero, le comportement doit etre exactement celui d'avant ce module."""
    d = directeur(agency=0.0)
    d.observe_key(LA_MINEUR)
    for step in (16, 32, 48):
        changement = d.on_step(step)
        assert changement.decision == "follow"
        assert changement.chord.name == "Am"  # la triade de la tonalite, point


def test_un_degre_de_la_grille_ne_re_ancre_pas():
    """Ancre en la mineur, le joueur passe sur fa : c'est le VI, rien n'a bouge."""
    d = directeur()
    d.observe_key(LA_MINEUR)
    grille = list(d.progression)
    assert d.explains(Key(5, "maj"))  # VI
    assert d.explains(Key(0, "maj"))  # III, le relatif majeur
    assert d.explains(Key(4, "min"))  # v
    assert d.observe_key(Key(5, "maj"), step_index=8) is None
    assert d.progression == grille


def test_une_vraie_modulation_re_ancre():
    d = directeur()
    d.observe_key(LA_MINEUR)
    assert not d.explains(Key(6, "min"))
    changement = d.observe_key(Key(6, "min"), step_index=8)
    assert changement is not None and changement.decision == "anchor"
    assert d.chord.name == "F#m"
    # ...et sur une autre grille, pour que le changement de tonalite s'entende.
    assert [c.name for c in d.progression] != ["F#m", "D", "A", "E"]


def test_la_grille_se_renouvelle_apres_quelques_tours():
    d = directeur(agency=1.0, renew_after_loops=1)
    d.observe_key(LA_MINEUR)
    depart = [c.name for c in d.progression]
    for step in range(16, 16 * 6, 16):
        d.on_step(step)
    assert [c.name for c in d.progression] != depart
    assert d.progression[0].name == "Am"  # toujours ancree sur la meme tonique


def test_le_directeur_se_recale_apres_une_pause_du_process():
    """Le sequenceur se recale sur le present ; le directeur doit suivre."""
    d = directeur()
    d.observe_key(LA_MINEUR, step_index=1000)
    assert d.on_step(0) is None  # horloge revenue en arriere : on se recale
    assert d.steps_to_change(0) == 16
    assert d.on_step(16) is not None


def test_sans_tonalite_le_directeur_ne_propose_rien():
    d = directeur()
    assert d.observe_key(None) is None
    assert d.on_step(16) is None
    assert d.chord is None and d.next_chord is None
