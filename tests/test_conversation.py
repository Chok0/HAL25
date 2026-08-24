"""A qui est le tour — et ce que l'appli fait quand c'est le sien.

Le point a couvrir n'est pas "l'appli sait mener" ni "l'appli sait suivre",
mais qu'elle sache passer de l'un a l'autre : c'est ce passage qui fait la
difference entre jouer devant quelqu'un et jouer avec quelqu'un.
"""

import pytest

from halj.config import ConversationConfig
from halj.generative.conversation import (
    APP,
    PLAYER,
    Conversation,
    response_phrase,
)
from halj.generative.harmony import Chord


def conversation(agency=0.5, **reglages):
    return Conversation(ConversationConfig(**reglages), agency)


def jouer(conv, densite, duree_s, depart=0.0, pas=0.05):
    """Pousse une densite constante pendant `duree_s`. Rend les bascules."""
    bascules, time_s = [], depart
    for _ in range(int(duree_s / pas)):
        time_s += pas
        change = conv.observe(densite, time_s)
        if change is not None:
            bascules.append((change, round(time_s - depart, 2)))
    return bascules, time_s


# --- le lead circule ---------------------------------------------------


def test_l_appli_prend_la_main_quand_on_lui_laisse_la_place():
    conv = conversation()
    _, t = jouer(conv, 0.35, 12.0)
    assert conv.lead == PLAYER  # tant que ca joue, elle accompagne

    bascules, _ = jouer(conv, 0.0, 12.0, depart=t)
    assert [b[0] for b in bascules] == [APP]
    assert 2.0 < bascules[0][1] < 9.0  # ni dans la seconde, ni au bout d'un siecle


def test_l_appli_rend_la_main_des_qu_on_relance():
    conv = conversation()
    _, t = jouer(conv, 0.35, 12.0)
    _, t = jouer(conv, 0.0, 12.0, depart=t)
    assert conv.leads

    bascules, _ = jouer(conv, 0.35, 6.0, depart=t)
    assert [b[0] for b in bascules] == [PLAYER]
    # Rendre doit etre plus rapide que prendre : un partenaire qui n'ecoute pas
    # est pire qu'un partenaire muet.
    assert bascules[0][1] < 2.5


def test_les_seuils_suivent_la_dynamique_du_joueur():
    """Une ligne melodique douce et un accompagnement gratte doivent donner le
    meme comportement : les seuils sont relatifs, jamais absolus."""
    dates = []
    for niveau in (0.35, 0.12, 0.05):
        conv = conversation()
        _, t = jouer(conv, niveau, 14.0)
        bascules, _ = jouer(conv, 0.0, 12.0, depart=t)
        dates.append(bascules[0][1])
    assert max(dates) - min(dates) < 1.5


def test_le_lead_ne_clignote_pas():
    """Une densite qui oscille autour du seuil ne doit pas faire osciller le lead."""
    conv = conversation()
    _, t = jouer(conv, 0.30, 12.0)
    bascules = []
    for cycle in range(12):
        bas, t = jouer(conv, 0.05, 1.0, depart=t)
        haut, t = jouer(conv, 0.30, 1.0, depart=t)
        bascules += bas + haut
    assert len(bascules) <= 1


def test_l_appli_ne_prend_pas_la_main_au_demarrage():
    """La presence part de zero : sans garde-fou, la place semblerait libre."""
    conv = conversation(agency=0.9)
    bascules, _ = jouer(conv, 0.35, 3.0)
    assert bascules == []
    assert conv.lead == PLAYER


def test_les_extremes_du_curseur_figent_le_lead():
    accompagnateur = conversation(agency=0.0)
    _, t = jouer(accompagnateur, 0.3, 10.0)
    bascules, _ = jouer(accompagnateur, 0.0, 40.0, depart=t)
    assert bascules == [] and accompagnateur.lead == PLAYER

    meneur = conversation(agency=1.0)
    bascules, t = jouer(meneur, 0.0, 10.0)
    assert meneur.leads
    bascules, _ = jouer(meneur, 0.9, 30.0, depart=t)
    assert bascules == [] and meneur.leads


def test_un_partenaire_entreprenant_prend_la_main_plus_tot():
    dates = []
    for agency in (0.2, 0.5, 0.9):
        conv = conversation(agency=agency)
        _, t = jouer(conv, 0.35, 14.0)
        bascules, _ = jouer(conv, 0.0, 20.0, depart=t)
        dates.append(bascules[0][1])
    assert dates == sorted(dates, reverse=True)


# --- ce qui en decoule -------------------------------------------------


def test_la_percu_n_abandonne_pas_quand_on_respire():
    """Le symptome de depart : la couche rythmique s'eteignait au premier blanc.

    Elle est censee motiver le jeu ; si elle disparait des qu'on respire, elle
    motive un silence.
    """
    conv = conversation()
    from halj.config import RhythmConfig

    gate = RhythmConfig().gate_density
    assert conv.density(0.0) > gate  # meme sans rien jouer, le fond tient
    assert conv.density(0.0) == pytest.approx(conv.config.follow_density)
    # Et ce qui est joue plus fort que le plancher passe tel quel.
    assert conv.density(0.8) == pytest.approx(0.8)


def test_l_appli_releve_le_rythme_quand_elle_mene():
    conv = conversation()
    accompagnement = conv.density(0.0)
    conv.lead = APP
    assert conv.density(0.0) > accompagnement * 2


def test_l_initiative_harmonique_suit_le_tour():
    conv = conversation(agency=0.5)
    assert conv.agency() < 0.5  # en accompagnement, elle se laisse detourner
    conv.lead = APP
    assert conv.agency() > 0.5  # a son tour, elle tient sa grille


def test_sans_conversation_rien_ne_bouge():
    """`--no-conversation` doit rendre exactement le comportement precedent."""
    conv = conversation(agency=0.7, enabled=False)
    _, t = jouer(conv, 0.0, 40.0)
    assert conv.lead == PLAYER
    assert conv.agency() == pytest.approx(0.7)
    assert conv.density(0.02) == pytest.approx(0.02)


# --- la phrase de reponse ----------------------------------------------


def test_l_appli_ne_repond_que_quand_c_est_son_tour():
    conv = conversation()
    assert not conv.wants_response()
    conv.lead = APP
    assert conv.wants_response()


def test_l_appli_ne_repond_pas_a_toutes_les_mesures():
    """Une phrase a chaque mesure n'est plus une reponse, c'est du bavardage."""
    conv = conversation()
    conv.lead = APP
    reponses = [conv.wants_response() for _ in range(8)]
    assert reponses == [True, False] * 4


def test_la_phrase_tient_dans_l_accord():
    for chord in (Chord(9, "min"), Chord(5, "maj"), Chord(7, "sus4")):
        for bar in range(4):
            notes = response_phrase(chord, bar)
            assert len(notes) == 4
            for note in notes:
                assert (note.midi % 12) in chord.pitch_classes
                assert 0 <= note.step < 16
                assert 0.0 < note.velocity <= 1.0


def test_la_phrase_change_d_une_mesure_a_l_autre():
    chord = Chord(9, "min")
    phrases = {
        tuple((n.step, n.midi) for n in response_phrase(chord, bar))
        for bar in range(4)
    }
    assert len(phrases) == 4


def test_la_phrase_sonne_au_dessus_du_drone():
    """Registre median : c'est ce qu'un haut-parleur de telephone reproduit."""
    from halj.notes import midi_to_hz

    notes = response_phrase(Chord(9, "min"), 0, octave=4)
    frequences = [midi_to_hz(n.midi) for n in notes]
    assert min(frequences) > 400.0  # bien au-dessus du drone (55-123 Hz)
    assert max(frequences) < 2093.0  # et dans la bande analysee
