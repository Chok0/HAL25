import dataclasses

import pytest

from halj.config import RhythmConfig
from halj.generative.scheduler import RhythmScheduler


@pytest.fixture
def scheduler():
    return RhythmScheduler(RhythmConfig(bpm=120.0, steps=16))


def test_duree_du_pas(scheduler):
    # 120 BPM, grille de 16 pas sur 4 temps -> double-croche = 125 ms.
    assert scheduler.step_duration == pytest.approx(0.125)


def test_rien_ne_sort_avant_l_heure(scheduler):
    scheduler.set_density(1.0)
    scheduler.advance(0.0)  # consomme le pas 0
    assert scheduler.advance(0.05) == []


def test_les_pas_sortent_dans_l_ordre(scheduler):
    scheduler.set_density(1.0)
    steps = [event.step for event in scheduler.advance(0.5)]
    assert steps == sorted(steps)


def test_une_densite_nulle_fait_taire_la_grille(scheduler):
    scheduler.set_density(0.0)
    assert scheduler.advance(4.0) == []


def test_le_seuil_de_silence_est_respecte():
    config = RhythmConfig(bpm=120.0, gate_density=0.5)
    scheduler = RhythmScheduler(config)
    scheduler.set_density(0.49)
    assert scheduler.advance(2.0) == []

    scheduler.reset()
    scheduler.set_density(0.51)
    assert scheduler.advance(2.0) != []


def test_plus_de_densite_donne_plus_d_impacts():
    counts = []
    for density in (0.2, 0.5, 0.9):
        scheduler = RhythmScheduler(RhythmConfig(bpm=120.0))
        scheduler.set_density(density)
        counts.append(len(scheduler.advance(4.0)))
    assert counts == sorted(counts)
    assert counts[0] < counts[-1]


def test_les_velocites_suivent_la_densite():
    faible = RhythmScheduler(RhythmConfig(bpm=120.0))
    faible.set_density(0.2)
    forte = RhythmScheduler(RhythmConfig(bpm=120.0))
    forte.set_density(1.0)
    assert max(e.velocity for e in forte.advance(2.0)) > max(
        e.velocity for e in faible.advance(2.0)
    )


def test_les_temps_forts_sont_accentues(scheduler):
    scheduler.set_density(1.0)
    events = scheduler.advance(2.0)
    forts = [e.velocity for e in events if e.step % 4 == 0]
    faibles = [e.velocity for e in events if e.step % 4 != 0]
    assert min(forts) > max(faibles)


def test_les_dates_tombent_sur_la_grille(scheduler):
    scheduler.set_density(1.0)
    for event in scheduler.advance(3.0):
        assert (event.time_s / scheduler.step_duration) % 1 == pytest.approx(
            0, abs=1e-9
        )


def test_les_deux_voix_sont_produites(scheduler):
    scheduler.set_density(0.9)
    voix = {event.voice for event in scheduler.advance(4.0)}
    assert voix == {"kick", "perc"}


def test_la_rotation_decale_les_percus():
    config = RhythmConfig(bpm=120.0, perc_rotation=3)
    scheduler = RhythmScheduler(config)
    scheduler.set_density(0.5)
    # Les rotations sont la pour que kick et percu ne tombent pas tous
    # ensemble : les deux motifs ne doivent pas etre identiques.
    assert scheduler.pattern("kick") != scheduler.pattern("perc")


def test_une_voix_inconnue_est_refusee(scheduler):
    with pytest.raises(ValueError):
        scheduler.pattern("trompette")


def test_le_retard_ne_se_rejoue_pas(scheduler):
    """Apres un gel du process, on se recale au lieu de vider une file."""
    scheduler.set_density(1.0)
    scheduler.advance(0.5)
    # 60 s plus tard : sans garde-fou, ce serait des centaines d'evenements
    # deverses d'un coup.
    rattrapage = scheduler.advance(60.0)
    assert len(rattrapage) <= 2 * scheduler.config.steps


def test_le_decoupage_de_l_horloge_n_influe_pas(scheduler):
    """Avancer par petits ou gros pas doit donner la meme sequence.

    Les deux cadences restent sous le garde-fou de rattrapage (une mesure),
    qui est teste separement.
    """
    scheduler.set_density(1.0)
    fin = [
        (event.time_s, event.voice, event.step)
        for centieme in range(1, 401)
        for event in scheduler.advance(centieme / 100)
    ]

    gros = RhythmScheduler(dataclasses.replace(scheduler.config))
    gros.set_density(1.0)
    grossier = [
        (event.time_s, event.voice, event.step)
        for cinquieme in range(1, 21)
        for event in gros.advance(cinquieme / 5)
    ]

    assert fin == grossier


def test_un_premier_appel_tres_tardif_se_recale(scheduler):
    """Une source qui demarre longtemps apres l'horloge repart du pas 0."""
    scheduler.set_density(1.0)
    evenements = scheduler.advance(60.0)
    assert [event.step for event in evenements] == [0, 0]  # kick + percu du pas 0


def test_reset_repart_du_premier_pas(scheduler):
    scheduler.set_density(1.0)
    scheduler.advance(2.0)
    assert scheduler.next_step > 0
    scheduler.reset()
    assert scheduler.next_step == 0
