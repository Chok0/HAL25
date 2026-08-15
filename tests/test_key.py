import numpy as np
import pytest

from halj.analysis.key import (
    KS_MAJOR,
    KS_MINOR,
    KeyTracker,
    correlate_profiles,
    estimate_key,
)
from halj.config import KeyConfig
from halj.notes import Key


def chroma_for(profile, tonic):
    """Chroma ideal d'une tonalite : le profil K-S lui-meme, transpose."""
    rolled = np.roll(profile, tonic)
    return rolled / rolled.sum()


@pytest.mark.parametrize("tonic", range(12))
def test_les_profils_ideaux_sont_reconnus_en_majeur(tonic):
    estimate = estimate_key(chroma_for(KS_MAJOR, tonic))
    assert estimate.key == Key(tonic, "maj")
    assert estimate.confidence > 0.95


@pytest.mark.parametrize("tonic", range(12))
def test_les_profils_ideaux_sont_reconnus_en_mineur(tonic):
    estimate = estimate_key(chroma_for(KS_MINOR, tonic))
    assert estimate.key == Key(tonic, "min")
    assert estimate.confidence > 0.95


def test_un_accord_parfait_majeur_donne_sa_tonalite():
    chroma = np.zeros(12)
    chroma[[0, 4, 7]] = [0.4, 0.3, 0.3]  # do - mi - sol
    assert estimate_key(chroma).key == Key(0, "maj")


def test_un_accord_parfait_mineur_donne_sa_tonalite():
    chroma = np.zeros(12)
    chroma[[9, 0, 4]] = [0.4, 0.3, 0.3]  # la - do - mi
    assert estimate_key(chroma).key == Key(9, "min")


def test_un_chroma_vide_ne_donne_aucune_estimation():
    assert estimate_key(np.zeros(12)) is None


def test_un_chroma_plat_reste_indecis():
    # Toutes les classes a egalite : aucune correlation ne se detache.
    estimate = estimate_key(np.full(12, 1 / 12))
    assert estimate is None or estimate.confidence < 0.2


def test_le_chroma_doit_avoir_douze_classes():
    with pytest.raises(ValueError):
        correlate_profiles(np.zeros(7))


def test_transposer_le_chroma_transpose_la_tonalite():
    base = chroma_for(KS_MAJOR, 0)
    for shift in range(12):
        assert estimate_key(np.roll(base, shift)).key == Key(shift, "maj")


# --- hysterese -------------------------------------------------------


def make_tracker(**overrides):
    config = KeyConfig(**{"hold_frames": 5, "switch_margin": 0.05, **overrides})
    return KeyTracker(config, hop_s=0.01)


def feed(tracker, chroma, frames, rms=0.5):
    for _ in range(frames):
        tracker.update(chroma, rms)


def test_la_premiere_tonalite_fiable_est_adoptee_immediatement():
    tracker = make_tracker()
    feed(tracker, chroma_for(KS_MAJOR, 0), tracker.window_frames)
    assert tracker.key == Key(0, "maj")


def test_une_tonalite_concurrente_breve_ne_fait_pas_basculer():
    tracker = make_tracker(hold_frames=20)
    feed(tracker, chroma_for(KS_MAJOR, 0), 40)
    established = tracker.key

    # Deux trames d'une autre tonalite : trop court pour la marge de maintien.
    feed(tracker, chroma_for(KS_MAJOR, 6), 2)
    assert tracker.key == established


def test_une_tonalite_concurrente_soutenue_finit_par_l_emporter():
    tracker = make_tracker(hold_frames=5)
    feed(tracker, chroma_for(KS_MAJOR, 0), 60)
    assert tracker.key == Key(0, "maj")

    feed(tracker, chroma_for(KS_MAJOR, 5), 200)
    assert tracker.key == Key(5, "maj")


def test_le_silence_ne_modifie_pas_la_tonalite():
    tracker = make_tracker()
    feed(tracker, chroma_for(KS_MAJOR, 2), 40)
    established = tracker.key

    # Sous le seuil de silence : la fenetre ne doit pas etre nourrie du tout.
    feed(tracker, chroma_for(KS_MAJOR, 8), 500, rms=0.0)
    assert tracker.key == established


def test_un_chroma_incertain_ne_declenche_pas_de_changement():
    tracker = make_tracker(min_confidence=0.9)
    feed(tracker, chroma_for(KS_MAJOR, 0), 40)
    assert tracker.key == Key(0, "maj")

    feed(tracker, np.full(12, 1 / 12) + np.linspace(0, 0.001, 12), 100)
    assert tracker.key == Key(0, "maj")


def test_reset_efface_l_etat():
    tracker = make_tracker()
    feed(tracker, chroma_for(KS_MAJOR, 3), 40)
    assert tracker.key is not None
    tracker.reset()
    assert tracker.key is None
    assert tracker.confidence == 0.0
