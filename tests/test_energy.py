import numpy as np
import pytest

from halj.analysis.energy import EnergyTracker, rms
from halj.config import EnergyConfig


@pytest.fixture
def tracker():
    return EnergyTracker(EnergyConfig(), hop_s=0.005)


def test_rms_d_un_sinus():
    t = np.linspace(0, 1, 44100, endpoint=False)
    assert rms(np.sin(2 * np.pi * 100 * t)) == pytest.approx(1 / np.sqrt(2), abs=1e-3)


def test_rms_du_vide():
    assert rms(np.zeros(0)) == 0.0
    assert rms(np.zeros(128)) == 0.0


def test_le_silence_donne_une_densite_nulle(tracker):
    for _ in range(200):
        tracker.update(0.0)
    assert tracker.density == pytest.approx(0.0, abs=1e-6)


def test_un_jeu_fort_sature_la_densite(tracker):
    for _ in range(400):
        tracker.update(0.5, onset_count=1)
    assert tracker.density > 0.9


def test_la_densite_reste_bornee(tracker):
    for _ in range(500):
        tracker.update(10.0, onset_count=5)
    assert 0.0 <= tracker.density <= 1.0


def test_la_montee_est_plus_rapide_que_la_descente(tracker):
    # Constantes asymetriques : le rythme doit relancer vite et retomber
    # doucement, pour ne pas hacher une phrase qui respire.
    for _ in range(40):
        tracker.update(0.3)
    monte = tracker.level
    for _ in range(40):
        tracker.update(0.0)
    descendu = tracker.level
    assert monte > 0.5
    assert descendu > monte * 0.5


def test_le_debit_d_attaques_augmente_la_densite():
    calme = EnergyTracker(EnergyConfig(), hop_s=0.005)
    agite = EnergyTracker(EnergyConfig(), hop_s=0.005)
    for index in range(400):
        calme.update(0.05)
        # meme niveau sonore, mais une attaque toutes les 20 trames
        agite.update(0.05, onset_count=1 if index % 20 == 0 else 0)
    assert agite.density > calme.density


def test_le_debit_d_attaques_retombe_sans_jeu(tracker):
    for _ in range(50):
        tracker.update(0.2, onset_count=1)
    haut = tracker.onset_rate
    for _ in range(600):
        tracker.update(0.2)
    assert tracker.onset_rate < haut * 0.2


def test_la_densite_est_monotone_avec_le_niveau():
    densites = []
    for amplitude in (0.001, 0.01, 0.05, 0.2, 0.8):
        tracker = EnergyTracker(EnergyConfig(), hop_s=0.005)
        for _ in range(400):
            tracker.update(amplitude)
        densites.append(tracker.density)
    assert densites == sorted(densites)


def test_reset(tracker):
    for _ in range(100):
        tracker.update(0.4, onset_count=1)
    tracker.reset()
    assert tracker.level == 0.0
    assert tracker.onset_rate == 0.0


def test_une_plage_db_invalide_est_refusee():
    tracker = EnergyTracker(EnergyConfig(floor_db=-10.0, ceil_db=-40.0), hop_s=0.005)
    with pytest.raises(ValueError):
        tracker.update(0.1)
