import dataclasses

import numpy as np
import pytest

from halj.analysis.onset import OnsetDetector, create_onset_detector
from halj.config import Config

from .conftest import click_train


def detect(signal, config: Config, detector=None):
    """Fait defiler un signal dans le detecteur, rend les dates d'attaque."""
    detector = detector or OnsetDetector(config.audio, config.onset)
    hop = config.audio.hop_size
    onsets = []
    for start in range(0, signal.size - hop, hop):
        onset = detector.process(
            signal[start : start + hop], start / config.audio.samplerate
        )
        if onset is not None:
            onsets.append(onset)
    return onsets


def test_les_attaques_sont_trouvees_aux_bonnes_dates(config, samplerate):
    expected = [0.5, 1.0, 1.5, 2.0, 2.5]
    signal = click_train(expected, samplerate, duration_s=3.0)

    detected = [onset.time_s for onset in detect(signal, config)]

    assert len(detected) == len(expected)
    for found, target in zip(detected, expected):
        # tolerance = une trame d'analyse : le detecteur confirme le pic sur la
        # trame suivante, donc un retard d'un hop est attendu par construction.
        assert found == pytest.approx(target, abs=0.03)


def test_le_silence_ne_produit_aucune_attaque(config, samplerate):
    assert detect(np.zeros(2 * samplerate), config) == []


def test_un_bruit_de_fond_constant_ne_produit_pas_d_attaque(config, samplerate):
    rng = np.random.default_rng(11)
    t = np.arange(2 * samplerate) / samplerate
    # Fondu d'entree : un souffle qui apparait d'un coup est un vrai transitoire,
    # ce n'est pas ce qu'on teste ici. Une fois etabli, il ne doit plus rien
    # declencher — c'est au seuil adaptatif de s'y fondre.
    souffle = rng.normal(0, 0.002, 2 * samplerate) * np.clip(t / 0.5, 0, 1)
    tardifs = [onset for onset in detect(souffle, config) if onset.time_s > 0.6]
    assert tardifs == []


def test_une_note_tenue_ne_declenche_qu_une_fois(config, samplerate):
    t = np.arange(int(1.5 * samplerate)) / samplerate
    # attaque nette puis tenue longue, sans nouvelle attaque
    signal = np.sin(2 * np.pi * 196 * t) * np.minimum(1.0, t * 200) * 0.5
    assert len(detect(signal, config)) == 1


def test_le_temps_mort_empeche_les_doublons(config, samplerate):
    # Deux impulsions separees de 10 ms, sous le temps mort de 45 ms.
    signal = click_train([0.5, 0.51], samplerate, duration_s=1.5)
    assert len(detect(signal, config)) == 1


def test_le_temps_mort_laisse_passer_un_jeu_rapide(config, samplerate):
    # 12 attaques a 120 ms d'intervalle : bien au-dessus du temps mort.
    expected = [0.3 + index * 0.12 for index in range(12)]
    signal = click_train(expected, samplerate, duration_s=2.5)
    assert len(detect(signal, config)) == len(expected)


def test_un_accord_gratte_est_detecte(config, guitar_chord):
    onsets = detect(guitar_chord, config)
    # Un grattage etale 5 cordes sur ~70 ms : avec un temps mort de 45 ms,
    # une a deux attaques sont attendues, pas une par corde.
    assert 1 <= len(onsets) <= 3
    assert onsets[0].time_s < 0.05


def test_le_seuil_s_adapte_au_niveau_de_jeu(config, samplerate):
    doux = click_train([0.5, 1.0, 1.5], samplerate, duration_s=2.0, amplitude=0.05)
    fort = click_train([0.5, 1.0, 1.5], samplerate, duration_s=2.0, amplitude=0.9)
    # Le seuil etant relatif a la mediane locale, le compte ne doit pas
    # dependre du niveau absolu.
    assert len(detect(doux, config)) == len(detect(fort, config)) == 3


def test_le_passe_haut_ignore_les_rumbles(config, samplerate):
    rng = np.random.default_rng(5)
    t = np.arange(int(2.5 * samplerate)) / samplerate
    # Bruit de fond permanent : sans lui, l'apparition du moindre signal apres
    # un silence numerique absolu est un vrai transitoire, et c'est lui qu'on
    # mesurerait au lieu du contenu grave.
    fond = rng.normal(0, 1e-4, t.size)
    # 40 Hz fort, sous le passe-haut a 120 Hz : pied de micro qui vibre, porte
    # qui claque a l'etage. Doit rester totalement invisible.
    rumble = fond + 0.8 * np.sin(2 * np.pi * 40 * t) * np.clip((t - 0.5) / 0.4, 0, 1)

    tardifs = [onset for onset in detect(rumble, config) if onset.time_s > 0.2]
    assert tardifs == []


def test_la_force_est_relative_au_seuil(config, samplerate):
    signal = click_train([0.5], samplerate, duration_s=1.5, amplitude=0.9)
    onsets = detect(signal, config)
    assert onsets[0].strength > 1.0


def test_un_hop_de_mauvaise_taille_est_refuse(config):
    detector = OnsetDetector(config.audio, config.onset)
    with pytest.raises(ValueError):
        detector.process(np.zeros(config.audio.hop_size + 1))


def test_reset_remet_le_detecteur_a_zero(config, samplerate):
    detector = OnsetDetector(config.audio, config.onset)
    signal = click_train([0.5, 1.0], samplerate, duration_s=1.5)
    first = detect(signal, config, detector)
    detector.reset()
    assert [o.time_s for o in detect(signal, config, detector)] == [
        o.time_s for o in first
    ]


def test_un_passe_haut_hors_bande_est_refuse(config):
    onset = dataclasses.replace(config.onset, highpass_hz=50_000.0)
    with pytest.raises(ValueError):
        OnsetDetector(config.audio, onset)


def test_backend_inconnu(config):
    with pytest.raises(ValueError):
        create_onset_detector(config.audio, config.onset, backend="magique")


def test_backend_aubio_si_disponible(config, samplerate):
    aubio = pytest.importorskip("aubio")
    del aubio
    detector = create_onset_detector(config.audio, config.onset, backend="aubio")
    signal = click_train([0.5, 1.0, 1.5], samplerate, duration_s=2.0)
    onsets = detect(signal, config, detector)
    # Fonction de detection differente : on verifie l'ordre de grandeur et le
    # respect de l'interface, pas l'egalite stricte avec le backend NumPy.
    assert 2 <= len(onsets) <= 5
