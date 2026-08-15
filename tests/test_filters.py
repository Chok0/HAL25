import numpy as np
import pytest

from halj.analysis.filters import Biquad, CascadedHighpass


def amplitude_a(filtre, freq_hz, samplerate=44100, duree_s=0.5):
    """Amplitude de sortie pour un sinus d'entree d'amplitude 1."""
    t = np.arange(int(duree_s * samplerate)) / samplerate
    sortie = filtre.process(np.sin(2 * np.pi * freq_hz * t))
    # on ignore le regime transitoire du debut
    return float(np.abs(sortie[samplerate // 10 :]).max())


def test_le_passe_haut_laisse_passer_l_aigu():
    filtre = Biquad.highpass(120.0, 44100)
    assert amplitude_a(filtre, 1000.0) == pytest.approx(1.0, abs=0.05)


def test_le_passe_haut_coupe_le_grave():
    filtre = Biquad.highpass(120.0, 44100)
    assert amplitude_a(filtre, 20.0) < 0.05


def test_la_coupure_est_a_moins_trois_db():
    filtre = Biquad.highpass(120.0, 44100)
    # -3 dB = 1/sqrt(2) : definition de la frequence de coupure.
    assert amplitude_a(filtre, 120.0) == pytest.approx(0.707, abs=0.05)


def test_la_cascade_coupe_plus_franchement():
    simple = Biquad.highpass(120.0, 44100)
    cascade = CascadedHighpass(120.0, 44100)
    assert amplitude_a(cascade, 40.0) < amplitude_a(simple, 40.0)


def test_la_cascade_preserve_la_bande_utile():
    # La fondamentale d'un mi grave de guitare est a 82 Hz, mais la detection
    # d'attaques travaille au-dessus de la coupure : c'est l'aigu qui compte.
    cascade = CascadedHighpass(120.0, 44100)
    assert amplitude_a(cascade, 800.0) == pytest.approx(1.0, abs=0.05)


def test_l_etat_est_conserve_entre_les_blocs():
    """Filtrer d'un bloc ou de plusieurs doit donner le meme signal."""
    signal = np.random.default_rng(0).normal(0, 1, 4096)

    entier = Biquad.highpass(120.0, 44100).process(signal)

    par_morceaux = Biquad.highpass(120.0, 44100)
    morceaux = np.concatenate(
        [par_morceaux.process(signal[start : start + 256]) for start in range(0, 4096, 256)]
    )
    assert np.allclose(entier, morceaux, atol=1e-9)


def test_reset_remet_l_etat_a_zero():
    filtre = Biquad.highpass(120.0, 44100)
    signal = np.random.default_rng(1).normal(0, 1, 512)
    premier = filtre.process(signal)
    filtre.reset()
    assert np.allclose(filtre.process(signal), premier)


def test_un_bloc_vide_ne_casse_rien():
    assert Biquad.highpass(120.0, 44100).process(np.zeros(0)).size == 0


def test_le_continu_est_supprime():
    filtre = CascadedHighpass(120.0, 44100)
    sortie = filtre.process(np.ones(8192))
    assert abs(sortie[-1]) < 1e-3


@pytest.mark.parametrize("cutoff", [0.0, -10.0, 22050.0, 30000.0])
def test_une_coupure_hors_bande_est_refusee(cutoff):
    with pytest.raises(ValueError):
        Biquad.highpass(cutoff, 44100)
