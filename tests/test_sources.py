import wave

import numpy as np
import pytest

from halj.audio.sources import SyntheticSource, WavFileSource, create_source
from halj.audio.synth import MAJOR_TRIAD, karplus_strong, strum
from halj.notes import hz_to_midi, midi_to_hz


def ecrire_wav(path, samples, samplerate, channels=1):
    data = np.clip(samples, -1, 1)
    if channels > 1:
        data = np.repeat(data[:, None], channels, axis=1).reshape(-1)
    pcm = (data * 32767).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(samplerate)
        handle.writeframes(pcm.tobytes())


# --- generateur ------------------------------------------------------


def test_une_corde_pincee_sonne_a_sa_hauteur(samplerate):
    """Le partiel dominant tombe sur une harmonique de la note visee.

    Ce n'est pas forcement la fondamentale : comme sur une vraie corde, la
    repartition initiale de l'energie entre partiels est aleatoire et l'octave
    l'emporte souvent. Ce qui compte en aval, c'est que la classe de hauteur
    soit la bonne — le chroma replie les octaves de toute facon.
    """
    signal = karplus_strong(220.0, 1.0, samplerate, rng=np.random.default_rng(0))
    spectre = np.abs(np.fft.rfft(signal * np.hanning(signal.size)))
    freqs = np.fft.rfftfreq(signal.size, 1 / samplerate)
    dominante = freqs[int(np.argmax(spectre))]

    ecart_en_demi_tons = hz_to_midi(dominante) - hz_to_midi(220.0)
    assert ecart_en_demi_tons % 12 == pytest.approx(0.0, abs=0.5)


def test_une_corde_pincee_a_de_l_energie_sur_sa_fondamentale(samplerate):
    signal = karplus_strong(220.0, 1.0, samplerate, rng=np.random.default_rng(0))
    spectre = np.abs(np.fft.rfft(signal * np.hanning(signal.size)))
    freqs = np.fft.rfftfreq(signal.size, 1 / samplerate)

    autour_de_f0 = (freqs > 210) & (freqs < 232)
    assert spectre[autour_de_f0].max() > 0.05 * spectre.max()


def test_une_corde_pincee_decroit(samplerate):
    signal = karplus_strong(220.0, 2.0, samplerate, rng=np.random.default_rng(0))
    debut = np.abs(signal[: samplerate // 10]).max()
    fin = np.abs(signal[-samplerate // 10 :]).max()
    assert fin < debut / 2


def test_une_frequence_invalide_est_refusee(samplerate):
    with pytest.raises(ValueError):
        karplus_strong(0.0, 1.0, samplerate)


def test_une_duree_nulle_donne_un_signal_vide(samplerate):
    assert karplus_strong(220.0, 0.0, samplerate).size == 0


def test_le_grattage_etale_les_cordes(samplerate):
    accord = strum(48, MAJOR_TRIAD, 1.0, samplerate, spread_s=0.02)
    # Les cordes entrent l'une apres l'autre : l'energie monte progressivement
    # sur les premieres dizaines de millisecondes.
    premiere = np.abs(accord[: int(0.005 * samplerate)]).max()
    apres = np.abs(accord[int(0.05 * samplerate) : int(0.06 * samplerate)]).max()
    assert apres > premiere


def test_le_signal_de_test_reste_dans_les_bornes(samplerate):
    accord = strum(48, MAJOR_TRIAD, 1.5, samplerate)
    assert np.abs(accord).max() <= 1.0


def test_le_generateur_est_deterministe(samplerate):
    premier = strum(48, MAJOR_TRIAD, 0.5, samplerate, rng=np.random.default_rng(9))
    second = strum(48, MAJOR_TRIAD, 0.5, samplerate, rng=np.random.default_rng(9))
    assert np.array_equal(premier, second)


# --- sources ---------------------------------------------------------


def test_la_source_synthetique_boucle(config):
    source = SyntheticSource(config.audio, loop=True, chord_duration_s=0.3)
    blocs = []
    for bloc in source.iter_blocks():
        blocs.append(bloc)
        if len(blocs) > source.audio.size // config.audio.block_size + 5:
            break
    # une source en boucle ne s'arrete jamais d'elle-meme
    assert len(blocs) > source.audio.size // config.audio.block_size


def test_la_source_synthetique_s_arrete_si_demande(config):
    source = SyntheticSource(config.audio, loop=False, chord_duration_s=0.3)
    total = sum(bloc.size for bloc in source.iter_blocks())
    assert total == source.audio.size


def test_lecture_d_un_wav(tmp_path, config, samplerate):
    signal = strum(48, MAJOR_TRIAD, 0.5, samplerate)
    chemin = tmp_path / "prise.wav"
    ecrire_wav(chemin, signal, samplerate)

    source = WavFileSource(str(chemin), config.audio)
    lu = np.concatenate(list(source.iter_blocks()))

    assert lu.size == signal.size
    assert np.allclose(lu, signal, atol=1e-3)  # quantification 16 bits


def test_un_wav_stereo_est_replie_en_mono(tmp_path, config, samplerate):
    signal = strum(48, MAJOR_TRIAD, 0.3, samplerate)
    chemin = tmp_path / "stereo.wav"
    ecrire_wav(chemin, signal, samplerate, channels=2)

    lu = np.concatenate(list(WavFileSource(str(chemin), config.audio).iter_blocks()))
    assert lu.size == signal.size


def test_un_wav_mal_echantillonne_est_refuse(tmp_path, config):
    chemin = tmp_path / "mauvais.wav"
    ecrire_wav(chemin, np.zeros(1000), 22050)
    with pytest.raises(ValueError, match="22050"):
        WavFileSource(str(chemin), config.audio)


def test_create_source_sans_fichier(config):
    with pytest.raises(ValueError, match="--input"):
        create_source("file", config.audio)


def test_create_source_inconnue(config):
    with pytest.raises(ValueError):
        create_source("theremine", config.audio)


def test_create_source_synthetique(config):
    assert isinstance(create_source("synth", config.audio), SyntheticSource)
