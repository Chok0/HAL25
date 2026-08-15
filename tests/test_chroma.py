import dataclasses

import numpy as np
import pytest

from halj.analysis.chroma import ChromaExtractor
from halj.analysis.key import estimate_key
from halj.audio.synth import MAJOR_TRIAD, MINOR_TRIAD, karplus_strong, strum
from halj.notes import midi_to_hz


def chroma_of(extractor, signal):
    """Chroma moyen d'un signal, fenetre par fenetre."""
    frame = extractor.frame_size
    vectors = []
    for start in range(0, max(1, signal.size - frame), frame // 2):
        segment = signal[start : start + frame]
        if segment.size < frame:
            break
        vector = extractor.process(segment)
        if vector.any():
            vectors.append(vector)
    return np.mean(vectors, axis=0) if vectors else np.zeros(12)


@pytest.fixture
def extractor(config):
    return ChromaExtractor(config.audio, config.key)


@pytest.mark.parametrize("pitch_class", range(12))
def test_une_note_pointe_sa_propre_classe(extractor, samplerate, pitch_class):
    midi = 48 + pitch_class  # a partir de do3
    signal = karplus_strong(midi_to_hz(midi), 1.5, samplerate, rng=np.random.default_rng(1))
    chroma = chroma_of(extractor, signal)
    assert int(np.argmax(chroma)) == pitch_class % 12


def test_l_octave_est_repliee(extractor, samplerate):
    # Meme note a deux octaves : le chroma doit etre quasiment identique,
    # c'est toute la raison d'utiliser un chroma plutot qu'un spectre.
    basse = karplus_strong(midi_to_hz(45), 1.5, samplerate, rng=np.random.default_rng(2))
    aigue = karplus_strong(midi_to_hz(57), 1.5, samplerate, rng=np.random.default_rng(2))
    assert int(np.argmax(chroma_of(extractor, basse))) == int(
        np.argmax(chroma_of(extractor, aigue))
    )


def test_le_chroma_est_normalise(extractor, guitar_chord):
    chroma = extractor.process(guitar_chord[: extractor.frame_size])
    assert chroma.shape == (12,)
    assert chroma.sum() == pytest.approx(1.0)
    assert (chroma >= 0).all()


def test_le_silence_donne_un_chroma_nul(extractor):
    assert not extractor.process(np.zeros(extractor.frame_size)).any()


def test_une_trame_de_mauvaise_taille_est_refusee(extractor):
    with pytest.raises(ValueError):
        extractor.process(np.zeros(extractor.frame_size // 2))


def test_les_trois_notes_de_l_accord_ressortent(extractor, samplerate):
    # La mineur : la, do, mi -> classes 9, 0, 4.
    signal = strum(45, MINOR_TRIAD, 2.0, samplerate, rng=np.random.default_rng(3))
    chroma = chroma_of(extractor, signal)
    trois_premieres = set(np.argsort(chroma)[-3:].tolist())
    assert trois_premieres == {9, 0, 4}


def test_une_bande_vide_est_refusee(config):
    # Au-dessus de Nyquist : aucun bin FFT ne tombe dans la bande.
    key = dataclasses.replace(config.key, fmin_hz=23000.0, fmax_hz=24000.0)
    with pytest.raises(ValueError):
        ChromaExtractor(config.audio, key)


def test_backend_inconnu(config):
    with pytest.raises(ValueError):
        ChromaExtractor.create(config.audio, config.key, backend="magique")


# --- banc de mesure --------------------------------------------------


def banc_accords(samplerate):
    """24 accords guitare : les 12 majeurs et les 12 mineurs."""
    for pitch_class in range(12):
        root = 48 + pitch_class if pitch_class < 8 else 36 + pitch_class
        yield strum(
            root, MAJOR_TRIAD, 2.0, samplerate, rng=np.random.default_rng(pitch_class)
        ), pitch_class, "maj"
        yield strum(
            root,
            MINOR_TRIAD,
            2.0,
            samplerate,
            rng=np.random.default_rng(pitch_class + 50),
        ), pitch_class, "min"


def banc_lignes(samplerate):
    """24 lignes monodiques : une gamme montante puis un arpege redescendant."""
    majeure = [0, 2, 4, 5, 7, 9, 11, 12, 7, 4, 0]
    mineure = [0, 2, 3, 5, 7, 8, 10, 12, 7, 3, 0]
    for pitch_class in range(12):
        root = 48 + pitch_class if pitch_class < 8 else 36 + pitch_class
        for intervals, mode, seed in (
            (majeure, "maj", pitch_class),
            (mineure, "min", pitch_class + 70),
        ):
            rng = np.random.default_rng(seed)
            yield np.concatenate(
                [
                    karplus_strong(midi_to_hz(root + i), 0.45, samplerate, rng=rng)
                    for i in intervals
                ]
            ), pitch_class, mode


def taux_de_reussite(extractor, banc):
    reussites = 0
    total = 0
    for signal, pitch_class, mode in banc:
        estimate = estimate_key(chroma_of(extractor, signal))
        total += 1
        if estimate and estimate.key.tonic == pitch_class and estimate.key.mode == mode:
            reussites += 1
    return reussites, total


def test_precision_sur_les_accords(extractor, samplerate):
    reussites, total = taux_de_reussite(extractor, banc_accords(samplerate))
    # Reference mesuree : 23/24. Le seuil laisse la place au bruit de la
    # synthese sans laisser passer une regression franche.
    assert reussites >= 21, f"{reussites}/{total} accords reconnus"


def test_precision_sur_les_lignes_monodiques(extractor, samplerate):
    # Le point dur du cadrage : un seul pipeline chroma doit gerer les notes
    # seules aussi bien que les accords, sans detecter le mode de jeu.
    reussites, total = taux_de_reussite(extractor, banc_lignes(samplerate))
    assert reussites >= 22, f"{reussites}/{total} lignes reconnues"


def test_la_compensation_spectrale_est_ce_qui_fait_la_difference(config, samplerate):
    """Garde-fou : sans compensation 1/f, la tierce prend le dessus.

    Ce test documente *pourquoi* `spectral_tilt` existe — s'il devenait
    inutile, c'est que le chroma aurait change de nature.
    """
    sans = ChromaExtractor(
        config.audio, dataclasses.replace(config.key, spectral_tilt=0.0)
    )
    avec = ChromaExtractor(config.audio, config.key)

    reussites_sans, _ = taux_de_reussite(sans, banc_accords(samplerate))
    reussites_avec, _ = taux_de_reussite(avec, banc_accords(samplerate))
    assert reussites_avec > reussites_sans + 8


def test_le_backend_librosa_donne_un_chroma_coherent(config, samplerate):
    """Le backend CQT doit designer les memes classes que le backend NumPy.

    Trop lent pour le direct (~x0.2 du temps reel), il sert a arbitrer la
    qualite hors ligne : encore faut-il qu'il soit d'accord sur l'essentiel.
    """
    pytest.importorskip("librosa")
    signal = strum(45, MINOR_TRIAD, 2.0, samplerate, rng=np.random.default_rng(3))

    numpy_chroma = chroma_of(ChromaExtractor(config.audio, config.key), signal)
    librosa_chroma = chroma_of(
        ChromaExtractor.create(config.audio, config.key, backend="librosa"), signal
    )

    assert librosa_chroma.sum() == pytest.approx(1.0)
    # meme classe dominante : la mineur -> la
    assert int(np.argmax(librosa_chroma)) == int(np.argmax(numpy_chroma)) == 9
