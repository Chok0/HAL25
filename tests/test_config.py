import dataclasses
import json

import pytest

from halj.config import AudioConfig, Config, KeyConfig, RhythmConfig
from halj.notes import Key, hz_to_midi, midi_to_hz, pitch_class_to_hz


def test_la_config_par_defaut_est_coherente():
    config = Config()
    # Le hop doit diviser le bloc du driver, sinon des echantillons restent
    # en attente a chaque callback.
    assert config.audio.block_size % config.audio.hop_size == 0
    # Les fenetres d'analyse doivent couvrir au moins un hop.
    assert config.key.frame_size >= config.audio.hop_size
    assert config.onset.frame_size >= config.audio.hop_size
    # La fenetre du chroma doit rester sous Nyquist.
    assert config.key.fmax_hz < config.audio.samplerate / 2


def test_hop_en_secondes():
    config = Config(audio=AudioConfig(samplerate=44100, hop_size=441))
    assert config.hop_s == pytest.approx(0.01)


def test_aller_retour_dictionnaire():
    config = Config(rhythm=RhythmConfig(bpm=97.0, steps=12))
    assert Config.from_dict(config.to_dict()) == config


def test_chargement_depuis_un_fichier(tmp_path):
    chemin = tmp_path / "config.json"
    chemin.write_text(json.dumps({"rhythm": {"bpm": 132.0}}), encoding="utf-8")
    config = Config.load(chemin)
    assert config.rhythm.bpm == 132.0
    # les sections absentes gardent leurs valeurs par defaut
    assert config.audio.samplerate == 44100


def test_chargement_sans_fichier():
    assert Config.load(None) == Config()


def test_une_section_inconnue_est_signalee():
    with pytest.raises(ValueError, match="inconnues"):
        Config.from_dict({"reverb": {"mix": 0.5}})


def test_surcharge_de_section():
    config = Config()
    modifie = config.with_overrides(rhythm=dataclasses.replace(config.rhythm, bpm=100.0))
    assert modifie.rhythm.bpm == 100.0
    assert config.rhythm.bpm == 84.0  # l'original est intact (frozen)


def test_la_config_est_immuable():
    with pytest.raises(dataclasses.FrozenInstanceError):
        Config().audio.samplerate = 48000


# --- notes -----------------------------------------------------------


def test_la_reference_est_le_la_440():
    assert midi_to_hz(69) == pytest.approx(440.0)
    assert hz_to_midi(440.0) == pytest.approx(69.0)


def test_aller_retour_hauteur():
    for midi in range(21, 108):
        assert hz_to_midi(midi_to_hz(midi)) == pytest.approx(midi)


def test_octave_scientifique():
    assert pitch_class_to_hz(0, 4) == pytest.approx(261.63, abs=0.01)  # do4
    assert pitch_class_to_hz(9, 4) == pytest.approx(440.0)  # la4


def test_une_frequence_nulle_est_refusee():
    with pytest.raises(ValueError):
        hz_to_midi(0.0)


def test_nom_de_tonalite():
    assert Key(9, "min").name == "A min"
    assert Key(0, "maj").name == "C maj"


def test_fondamentale_de_la_tonalite():
    assert Key(9, "min").root_hz(octave=2) == pytest.approx(110.0, abs=0.01)


@pytest.mark.parametrize("tonic, mode", [(12, "maj"), (-1, "min"), (0, "dorien")])
def test_tonalite_invalide(tonic, mode):
    with pytest.raises(ValueError):
        Key(tonic, mode)


def test_les_tonalites_se_comparent():
    assert Key(3, "maj") == Key(3, "maj")
    assert Key(3, "maj") != Key(3, "min")
    assert len({Key(3, "maj"), Key(3, "maj")}) == 1
