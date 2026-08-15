"""Tests de bout en bout : du signal audio a l'etat musical."""

import numpy as np
import pytest

from halj.analysis.engine import AnalysisEngine
from halj.audio.synth import MINOR_TRIAD, chord_progression, strum
from halj.bridge.osc import JamBridge, RecordingSink
from halj.config import Config, DroneConfig, OscConfig
from halj.notes import Key
from halj.runtime import JamSession, analyze_offline


@pytest.fixture
def engine(config):
    return AnalysisEngine(config)


def test_une_trame_par_hop(engine, config):
    frames = engine.process_block(np.zeros(config.audio.hop_size * 10))
    assert len(frames) == 10


def test_le_reliquat_est_conserve_entre_les_blocs(engine, config):
    hop = config.audio.hop_size
    assert engine.process_block(np.zeros(hop - 1)) == []
    # le bloc suivant complete le hop precedent
    assert len(engine.process_block(np.zeros(hop + 1))) == 2


def test_le_decoupage_en_blocs_n_influe_pas_sur_le_resultat(config, guitar_chord):
    reference = AnalysisEngine(config).process_block(guitar_chord)

    par_morceaux = AnalysisEngine(config)
    frames = []
    for start in range(0, guitar_chord.size, 997):  # taille volontairement bizarre
        frames.extend(par_morceaux.process_block(guitar_chord[start : start + 997]))

    assert len(frames) == len(reference)
    assert [f.onset is not None for f in frames] == [
        f.onset is not None for f in reference
    ]
    assert [f.key for f in frames] == [f.key for f in reference]


def test_les_dates_avancent_d_un_hop(engine, config):
    frames = engine.process_block(np.zeros(config.audio.hop_size * 5))
    dates = [frame.time_s for frame in frames]
    assert dates == pytest.approx([index * config.hop_s for index in range(5)])


def test_le_chroma_est_decime(engine, config):
    frames = engine.process_block(
        np.zeros(config.audio.hop_size * config.key.decimation * 4)
    )
    calcules = [frame for frame in frames if frame.chroma is not None]
    assert len(calcules) == 4


def test_le_silence_ne_produit_ni_tonalite_ni_attaque(engine, samplerate):
    frames = engine.process_block(np.zeros(samplerate))
    assert all(frame.key is None for frame in frames)
    assert all(frame.onset is None for frame in frames)
    assert all(frame.density == 0.0 for frame in frames)


def test_un_accord_mineur_est_reconnu(config, guitar_chord):
    frames = AnalysisEngine(config).process_block(np.tile(guitar_chord, 2))
    tonalites = [frame.key for frame in frames if frame.key]
    assert tonalites, "aucune tonalite detectee"
    assert tonalites[-1] == Key(9, "min")  # la mineur


def test_un_accord_majeur_est_reconnu(config, guitar_major_chord):
    frames = AnalysisEngine(config).process_block(np.tile(guitar_major_chord, 2))
    tonalites = [frame.key for frame in frames if frame.key]
    assert tonalites[-1] == Key(0, "maj")  # do majeur


def test_une_progression_donne_sa_tonique(config, samplerate):
    # La mineur - Fa - Do - Sol : la tonique doit dominer le decompte.
    audio = chord_progression(
        [45, 41, 48, 43],
        samplerate,
        modes=["min", "maj", "maj", "maj"],
        chord_duration_s=2.5,
    )
    frames = AnalysisEngine(config).process_block(audio)
    tonalites = [frame.key for frame in frames if frame.key]
    dominante = max(set(tonalites), key=tonalites.count)
    assert dominante == Key(9, "min")


def test_chaque_grattage_donne_une_attaque(config, samplerate):
    audio = chord_progression(
        [45, 41, 48, 43], samplerate, modes=["min"] * 4, chord_duration_s=2.5
    )
    frames = AnalysisEngine(config).process_block(audio)
    onsets = [frame.onset for frame in frames if frame.onset]
    assert 4 <= len(onsets) <= 8  # 4 grattages, tolerance sur les cordes etalees


def test_jouer_fait_monter_la_densite(config, guitar_chord, samplerate):
    silence = np.zeros(samplerate // 2)
    frames = AnalysisEngine(config).process_block(
        np.concatenate([silence, np.tile(guitar_chord, 2)])
    )
    pendant_le_silence = [f.density for f in frames if f.time_s < 0.4]
    apres = max(frame.density for frame in frames)
    assert max(pendant_le_silence) == pytest.approx(0.0, abs=1e-9)
    assert apres > 0.2


def test_reset_remet_l_engine_a_zero(config, guitar_chord):
    engine = AnalysisEngine(config)
    premier = engine.process_block(guitar_chord)
    engine.reset()
    assert engine.hop_index == 0
    second = engine.process_block(guitar_chord)
    assert [f.key for f in second] == [f.key for f in premier]


def test_l_analyse_tient_le_temps_reel(config, samplerate):
    """Le budget latence du cadrage suppose de tenir le temps reel largement."""
    import time

    audio = chord_progression(
        [45, 41, 48, 43], samplerate, modes=["min"] * 4, chord_duration_s=2.5
    )
    engine = AnalysisEngine(config)
    debut = time.perf_counter()
    engine.process_block(audio)
    ecoule = time.perf_counter() - debut

    facteur = (audio.size / samplerate) / ecoule
    # Mesure typique : x30. Un seuil a 3 laisse passer une machine de CI lente
    # tout en attrapant un effondrement de performance.
    assert facteur > 3.0, f"analyse a x{facteur:.1f} du temps reel seulement"


# --- integration runtime --------------------------------------------


def test_analyze_offline_emet_le_protocole(config, guitar_chord):
    sink = RecordingSink()
    bridge = JamBridge(sink, OscConfig(), DroneConfig())
    analyze_offline(config, np.tile(guitar_chord, 2), bridge)

    adresses = set(sink.addresses())
    assert "/jam/key" in adresses
    assert "/jam/onset" in adresses
    assert "/jam/energy" in adresses


def test_une_session_complete_produit_du_rythme(config, samplerate):
    from halj.audio.sources import SyntheticSource

    sink = RecordingSink()
    source = SyntheticSource(config.audio, loop=False, chord_duration_s=2.0)
    session = JamSession(
        config,
        source,
        bridge=JamBridge(sink, OscConfig(), DroneConfig()),
        display=None,
        pace_realtime=False,
    )
    stats = session.run()

    assert stats.frames > 0
    assert stats.onsets > 0
    assert stats.hits["kick"] > 0
    assert "/jam/stop" in sink.addresses()
    assert stats.realtime_factor > 1.0


def test_le_mode_analyse_seule_ne_genere_pas_de_rythme(config):
    from halj.audio.sources import SyntheticSource

    sink = RecordingSink()
    source = SyntheticSource(config.audio, loop=False, chord_duration_s=1.5)
    session = JamSession(
        config,
        source,
        bridge=JamBridge(sink, OscConfig(), DroneConfig()),
        display=None,
        enable_rhythm=False,
        pace_realtime=False,
    )
    stats = session.run()

    assert stats.hits == {"kick": 0, "perc": 0}
    assert "/jam/kick" not in sink.addresses()


def test_une_session_s_arrete_sur_demande(config):
    from halj.audio.sources import SyntheticSource

    source = SyntheticSource(config.audio, loop=True)  # source infinie
    session = JamSession(config, source, display=None, pace_realtime=False)
    stats = session.run(max_duration_s=1.0)
    assert 1.0 <= stats.duration_s < 3.0
