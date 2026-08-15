import json

import pytest

from halj.bridge.osc import JamBridge, RecordingSink
from halj.cli import build_parser, main
from halj.config import Config, DroneConfig, OscConfig
from halj.mock import run_mock
from halj.ui.terminal import TerminalDisplay, bar, pattern_str


# --- analyse des arguments -------------------------------------------


def test_les_sous_commandes_de_la_roadmap_existent():
    parser = build_parser()
    for commande in ("analyze", "drone", "jam", "mock", "devices", "patterns"):
        assert parser.parse_args([commande]).command == commande


def test_une_commande_est_obligatoire():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_les_options_osc_sont_lues():
    args = build_parser().parse_args(
        ["jam", "--osc-host", "10.0.0.2", "--osc-port", "9000", "--bpm", "96"]
    )
    assert (args.osc_host, args.osc_port, args.bpm) == ("10.0.0.2", 9000, 96.0)


def test_une_source_inconnue_est_rejetee():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["jam", "--source", "theremine"])


# --- execution -------------------------------------------------------


def test_analyze_sur_source_synthetique(capsys):
    code = main(
        ["analyze", "--source", "synth", "--duration", "1.0", "--no-pacing", "--quiet"]
    )
    assert code == 0
    assert "analysees" in capsys.readouterr().out


def test_jam_sur_source_synthetique_sans_serveur_osc(capsys):
    # Rien n'ecoute en face : l'UDP part dans le vide, la session doit
    # neanmoins se derouler et se terminer proprement.
    code = main(
        ["jam", "--source", "synth", "--duration", "1.0", "--no-pacing", "--quiet"]
    )
    assert code == 0
    assert "attaques" in capsys.readouterr().out


def test_jam_sur_un_fichier(tmp_path, capsys, samplerate):
    from halj.audio.synth import MAJOR_TRIAD, strum

    from .test_sources import ecrire_wav

    chemin = tmp_path / "prise.wav"
    ecrire_wav(chemin, strum(48, MAJOR_TRIAD, 1.5, samplerate), samplerate)

    assert main(["jam", "--source", "file", "--input", str(chemin), "--no-pacing", "--quiet"]) == 0
    assert "analysees" in capsys.readouterr().out


def test_un_fichier_absent_est_signale(capsys):
    code = main(["analyze", "--source", "file", "--input", "/inexistant.wav", "--quiet"])
    assert code == 2
    assert "indisponible" in capsys.readouterr().err


def test_le_fichier_de_configuration_est_pris_en_compte(tmp_path, capsys):
    chemin = tmp_path / "config.json"
    chemin.write_text(json.dumps({"rhythm": {"bpm": 150.0}}), encoding="utf-8")

    code = main(["patterns", "--config", str(chemin)])
    assert code == 0
    assert "150" in capsys.readouterr().out


def test_patterns_affiche_toute_l_echelle_de_densite(capsys):
    assert main(["patterns"]) == 0
    sortie = capsys.readouterr().out
    assert "densite   0%" in sortie
    assert "densite 100%" in sortie
    assert sortie.count("kick") == 11


# --- mode mock -------------------------------------------------------


def test_le_mock_emet_le_protocole_complet():
    sink = RecordingSink()
    bridge = JamBridge(sink, OscConfig(), DroneConfig())
    run_mock(Config(), bridge, duration_s=0.6, key_period_s=0.2, density_period_s=0.4)

    adresses = set(sink.addresses())
    assert {"/jam/drone", "/jam/key", "/jam/energy", "/jam/stop"} <= adresses


def test_le_mock_parcourt_plusieurs_tonalites():
    sink = RecordingSink()
    bridge = JamBridge(sink, OscConfig(), DroneConfig())
    run_mock(Config(), bridge, duration_s=1.0, key_period_s=0.2, density_period_s=0.5)

    tonalites = {args[:2] for address, args in sink.messages if address == "/jam/key"}
    assert len(tonalites) >= 2


def test_le_mock_fait_varier_la_densite():
    sink = RecordingSink()
    bridge = JamBridge(sink, OscConfig(), DroneConfig())
    run_mock(Config(), bridge, duration_s=1.2, density_period_s=0.6)

    densites = [args[1] for address, args in sink.messages if address == "/jam/energy"]
    assert max(densites) - min(densites) > 0.4


# --- affichage -------------------------------------------------------


def test_barre_de_niveau():
    assert bar(0.0, 10) == "." * 10
    assert bar(1.0, 10) == "#" * 10
    assert bar(0.5, 10).count("#") == 5
    # les valeurs hors bornes sont clampees plutot que de casser l'affichage
    assert bar(-1.0, 10) == "." * 10
    assert bar(5.0, 10) == "#" * 10


def test_affichage_d_un_motif():
    assert pattern_str([True, False, True, False]) == "x  -  x  -"
    assert "[x]" in pattern_str([True, False], cursor=0)


def test_la_ligne_d_etat_montre_l_essentiel(config, guitar_chord):
    from halj.analysis.engine import AnalysisEngine

    frames = AnalysisEngine(config).process_block(guitar_chord)
    avec_tonalite = [frame for frame in frames if frame.key]
    ligne = TerminalDisplay().format_line(avec_tonalite[-1])

    assert "key" in ligne and "conf" in ligne and "dens" in ligne
    assert avec_tonalite[-1].key.name in ligne


def test_l_affichage_est_limite_en_frequence(config):
    import io

    from halj.analysis.engine import AnalysisEngine

    flux = io.StringIO()
    display = TerminalDisplay(stream=flux, refresh_hz=10.0)
    frames = AnalysisEngine(config).process_block(
        __import__("numpy").zeros(config.audio.hop_size * 100)
    )
    rendus = [display.render(frame) for frame in frames]

    # 100 trames de 5.8 ms couvrent 0.58 s, soit ~6 rafraichissements a 10 Hz.
    assert 3 <= sum(ligne is not None for ligne in rendus) <= 8


# --- backends optionnels ---------------------------------------------


def test_le_backend_aubio_est_selectionnable(capsys):
    pytest.importorskip("aubio")
    code = main(
        [
            "analyze",
            "--source",
            "synth",
            "--duration",
            "1.0",
            "--no-pacing",
            "--quiet",
            "--onset-backend",
            "aubio",
        ]
    )
    assert code == 0
    assert "attaques" in capsys.readouterr().out


def test_un_backend_absent_est_signale_proprement(monkeypatch, capsys):
    import halj.analysis.engine as engine_module

    def refuse(*args, **kwargs):
        raise ImportError("aubio")

    monkeypatch.setattr(engine_module, "create_onset_detector", refuse)
    code = main(
        ["analyze", "--source", "synth", "--duration", "1.0", "--quiet",
         "--onset-backend", "aubio"]
    )
    assert code == 2
    assert "pip install" in capsys.readouterr().err
