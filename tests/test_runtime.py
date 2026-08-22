"""La session : ce que l'analyse, la grille et l'auto-ecoute se disent entre elles.

Les briques ont leurs propres tests ; ici on verifie le cablage — que l'appli
declare bien a son analyse ce qu'elle vient de jouer, et qu'elle annonce sa
grille sur le protocole.
"""

import pytest

from halj.audio.sources import SyntheticSource
from halj.bridge.osc import JamBridge, RecordingSink
from halj.config import HarmonyConfig
from halj.notes import Key
from halj.runtime import JamSession


@pytest.fixture
def session_factory(config):
    """Une session complete sur la guitare de synthese, jouee sans temps reel."""

    def build(**kwargs):
        sink = RecordingSink()
        session_config = kwargs.pop("config", config)
        source = SyntheticSource(session_config.audio, loop=True)
        session = JamSession(
            session_config,
            source,
            bridge=JamBridge(sink, session_config.osc, session_config.drone),
            pace_realtime=False,
            **kwargs,
        )
        return session, sink

    return build


def test_la_session_annonce_sa_grille(session_factory):
    session, sink = session_factory(enable_rhythm=True)
    stats = session.run(max_duration_s=30.0)

    accords = [args for adresse, args in sink.messages if adresse.endswith("/chord")]
    assert accords, "aucun accord annonce"
    assert stats.chord_changes == len(accords)
    # Un accord n'est jamais reemis a l'identique : le protocole reste calme.
    assert all(a != b for a, b in zip(accords, accords[1:]))
    # Et chacun porte de quoi le jouer : fondamentale, qualite, hauteur, decision.
    for root, quality, root_hz, decision in accords:
        assert 0 <= root < 12
        assert quality in ("maj", "min", "sus4")
        assert 30.0 < root_hz < 500.0
        assert decision in ("lead", "follow", "anchor")


def test_la_session_declare_son_drone_a_l_auto_ecoute(session_factory):
    """Sans cette declaration, l'analyse ne saurait pas quoi retirer."""
    session, _ = session_factory(enable_rhythm=True)
    session.run(max_duration_s=20.0)
    assert session.engine.self_listen.active


def test_la_session_declare_ses_impacts(session_factory):
    session, sink = session_factory(enable_rhythm=True)
    session.run(max_duration_s=20.0)
    impacts = sum(
        1 for adresse, _ in sink.messages if adresse.endswith(("/kick", "/perc"))
    )
    assert impacts > 0
    # Chaque impact emis a ete signale au moteur d'analyse (la file se vide au
    # fil du temps, on verifie donc le compteur cote session).
    assert session.stats.hits["kick"] + session.stats.hits["perc"] == impacts


def test_sans_grille_le_drone_suit_la_tonalite(session_factory, config):
    """`--no-harmony` doit rendre exactement le comportement d'avant."""
    session, sink = session_factory(
        config=config.with_overrides(harmony=HarmonyConfig(enabled=False)),
        enable_rhythm=True,
    )
    session.run(max_duration_s=20.0)
    assert session.harmony is None
    assert not any(adresse.endswith("/chord") for adresse, _ in sink.messages)
    assert any(adresse.endswith("/key") for adresse, _ in sink.messages)
    # L'auto-ecoute reste active : le drone joue toujours, il faut le retirer.
    assert session.engine.self_listen.active


def test_le_dernier_pas_avant_un_changement_est_accentue(session_factory):
    """La grille doit s'entendre venir : un batteur annonce le changement."""
    session, _ = session_factory(enable_rhythm=True)
    session.harmony.observe_key(Key(9, "min"))
    frame = type("Trame", (), {"time_s": 0.0})()
    session.harmony._next_change_step = 1  # le changement arrive au pas suivant
    accentuee = session._cued(0.5, frame)
    session.harmony._next_change_step = 10  # ...et la, il est loin
    normale = session._cued(0.5, frame)
    assert accentuee > normale == 0.5
    assert accentuee <= 1.0
