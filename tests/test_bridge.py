import threading
import time

import pytest

from halj.bridge.osc import JamBridge, NullSink, RecordingSink, create_sink
from halj.config import DroneConfig, OscConfig
from halj.notes import Key


@pytest.fixture
def sink():
    return RecordingSink()


@pytest.fixture
def bridge(sink):
    return JamBridge(sink, OscConfig(), DroneConfig())


def test_le_prefixe_est_applique(sink):
    bridge = JamBridge(sink, OscConfig(prefix="/autre"), DroneConfig())
    bridge.send_onset(1.0)
    assert sink.addresses() == ["/autre/onset"]


def test_la_tonalite_porte_sa_frequence(bridge, sink):
    bridge.send_key(Key(9, "min"), 0.8)
    address, args = sink.messages[0]
    tonic, mode, root_hz, confidence = args
    assert address == "/jam/key"
    assert (tonic, mode) == (9, "min")
    assert root_hz == pytest.approx(110.0, abs=0.1)  # la1
    assert confidence == pytest.approx(0.8)


def test_la_tonalite_n_est_reemise_que_si_elle_change(bridge, sink):
    assert bridge.send_key(Key(0, "maj"), 0.7) is True
    assert bridge.send_key(Key(0, "maj"), 0.9) is False
    assert bridge.send_key(Key(5, "maj"), 0.9) is True
    assert len(sink.messages) == 2


def test_la_tonalite_peut_etre_reemise_de_force(bridge, sink):
    bridge.send_key(Key(0, "maj"), 0.7)
    assert bridge.send_key(Key(0, "maj"), 0.7, force=True) is True
    assert len(sink.messages) == 2


def test_l_energie_ne_repete_pas_les_micro_variations(bridge, sink):
    assert bridge.send_energy(0.50, 0.50) is True
    assert bridge.send_energy(0.505, 0.505) is False  # sous epsilon
    assert bridge.send_energy(0.60, 0.60) is True
    assert len(sink.messages) == 2


def test_l_octave_du_drone_est_respectee(sink):
    grave = JamBridge(sink, OscConfig(), DroneConfig(octave=1))
    grave.send_key(Key(9, "min"), 1.0)
    assert sink.messages[0][1][2] == pytest.approx(55.0, abs=0.1)


def test_les_impacts_portent_velocite_et_pas(bridge, sink):
    bridge.send_hit("kick", 0.9, 4)
    assert sink.messages[0] == ("/jam/kick", (0.9, 4))


def test_une_voix_inconnue_est_refusee(bridge):
    with pytest.raises(ValueError):
        bridge.send_hit("trompette", 0.5, 0)


def test_le_demarrage_du_drone_transmet_ses_reglages(sink):
    bridge = JamBridge(sink, OscConfig(), DroneConfig(amp=0.5, glide_s=3.0))
    bridge.start_drone()
    assert sink.messages[0] == ("/jam/drone", (0.5, 3.0))


def test_l_arret_est_explicite(bridge, sink):
    bridge.stop()
    assert sink.addresses() == ["/jam/stop"]


def test_le_puits_inerte_n_emet_rien():
    bridge = JamBridge(NullSink(), OscConfig(), DroneConfig())
    bridge.send_key(Key(0, "maj"), 1.0)
    bridge.send_onset(2.0)
    bridge.stop()  # aucune exception : c'est tout ce qu'on demande


def test_create_sink_desactive():
    assert isinstance(create_sink(OscConfig(), enabled=False), NullSink)


def test_transmission_udp_reelle():
    """Verifie le protocole de bout en bout, sur une vraie socket."""
    pytest.importorskip("pythonosc")
    from pythonosc.dispatcher import Dispatcher
    from pythonosc.osc_server import ThreadingOSCUDPServer

    recu: list[tuple] = []
    evenement = threading.Event()

    dispatcher = Dispatcher()
    dispatcher.map(
        "/jam/*",
        lambda address, *args: (recu.append((address, args)), evenement.set()),
    )
    # port 0 : le systeme en attribue un libre, pas de collision en CI.
    server = ThreadingOSCUDPServer(("127.0.0.1", 0), dispatcher)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        config = OscConfig(host="127.0.0.1", port=port)
        bridge = JamBridge(create_sink(config), config, DroneConfig())
        bridge.send_key(Key(2, "min"), 0.77)

        deadline = time.time() + 5.0
        while not recu and time.time() < deadline:
            evenement.wait(0.05)
    finally:
        server.shutdown()
        server.server_close()

    assert recu, "aucun message OSC recu"
    address, args = recu[0]
    assert address == "/jam/key"
    assert args[0] == 2
    assert args[1] == "min"
    assert args[3] == pytest.approx(0.77, abs=1e-6)
