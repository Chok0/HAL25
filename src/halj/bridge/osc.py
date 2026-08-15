"""Pont OSC vers le moteur de synthese.

Le decouplage total analyse / synthese est un choix de cadrage : Python fait
l'intelligence, SuperCollider fait le son, OSC les relie. Consequence pratique
— chacun des deux cotes se teste seul (cf. `halj mock`, qui emet le protocole
complet sans analyser quoi que ce soit).

Protocole (prefixe configurable, /jam par defaut) :

    /jam/key    tonic:int(0-11) mode:str("maj"|"min") root_hz:float conf:float
    /jam/energy level:float density:float
    /jam/onset  strength:float
    /jam/kick   velocity:float step:int
    /jam/perc   velocity:float step:int
    /jam/drone  amp:float glide:float
    /jam/stop   (extinction propre de toutes les voix)
"""

from __future__ import annotations

from typing import Any, Protocol

from ..config import DroneConfig, OscConfig
from ..notes import Key


class Sink(Protocol):
    """Destination des messages de controle."""

    def send(self, address: str, *args: Any) -> None: ...

    def close(self) -> None: ...


class NullSink:
    """Ne fait rien : mode analyse seule (phase 1 de la roadmap)."""

    def send(self, address: str, *args: Any) -> None:  # noqa: D102
        return None

    def close(self) -> None:  # noqa: D102
        return None


class RecordingSink:
    """Enregistre les messages en memoire : utilise par les tests."""

    def __init__(self) -> None:
        self.messages: list[tuple[str, tuple[Any, ...]]] = []

    def send(self, address: str, *args: Any) -> None:
        self.messages.append((address, args))

    def close(self) -> None:
        return None

    def addresses(self) -> list[str]:
        return [address for address, _ in self.messages]


class OscSink:
    """Client OSC UDP (python-osc)."""

    def __init__(self, config: OscConfig):
        from pythonosc.udp_client import SimpleUDPClient  # import tardif

        self.config = config
        self._client = SimpleUDPClient(config.host, config.port)

    def send(self, address: str, *args: Any) -> None:
        # UDP en local : pas de retransmission, pas de blocage. Un paquet perdu
        # est sans consequence, l'etat est reemis a la trame suivante.
        self._client.send_message(address, list(args))

    def close(self) -> None:
        return None


class JamBridge:
    """Traduit l'etat musical en messages OSC, en evitant le bavardage.

    Les messages d'etat (tonalite, energie) ne sont emis que lorsqu'ils
    changent significativement : inonder le reseau local a chaque hop
    n'apporte rien et ajoute de la gigue aux messages qui comptent.
    """

    def __init__(
        self,
        sink: Sink,
        osc: OscConfig,
        drone: DroneConfig | None = None,
        energy_epsilon: float = 0.02,
    ):
        self.sink = sink
        self.osc = osc
        self.drone = drone or DroneConfig()
        self.energy_epsilon = energy_epsilon
        self._last_key: Key | None = None
        self._last_level: float | None = None
        self._last_density: float | None = None

    def _address(self, leaf: str) -> str:
        return f"{self.osc.prefix}/{leaf}"

    def send_key(self, key: Key, confidence: float, force: bool = False) -> bool:
        """Emet la tonalite si elle a change. Retourne True si un message est parti."""
        if not force and key == self._last_key:
            return False
        self._last_key = key
        self.sink.send(
            self._address("key"),
            int(key.tonic),
            key.mode,
            float(key.root_hz(self.drone.octave)),
            float(confidence),
        )
        return True

    def send_energy(self, level: float, density: float, force: bool = False) -> bool:
        changed = (
            self._last_level is None
            or abs(level - self._last_level) >= self.energy_epsilon
            or abs(density - (self._last_density or 0.0)) >= self.energy_epsilon
        )
        if not force and not changed:
            return False
        self._last_level, self._last_density = level, density
        self.sink.send(self._address("energy"), float(level), float(density))
        return True

    def send_onset(self, strength: float) -> None:
        self.sink.send(self._address("onset"), float(strength))

    def send_hit(self, voice: str, velocity: float, step: int) -> None:
        if voice not in ("kick", "perc"):
            raise ValueError(f"voix inconnue : {voice}")
        self.sink.send(self._address(voice), float(velocity), int(step))

    def start_drone(self) -> None:
        self.sink.send(
            self._address("drone"),
            float(self.drone.amp),
            float(self.drone.glide_s),
        )

    def stop(self) -> None:
        """Extinction propre : a appeler avant de quitter, sinon le drone reste."""
        self.sink.send(self._address("stop"))

    def close(self) -> None:
        self.sink.close()


def create_sink(config: OscConfig, enabled: bool = True) -> Sink:
    """Retourne un client OSC reel, ou un puits inerte si la synthese est coupee."""
    return OscSink(config) if enabled else NullSink()
