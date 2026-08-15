"""Mode mock : emet le protocole OSC sans analyser quoi que ce soit.

Prevu des le cadrage comme parade au principal risque de dev — l'overhead du
setup cross-process. Il permet de developper et regler tout le patch
SuperCollider (drones, percus, mixage) sans micro, sans guitare, et sans que
l'analyse soit terminee.
"""

from __future__ import annotations

import math
import time

from .bridge.osc import JamBridge
from .config import Config
from .generative.scheduler import RhythmScheduler
from .notes import Key

# Progression de reference : tourne autour du relatif majeur/mineur, donc elle
# exerce aussi la reponse du drone a un changement de mode.
DEFAULT_KEYS: tuple[Key, ...] = (
    Key(9, "min"),  # A min
    Key(5, "maj"),  # F maj
    Key(0, "maj"),  # C maj
    Key(7, "maj"),  # G maj
)


def run_mock(
    config: Config,
    bridge: JamBridge,
    duration_s: float = 60.0,
    key_period_s: float = 8.0,
    density_period_s: float = 12.0,
    on_tick=None,
) -> int:
    """Joue une session simulee. Retourne le nombre de messages rythmiques emis.

    La densite suit une sinusoide lente : de quoi voir la grille euclidienne
    se densifier et s'aerer toute seule, et verifier a l'oreille que les
    transitions de motif ne claquent pas.
    """
    scheduler = RhythmScheduler(config.rhythm)
    bridge.start_drone()

    start = time.perf_counter()
    hits = 0
    try:
        while True:
            now = time.perf_counter() - start
            if now >= duration_s:
                break

            key = DEFAULT_KEYS[int(now / key_period_s) % len(DEFAULT_KEYS)]
            bridge.send_key(key, confidence=0.9)

            # sinusoide dans [0, 1], demarree au creux.
            density = 0.5 - 0.5 * math.cos(2 * math.pi * now / density_period_s)
            bridge.send_energy(level=density, density=density)
            scheduler.set_density(density)

            for event in scheduler.advance(now):
                bridge.send_hit(event.voice, event.velocity, event.step)
                hits += 1

            if on_tick is not None:
                on_tick(now, key, density)

            # Pas de sommeil long : la grille a une resolution de l'ordre de
            # 20 ms a 84 BPM, on echantillonne nettement plus fin.
            time.sleep(0.005)
    except KeyboardInterrupt:
        pass
    finally:
        bridge.stop()
        bridge.close()
    return hits
