"""Boucle temps reel : source audio -> analyse -> OSC + affichage.

Le temps de reference est l'horloge audio (nombre d'echantillons consommes),
pas l'horloge murale. Consequence : sur une source fichier ou synthetique, la
session se rejoue a l'identique, aussi vite que le CPU le permet, et les dates
des evenements restent exactes.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .analysis.engine import AnalysisEngine, AnalysisFrame
from .audio.sources import AudioSource
from .bridge.osc import JamBridge
from .config import Config
from .generative.conversation import Conversation, ResponseNote, response_phrase
from .generative.harmony import HarmonyDirector, voiced_root_hz
from .generative.scheduler import RhythmScheduler
from .notes import midi_to_hz
from .ui.terminal import TerminalDisplay


@dataclass
class SessionStats:
    """De quoi juger une session apres coup (et calibrer les seuils)."""

    frames: int = 0
    onsets: int = 0
    key_changes: int = 0
    chord_changes: int = 0
    lead_changes: int = 0
    response_notes: int = 0
    hits: dict[str, int] = field(default_factory=lambda: {"kick": 0, "perc": 0})
    duration_s: float = 0.0
    wall_time_s: float = 0.0

    @property
    def realtime_factor(self) -> float:
        """Duree audio / temps CPU. > 1 : on tient le temps reel."""
        if self.wall_time_s <= 0:
            return float("inf")
        return self.duration_s / self.wall_time_s

    def summary(self) -> str:
        return (
            f"{self.duration_s:.1f}s analysees en {self.wall_time_s:.1f}s "
            f"(x{self.realtime_factor:.1f} temps reel) | "
            f"{self.onsets} attaques | {self.key_changes} changements de tonalite | "
            f"{self.chord_changes} accords | "
            f"{self.lead_changes} passages de main, {self.response_notes} notes "
            f"de reponse | "
            f"{self.hits['kick']} kicks, {self.hits['perc']} percus"
        )


class JamSession:
    """Assemble analyse, generation et sortie pour une session de jeu."""

    def __init__(
        self,
        config: Config,
        source: AudioSource,
        bridge: JamBridge | None = None,
        display: TerminalDisplay | None = None,
        enable_rhythm: bool = True,
        enable_drone: bool = True,
        pace_realtime: bool | None = None,
        chroma_backend: str = "numpy",
        onset_backend: str = "numpy",
    ):
        self.config = config
        self.source = source
        self.bridge = bridge
        self.display = display
        self.enable_rhythm = enable_rhythm
        self.enable_drone = enable_drone
        # Une source non temps reel (fichier, synthese) defile aussi vite que
        # le CPU le permet ; on la cadence seulement si on veut l'ecouter.
        self.pace_realtime = (
            (not source.realtime) if pace_realtime is None else pace_realtime
        )

        self.engine = AnalysisEngine(
            config, chroma_backend=chroma_backend, onset_backend=onset_backend
        )
        self.scheduler = RhythmScheduler(config.rhythm)
        # Le directeur harmonique partage la grille du sequenceur : les accords
        # changent a la mesure, pas au milieu d'un motif.
        self.harmony = (
            HarmonyDirector(config.harmony, steps_per_bar=config.rhythm.steps)
            if config.harmony.enabled and enable_drone
            else None
        )
        # Qui mene, a cet instant. La grille d'accords et la densite rythmique
        # en dependent toutes les deux : c'est la meme decision.
        self.conversation = (
            Conversation(config.conversation, config.harmony.agency)
            if config.conversation.enabled and enable_drone
            else None
        )
        self.stats = SessionStats()
        self._stop = False
        self._last_key = None
        self._drone_root_hz: float | None = None
        self._pending_notes: list[tuple[float, ResponseNote]] = []

    def stop(self) -> None:
        """Demande l'arret ; la boucle sort a la fin du bloc courant."""
        self._stop = True

    def run(self, max_duration_s: float | None = None) -> SessionStats:
        """Boucle principale. Rend les statistiques de la session."""
        wall_start = time.perf_counter()
        if self.bridge and self.enable_drone:
            self.bridge.start_drone()

        try:
            for block in self.source.iter_blocks():
                if self._stop:
                    break
                for frame in self.engine.process_block(np.asarray(block)):
                    self._handle_frame(frame)
                    if self.pace_realtime:
                        self._pace(frame.time_s, wall_start)
                if max_duration_s is not None and self.stats.duration_s >= max_duration_s:
                    break
        except KeyboardInterrupt:
            pass
        finally:
            self.stats.wall_time_s = time.perf_counter() - wall_start
            self._teardown()
        return self.stats

    # -- traitement d'une trame ---------------------------------------

    def _handle_frame(self, frame: AnalysisFrame) -> None:
        self.stats.frames += 1
        self.stats.duration_s = frame.time_s

        if frame.onset is not None:
            self.stats.onsets += 1
            if self.bridge:
                self.bridge.send_onset(frame.onset.strength)

        # Compte tenu cote analyse, pas cote OSC : en mode analyse seule il
        # n'y a pas de bridge actif, et le decompte doit rester juste.
        if frame.key is not None and frame.key != self._last_key:
            self.stats.key_changes += 1
            self._last_key = frame.key

        if self.bridge:
            if frame.key is not None and self.enable_drone:
                self.bridge.send_key(frame.key, frame.key_confidence)
            self.bridge.send_energy(frame.level, frame.density)

        self._advance_conversation(frame)
        self._advance_harmony(frame)
        self._emit_response(frame)

        if self.enable_rhythm:
            self.scheduler.set_density(self._played_density(frame.density))
            for event in self.scheduler.advance(frame.time_s):
                self.stats.hits[event.voice] += 1
                # Declare l'impact au moteur d'analyse : dans une seconde il
                # sera dans le micro, et il ne doit pas y passer pour une
                # attaque du joueur.
                self.engine.note_self_hit(event.time_s)
                if self.bridge:
                    self.bridge.send_hit(
                        event.voice, self._cued(event.velocity, frame), event.step
                    )

        if self.display:
            self.display.render(
                frame,
                self.scheduler if self.enable_rhythm else None,
                harmony=self.harmony,
                conversation=self.conversation,
            )

    # -- a qui est le tour ----------------------------------------------

    def _advance_conversation(self, frame: AnalysisFrame) -> None:
        """Met a jour le meneur du moment, et ce qui en decoule."""
        if self.conversation is None:
            return
        # Une trame ou l'appli s'entendait parler ne dit rien du jeu : la
        # compter ferait croire que l'instrumentiste occupe le terrain.
        if not frame.muted:
            change = self.conversation.observe(frame.density, frame.time_s)
            if change is not None:
                self.stats.lead_changes += 1
                if self.bridge:
                    self.bridge.send_lead(change)
        if self.harmony is not None:
            self.harmony.agency = self.conversation.agency()

    def _played_density(self, measured: float) -> float:
        """La densite reellement jouee : mesuree, avec le plancher du moment."""
        if self.conversation is None:
            return measured
        return self.conversation.density(measured)

    def _emit_response(self, frame: AnalysisFrame) -> None:
        """Emet les notes de la phrase de reponse arrivees a echeance."""
        while self._pending_notes and self._pending_notes[0][0] <= frame.time_s:
            time_s, note = self._pending_notes.pop(0)
            duree = self.config.conversation.response_note_s
            # Declaree avant d'etre emise : le temps qu'elle sonne, l'analyse
            # se met en pause plutot que de prendre l'appli pour un musicien.
            self.engine.note_self_voice(frame.time_s, duree)
            self.stats.response_notes += 1
            if self.bridge:
                self.bridge.send_note(midi_to_hz(note.midi), note.velocity, duree)

    def _plan_response(self, chord, step_index: int) -> None:
        """Prepare une phrase de reponse sur le nouvel accord, si c'est le tour."""
        if self.conversation is None or not self.conversation.wants_response():
            return
        bar = step_index // max(1, self.config.rhythm.steps)
        step_s = self.scheduler.step_duration
        self._pending_notes = [
            (max(0.0, (step_index + note.step) * step_s), note)
            for note in response_phrase(
                chord, bar, self.config.conversation.response_octave
            )
        ]

    # -- grille d'accords ----------------------------------------------

    def _step_index(self, time_s: float) -> int:
        """Pas de la grille a cet instant, cale sur le meme zero que le sequenceur."""
        return int(time_s / self.scheduler.step_duration)

    def _advance_harmony(self, frame: AnalysisFrame) -> None:
        """Ancre, nourrit et fait avancer la grille d'accords."""
        if self.harmony is None:
            if self.enable_drone and frame.key is not None:
                # Sans directeur, le drone tient la tonalite : c'est quand meme
                # ce qu'il faut declarer a l'auto-ecoute.
                intervals = (0, 3, 7) if frame.key.mode == "min" else (0, 4, 7)
                self.engine.set_self_drone(
                    frame.key.root_hz(self.config.drone.octave), intervals
                )
            return

        if frame.chroma is not None:
            self.harmony.observe_chroma(frame.chroma)

        step = self._step_index(frame.time_s)
        change = self.harmony.observe_key(frame.key, step) or self.harmony.on_step(step)
        if change is None:
            return

        self.stats.chord_changes += 1
        self._plan_response(change.chord, step)
        root_hz = voiced_root_hz(
            change.chord.root, self.config.drone.octave, self._drone_root_hz
        )
        self._drone_root_hz = root_hz
        # Le drone joue cet accord : le moteur d'analyse peut donc le retirer
        # de ce que le micro lui renvoie.
        self.engine.set_self_drone(root_hz, change.chord.intervals)
        if self.bridge:
            self.bridge.send_chord(change.chord, root_hz, change.decision)

    def _cued(self, velocity: float, frame: AnalysisFrame) -> float:
        """Accentue le dernier pas avant un changement d'accord.

        Une grille qui bouge sans prevenir ne se joue pas : l'accent annonce
        que l'accord suivant arrive, comme le ferait un batteur.
        """
        if self.harmony is None:
            return velocity
        if self.harmony.steps_to_change(self._step_index(frame.time_s)) > 1:
            return velocity
        return round(min(1.0, velocity * 1.3), 4)

    def _pace(self, audio_time_s: float, wall_start: float) -> None:
        """Freine pour coller au temps reel quand la source ne le fait pas."""
        ahead = audio_time_s - (time.perf_counter() - wall_start)
        if ahead > 0.002:
            time.sleep(ahead)

    def _teardown(self) -> None:
        if self.bridge:
            self.bridge.stop()
            self.bridge.close()
        if self.display:
            self.display.close()
        self.source.close()


def analyze_offline(
    config: Config, samples: np.ndarray, bridge: JamBridge | None = None
) -> list[AnalysisFrame]:
    """Analyse un signal deja en memoire, sans temps reel ni source.

    Utilise par les tests et par `halj analyze --input` pour rejouer une prise.
    """
    engine = AnalysisEngine(config)
    frames = engine.process_block(np.asarray(samples))
    if bridge:
        for frame in frames:
            if frame.onset is not None:
                bridge.send_onset(frame.onset.strength)
            if frame.key is not None:
                bridge.send_key(frame.key, frame.key_confidence)
            bridge.send_energy(frame.level, frame.density)
    return frames
