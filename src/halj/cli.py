"""Interface en ligne de commande.

Les sous-commandes suivent la roadmap du cadrage, dans l'ordre :

    halj analyze   phase 1 : analyse seule, aucun son genere
    halj drone     phase 2 : drone accorde, sans rythme
    halj jam       phase 3 : drone + couche rythmique
    halj mock      synthese seule, sans analyse (reglage du patch SC)
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace

from .bridge.osc import JamBridge, NullSink, create_sink
from .config import Config
from .generative.euclidean import euclidean_pattern, pulses_for_density
from .mock import run_mock
from .runtime import JamSession
from .ui.terminal import TerminalDisplay, pattern_str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="halj",
        description="Jam generatif local : guitare -> drone accorde + rythme euclidien",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    for name, help_text in (
        ("analyze", "phase 1 : analyse seule (tonalite + attaques), aucun son"),
        ("drone", "phase 2 : drone accorde sur la tonalite detectee"),
        ("jam", "phase 3 : drone + couche rythmique"),
    ):
        cmd = sub.add_parser(name, help=help_text)
        _add_common_args(cmd)
        _add_audio_args(cmd)

    mock = sub.add_parser("mock", help="emet le protocole OSC sans analyse")
    _add_common_args(mock)
    mock.add_argument(
        "--duration", type=float, default=60.0, help="duree en secondes (defaut 60)"
    )

    devices = sub.add_parser("devices", help="liste les entrees audio disponibles")
    devices.set_defaults(needs_config=False)

    patterns = sub.add_parser(
        "patterns", help="affiche les motifs euclidiens par palier de densite"
    )
    patterns.add_argument("--steps", type=int, default=None)
    patterns.add_argument("--config", default=None)

    return parser


def _add_common_args(cmd: argparse.ArgumentParser) -> None:
    cmd.add_argument("--config", default=None, help="fichier JSON de configuration")
    cmd.add_argument("--osc-host", default=None, help="hote OSC (defaut 127.0.0.1)")
    cmd.add_argument("--osc-port", type=int, default=None, help="port OSC (defaut 57120)")
    cmd.add_argument("--bpm", type=float, default=None, help="tempo de la grille")
    cmd.add_argument(
        "--agency",
        type=float,
        default=None,
        metavar="0..1",
        help=(
            "part d'initiative harmonique : 0 = suit la tonalite detectee, "
            "0.5 = deroule une grille mais cede au jeu, 1 = mene (defaut 0.5)"
        ),
    )
    cmd.add_argument(
        "--no-harmony",
        action="store_true",
        help="coupe la grille d'accords : le drone tient la tonalite detectee",
    )
    cmd.add_argument(
        "--no-conversation",
        action="store_true",
        help=(
            "le lead ne circule plus : l'appli garde l'initiative fixee par "
            "--agency au lieu de la prendre et de la rendre selon la place "
            "laissee par le jeu"
        ),
    )
    cmd.add_argument(
        "--no-self-listen",
        action="store_true",
        help=(
            "n'essaie pas de retirer du micro ce que l'appli joue "
            "(a n'utiliser qu'au casque, ou pour comparer)"
        ),
    )


def _add_audio_args(cmd: argparse.ArgumentParser) -> None:
    cmd.add_argument(
        "--source",
        choices=("mic", "file", "synth"),
        default="mic",
        help="origine du signal (defaut : mic)",
    )
    cmd.add_argument("--input", default=None, help="fichier WAV si --source file")
    cmd.add_argument("--device", default=None, help="index ou nom du peripherique d'entree")
    cmd.add_argument("--loop", action="store_true", help="rejoue le fichier en boucle")
    cmd.add_argument(
        "--duration", type=float, default=None, help="arret automatique apres N secondes"
    )
    cmd.add_argument("--quiet", action="store_true", help="pas d'affichage terminal")
    cmd.add_argument(
        "--chroma-backend",
        choices=("numpy", "librosa"),
        default="numpy",
        help=(
            "moteur du chroma : numpy (defaut) ou librosa (CQT). "
            "librosa mesure ~x0.2 du temps reel : reserve a l'analyse d'un "
            "fichier, inutilisable en direct"
        ),
    )
    cmd.add_argument(
        "--onset-backend",
        choices=("numpy", "aubio"),
        default="numpy",
        help="moteur de detection d'attaques : numpy (defaut) ou aubio",
    )
    cmd.add_argument(
        "--no-pacing",
        action="store_true",
        help="analyse une source fichier/synth a pleine vitesse (pas d'ecoute)",
    )


def _resolve_config(args: argparse.Namespace) -> Config:
    config = Config.load(getattr(args, "config", None))

    osc = config.osc
    if getattr(args, "osc_host", None):
        osc = replace(osc, host=args.osc_host)
    if getattr(args, "osc_port", None):
        osc = replace(osc, port=args.osc_port)

    rhythm = config.rhythm
    if getattr(args, "bpm", None):
        rhythm = replace(rhythm, bpm=args.bpm)

    audio = config.audio
    if getattr(args, "device", None):
        device = args.device
        audio = replace(audio, device=int(device) if device.isdigit() else device)

    harmony = config.harmony
    if getattr(args, "agency", None) is not None:
        harmony = replace(harmony, agency=min(1.0, max(0.0, args.agency)))
    if getattr(args, "no_harmony", False):
        harmony = replace(harmony, enabled=False)

    self_listen = config.self_listen
    if getattr(args, "no_self_listen", False):
        self_listen = replace(self_listen, enabled=False)

    conversation = config.conversation
    if getattr(args, "no_conversation", False):
        conversation = replace(conversation, enabled=False)

    return config.with_overrides(
        osc=osc,
        rhythm=rhythm,
        audio=audio,
        harmony=harmony,
        self_listen=self_listen,
        conversation=conversation,
    )


def _run_session(args: argparse.Namespace, config: Config) -> int:
    from .audio.sources import create_source  # import tardif : sounddevice optionnel

    analysis_only = args.command == "analyze"
    enable_rhythm = args.command == "jam"

    try:
        source = create_source(args.source, config.audio, args.input, loop=args.loop)
    except ImportError:
        print(
            "sounddevice est requis pour la capture micro : pip install '.[audio]'\n"
            "(ou utilisez --source synth / --source file pour travailler sans micro)",
            file=sys.stderr,
        )
        return 2
    except (OSError, ValueError) as exc:
        print(f"source audio indisponible : {exc}", file=sys.stderr)
        return 2

    sink = NullSink() if analysis_only else create_sink(config.osc)
    bridge = JamBridge(sink, config.osc, config.drone)
    display = None if args.quiet else TerminalDisplay(show_pattern=enable_rhythm)

    try:
        session = JamSession(
            config,
            source,
            bridge=bridge,
            display=display,
            enable_rhythm=enable_rhythm,
            enable_drone=not analysis_only,
            pace_realtime=False if args.no_pacing else None,
            chroma_backend=args.chroma_backend,
            onset_backend=args.onset_backend,
        )
    except ImportError as exc:
        print(
            f"backend indisponible ({exc}) : pip install '.[mir]'",
            file=sys.stderr,
        )
        return 2

    if not args.quiet:
        target = "aucune (analyse seule)" if analysis_only else (
            f"{config.osc.host}:{config.osc.port}{config.osc.prefix}"
        )
        print(f"source : {args.source} | sortie OSC : {target}")
        print("Ctrl-C pour arreter.")

    stats = session.run(max_duration_s=args.duration)
    print(stats.summary())
    return 0


def _run_mock(args: argparse.Namespace, config: Config) -> int:
    bridge = JamBridge(create_sink(config.osc), config.osc, config.drone)
    print(
        f"mock -> {config.osc.host}:{config.osc.port}{config.osc.prefix} "
        f"pendant {args.duration:.0f}s (Ctrl-C pour arreter)"
    )
    hits = run_mock(config, bridge, duration_s=args.duration)
    print(f"{hits} impacts rythmiques emis.")
    return 0


def _run_devices() -> int:
    try:
        import sounddevice as sd
    except (ImportError, OSError) as exc:
        print(f"sounddevice indisponible ({exc}) : pip install '.[audio]'", file=sys.stderr)
        return 2
    for index, device in enumerate(sd.query_devices()):
        if device["max_input_channels"] > 0:
            print(
                f"[{index}] {device['name']} "
                f"({device['max_input_channels']} ch, "
                f"{device['default_samplerate']:.0f} Hz)"
            )
    return 0


def _run_patterns(args: argparse.Namespace) -> int:
    config = Config.load(args.config)
    rhythm = config.rhythm
    steps = args.steps or rhythm.steps
    print(f"grille de {steps} pas, {rhythm.bpm:.0f} BPM\n")
    for percent in range(0, 101, 10):
        density = percent / 100.0
        kick = pulses_for_density(density, rhythm.kick_pulses_min, rhythm.kick_pulses_max)
        perc = pulses_for_density(density, rhythm.perc_pulses_min, rhythm.perc_pulses_max)
        gated = " (sous le seuil : silence)" if density < rhythm.gate_density else ""
        print(f"densite {percent:3d}%{gated}")
        print(
            f"  kick E({kick:2d},{steps}) "
            f"{pattern_str(euclidean_pattern(steps, min(kick, steps), rhythm.kick_rotation))}"
        )
        print(
            f"  perc E({perc:2d},{steps}) "
            f"{pattern_str(euclidean_pattern(steps, min(perc, steps), rhythm.perc_rotation))}"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "devices":
        return _run_devices()
    if args.command == "patterns":
        return _run_patterns(args)

    config = _resolve_config(args)
    if args.command == "mock":
        return _run_mock(args, config)
    return _run_session(args, config)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
