"""Affichage temps reel dans le terminal.

C'est le livrable de la phase 1 de la roadmap : voir la tonalite et les
attaques defiler, sans une seule note generee, pour juger de la fiabilite de la
detection sur une vraie guitare.
"""

from __future__ import annotations

import shutil
import sys
from typing import TextIO

from ..analysis.engine import AnalysisFrame
from ..generative.scheduler import RhythmScheduler


def bar(value: float, width: int = 14, fill: str = "#", empty: str = ".") -> str:
    """Barre de niveau ASCII (pas d'unicode : ca casse selon les terminaux)."""
    value = min(1.0, max(0.0, value))
    filled = int(round(value * width))
    return fill * filled + empty * (width - filled)


def pattern_str(pattern: list[bool], cursor: int | None = None) -> str:
    """Rend un motif euclidien, avec le pas courant entre crochets."""
    cells = []
    for index, active in enumerate(pattern):
        glyph = "x" if active else "-"
        cells.append(f"[{glyph}]" if index == cursor else f" {glyph} ")
    return "".join(cells).strip()


class TerminalDisplay:
    """Rafraichit une ligne d'etat, sans faire defiler le terminal."""

    def __init__(
        self,
        stream: TextIO | None = None,
        refresh_hz: float = 20.0,
        show_pattern: bool = True,
    ):
        self.stream = stream or sys.stdout
        self.min_interval_s = 1.0 / refresh_hz
        self.show_pattern = show_pattern
        self._last_render_s = -1.0
        self._onset_flash_until = -1.0

    def render(
        self,
        frame: AnalysisFrame,
        scheduler: RhythmScheduler | None = None,
        force: bool = False,
    ) -> str | None:
        """Affiche l'etat si le moment est venu. Retourne la ligne ecrite."""
        if frame.onset is not None:
            # On tient l'eclair allume assez longtemps pour qu'il soit visible
            # meme si l'attaque tombe entre deux rafraichissements.
            self._onset_flash_until = frame.time_s + 0.08

        if not force and frame.time_s - self._last_render_s < self.min_interval_s:
            return None
        self._last_render_s = frame.time_s

        line = self.format_line(frame, scheduler)
        width = shutil.get_terminal_size((100, 24)).columns
        self.stream.write("\r" + line[: width - 1].ljust(width - 1))
        self.stream.flush()
        return line

    def format_line(
        self, frame: AnalysisFrame, scheduler: RhythmScheduler | None = None
    ) -> str:
        key = frame.key.name if frame.key else "  --  "
        flash = "*" if frame.time_s <= self._onset_flash_until else " "
        parts = [
            f"{frame.time_s:7.2f}s",
            f"key {key:<7}",
            f"conf {frame.key_confidence:4.2f}",
            f"lvl {bar(frame.level, 10)}",
            f"dens {bar(frame.density, 10)}",
            f"onset {flash}",
        ]
        if scheduler is not None and self.show_pattern:
            slot = scheduler.next_step % scheduler.config.steps
            parts.append(f"kick {pattern_str(scheduler.pattern('kick'), slot)}")
        return " | ".join(parts)

    def close(self) -> None:
        self.stream.write("\n")
        self.stream.flush()
