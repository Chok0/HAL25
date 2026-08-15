"""Sources audio : micro/interface, fichier WAV, guitare synthetique.

Toutes exposent la meme interface (`iter_blocks`), donc le runtime ne sait pas
d'ou vient le son. C'est ce qui permet de rejouer une prise en boucle pour
calibrer les seuils sans avoir la guitare a la main.
"""

from __future__ import annotations

import queue
import wave
from typing import Iterator, Protocol

import numpy as np

from ..config import AudioConfig
from .synth import chord_progression


class AudioSource(Protocol):
    """Fournit des blocs mono float64 dans [-1, 1]."""

    samplerate: int
    realtime: bool  # True si la source impose deja son propre debit

    def iter_blocks(self) -> Iterator[np.ndarray]: ...

    def close(self) -> None: ...


class MicSource:
    """Entree micro / interface son via sounddevice (PortAudio).

    Le callback PortAudio ne fait que deposer une copie du bloc dans une file :
    aucun calcul, aucune allocation lourde, aucun verrou cote temps reel. Toute
    l'analyse se passe dans le thread appelant.
    """

    realtime = True

    def __init__(self, config: AudioConfig, max_queue: int = 32):
        import sounddevice as sd  # import tardif : dependance optionnelle

        self._sd = sd
        self.config = config
        self.samplerate = config.samplerate
        self._queue: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=max_queue)
        self.dropped_blocks = 0
        self._stream = sd.InputStream(
            samplerate=config.samplerate,
            blocksize=config.block_size,
            channels=config.channels,
            dtype="float32",
            device=config.device,
            callback=self._callback,
        )

    def _callback(self, indata, frames, time_info, status) -> None:  # noqa: ANN001
        if status:  # xruns : on veut le savoir, sans polluer le temps reel
            self.dropped_blocks += 1
        try:
            self._queue.put_nowait(indata[:, 0].astype(np.float64, copy=True))
        except queue.Full:
            # Le consommateur a pris du retard : on jette le bloc le plus
            # recent plutot que de bloquer le thread audio.
            self.dropped_blocks += 1

    def iter_blocks(self) -> Iterator[np.ndarray]:
        self._stream.start()
        try:
            while True:
                block = self._queue.get()
                if block is None:
                    return
                yield block
        finally:
            self._stream.stop()

    def close(self) -> None:
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        self._stream.close()


class WavFileSource:
    """Lecture d'un WAV PCM (module `wave` de la stdlib, aucune dependance)."""

    realtime = False

    def __init__(self, path: str, config: AudioConfig, loop: bool = False):
        self.path = path
        self.config = config
        self.loop = loop
        with wave.open(path, "rb") as handle:
            self.samplerate = handle.getframerate()
            self.channels = handle.getnchannels()
            self.sample_width = handle.getsampwidth()
        if self.sample_width not in (2, 4):
            raise ValueError(
                f"WAV {self.sample_width * 8} bits non gere : fournir du PCM 16 ou 32 bits"
            )
        if self.samplerate != config.samplerate:
            raise ValueError(
                f"le fichier est a {self.samplerate} Hz, la config a "
                f"{config.samplerate} Hz : reechantillonner ou ajuster la config"
            )

    def iter_blocks(self) -> Iterator[np.ndarray]:
        dtype = np.int16 if self.sample_width == 2 else np.int32
        scale = float(np.iinfo(dtype).max)
        while True:
            with wave.open(self.path, "rb") as handle:
                while True:
                    raw = handle.readframes(self.config.block_size)
                    if not raw:
                        break
                    data = np.frombuffer(raw, dtype=dtype).astype(np.float64) / scale
                    if self.channels > 1:
                        data = data.reshape(-1, self.channels).mean(axis=1)
                    yield data
            if not self.loop:
                return

    def close(self) -> None:
        return None


class SyntheticSource:
    """Guitare synthetique : une progression jouee en boucle.

    Permet de faire tourner la chaine complete — analyse comprise — sans
    materiel, ce qui est precieux pour valider le protocole OSC et le rendu
    SuperCollider avant d'avoir cable quoi que ce soit.
    """

    realtime = False

    def __init__(
        self,
        config: AudioConfig,
        roots_midi: list[int] | None = None,
        modes: list[str] | None = None,
        chord_duration_s: float = 2.5,
        loop: bool = True,
        seed: int = 0,
    ):
        self.config = config
        self.samplerate = config.samplerate
        self.loop = loop
        # Par defaut : i - VI - III - VII en La mineur, terrain naturel du drone.
        self.roots_midi = roots_midi or [45, 41, 48, 43]
        self.modes = modes or ["min", "maj", "maj", "maj"]
        self._audio = chord_progression(
            self.roots_midi,
            config.samplerate,
            modes=self.modes,
            chord_duration_s=chord_duration_s,
            seed=seed,
        )

    @property
    def audio(self) -> np.ndarray:
        return self._audio

    def iter_blocks(self) -> Iterator[np.ndarray]:
        size = self.config.block_size
        while True:
            for start in range(0, self._audio.size, size):
                yield self._audio[start : start + size]
            if not self.loop:
                return

    def close(self) -> None:
        return None


def create_source(
    kind: str,
    config: AudioConfig,
    path: str | None = None,
    loop: bool = False,
) -> AudioSource:
    """Fabrique une source : "mic", "file" (requiert `path`) ou "synth"."""
    if kind == "mic":
        return MicSource(config)
    if kind == "file":
        if not path:
            raise ValueError("source 'file' : --input est requis")
        return WavFileSource(path, config, loop=loop)
    if kind == "synth":
        return SyntheticSource(config, loop=True)
    raise ValueError(f"source inconnue : {kind}")
