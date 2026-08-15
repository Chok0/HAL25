"""Detection d'attaques (onset detection) par flux spectral.

Choix de cadrage : onset detection plutot que beat tracking pour le v0. On ne
cherche pas a savoir ou est le temps, seulement quand le guitariste attaque —
c'est plus robuste, et suffit a piloter l'intensite du rythme genere.

Le seuil est adaptatif (mediane glissante x delta + plancher absolu) : un jeu
doux et un jeu fort declenchent tous les deux, sans avoir a recalibrer.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from ..config import AudioConfig, OnsetConfig
from .filters import CascadedHighpass


@dataclass(frozen=True)
class Onset:
    """Une attaque detectee."""

    time_s: float
    strength: float  # flux normalise par le seuil : 1.0 = pile au seuil


class OnsetDetector:
    """Flux spectral + seuil adaptatif + selection de pic.

    Le detecteur possede sa propre fenetre d'analyse : on lui pousse des hops,
    il entretient le tampon. C'est necessaire parce que le passe-haut est un
    filtre a etat — le filtrer par trame reinitialiserait sa memoire a chaque
    fenetre et fabriquerait une discontinuite a chaque bord.
    """

    def __init__(self, audio: AudioConfig, config: OnsetConfig):
        self.audio = audio
        self.config = config
        self.frame_size = config.frame_size
        self._window = np.hanning(self.frame_size).astype(np.float64)
        self._hop_s = audio.hop_size / audio.samplerate
        self._buffer = np.zeros(self.frame_size, dtype=np.float64)

        # Passe-haut reel dans le domaine temporel : le bruit de manche, les
        # frottements de mediator et les rumbles de la piece vivent sous
        # ~120 Hz. Se contenter de masquer les bins FFT ne suffit pas — la
        # fuite spectrale d'un grave fort ressort dans tout le spectre.
        self._highpass = CascadedHighpass(config.highpass_hz, audio.samplerate)

        freqs = np.fft.rfftfreq(self.frame_size, 1.0 / audio.samplerate)
        # Le masque de bins reste, en complement du filtre : il retire le
        # residu qui subsiste juste sous la coupure.
        self._band = freqs >= config.highpass_hz
        if not self._band.any():
            raise ValueError("highpass_hz au-dessus de toute la bande analysee")

        self._prev_magnitude: np.ndarray | None = None
        # Garde le rapport a 0/0 pendant le silence absolu plutot qu'a NaN.
        self._energy_epsilon = 1e-6 * float(np.count_nonzero(self._band))
        median_frames = max(3, int(round(config.median_window_s / self._hop_s)))
        self._history: deque[float] = deque(maxlen=median_frames)
        # Selection de pic : on ne valide une trame qu'apres avoir vu la
        # suivante, pour confirmer que le flux redescend (vrai maximum local).
        self._pending: tuple[float, float, float] | None = None
        self._last_onset_time: float = -np.inf
        self._time_s: float = 0.0
        self.last_flux: float = 0.0
        self.last_threshold: float = 0.0

    @property
    def median_frames(self) -> int:
        return self._history.maxlen or 3

    def reset(self) -> None:
        self._prev_magnitude = None
        self._history.clear()
        self._pending = None
        self._last_onset_time = -np.inf
        self._time_s = 0.0
        self.last_flux = 0.0
        self.last_threshold = 0.0
        self._buffer[:] = 0.0
        self._highpass.reset()

    def process(self, hop: np.ndarray, time_s: float | None = None) -> Onset | None:
        """Consomme un hop, retourne une attaque si elle vient d'etre confirmee.

        L'attaque retournee correspond au hop *precedent* (retard d'un hop,
        soit ~6 ms a 44.1 kHz) : c'est le prix de la confirmation du pic.
        """
        hop = np.asarray(hop, dtype=np.float64).reshape(-1)
        if hop.shape[0] != self.audio.hop_size:
            raise ValueError(
                f"hop de {hop.shape[0]} echantillons, "
                f"{self.audio.hop_size} attendus"
            )
        now = self._time_s if time_s is None else time_s
        self._time_s = now + self._hop_s

        self._buffer = np.concatenate(
            [self._buffer[hop.size :], self._highpass.process(hop)]
        )
        magnitude = np.abs(
            np.fft.rfft(self._buffer * self._window)
        )[self._band]

        if self._prev_magnitude is None:
            self._prev_magnitude = magnitude
            return None

        # Flux spectral demi-redresse : seules les montees d'energie comptent,
        # une extinction de note n'est pas une attaque.
        #
        # La normalisation par l'energie des deux trames est ce qui rend le
        # flux exploitable. Un flux brut suit la dynamique absolue : un jeu
        # fort ecrase un jeu doux, et surtout l'ondulation d'interference
        # d'une note *tenue* suffit a franchir n'importe quel seuil. Ici le
        # flux est un ratio dans [0, 1] — proportion d'energie qui vient
        # d'apparaitre — donc une note tenue tend vers 0 quel que soit son
        # niveau, et une attaque depuis le silence vaut 1 sans exploser.
        numerator = float(np.sum(np.maximum(magnitude - self._prev_magnitude, 0.0)))
        denominator = float(np.sum(magnitude) + np.sum(self._prev_magnitude))
        flux = numerator / (denominator + self._energy_epsilon)
        self._prev_magnitude = magnitude

        threshold = self._threshold()
        self._history.append(flux)
        self.last_flux = flux
        self.last_threshold = threshold

        return self._pick_peak(flux, threshold, now)

    def _threshold(self) -> float:
        if not self._history:
            return self.config.floor
        median = float(np.median(self._history))
        return max(self.config.floor, median * self.config.delta)

    def _pick_peak(self, flux: float, threshold: float, now: float) -> Onset | None:
        """Valide un maximum local du flux au-dessus du seuil."""
        onset: Onset | None = None
        if self._pending is not None:
            prev_flux, prev_threshold, prev_time = self._pending
            # Le pic est confirme si le flux redescend apres lui.
            if prev_flux >= flux and prev_flux > prev_threshold:
                if prev_time - self._last_onset_time >= self.config.min_interval_s:
                    self._last_onset_time = prev_time
                    onset = Onset(
                        time_s=prev_time,
                        strength=prev_flux / max(prev_threshold, 1e-9),
                    )
            self._pending = None

        if flux > threshold:
            self._pending = (flux, threshold, now)
        return onset


class AubioOnsetDetector:
    """Backend aubio, aligne sur l'interface d'`OnsetDetector`.

    aubio implemente des fonctions de detection eprouvees (complex domain, HFC,
    specflux). Dependance optionnelle : on ne l'instancie que sur demande.
    """

    def __init__(self, audio: AudioConfig, config: OnsetConfig, method: str = "complex"):
        import aubio  # import tardif : dependance optionnelle

        self.audio = audio
        self.config = config
        self.frame_size = config.frame_size
        self._hop_s = audio.hop_size / audio.samplerate
        self._detector = aubio.onset(
            method, config.frame_size, audio.hop_size, audio.samplerate
        )
        self._detector.set_minioi_s(config.min_interval_s)
        self._detector.set_threshold(config.aubio_threshold)
        self._time_s = 0.0
        self.last_flux = 0.0
        self.last_threshold = 0.0

    def reset(self) -> None:
        self._detector.reset()
        self._time_s = 0.0

    def process(self, hop: np.ndarray, time_s: float | None = None) -> Onset | None:
        now = self._time_s if time_s is None else time_s
        self._time_s = now + self._hop_s
        hop = np.ascontiguousarray(hop, dtype=np.float32)
        if hop.shape[0] != self.audio.hop_size:
            raise ValueError(
                f"hop de {hop.shape[0]} echantillons, "
                f"{self.audio.hop_size} attendus"
            )
        fired = bool(self._detector(hop)[0])
        self.last_flux = float(self._detector.get_descriptor())
        self.last_threshold = float(self._detector.get_thresholded_descriptor())
        if not fired:
            return None
        return Onset(time_s=now, strength=max(1.0, abs(self.last_flux)))


def create_onset_detector(
    audio: AudioConfig, config: OnsetConfig, backend: str = "numpy"
):
    """Fabrique le detecteur : "numpy" (defaut, sans dependance) ou "aubio"."""
    if backend == "numpy":
        return OnsetDetector(audio, config)
    if backend == "aubio":
        return AubioOnsetDetector(audio, config)
    raise ValueError(f"backend onset inconnu : {backend}")
