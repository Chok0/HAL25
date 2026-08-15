import numpy as np
import pytest

from halj.audio.synth import MAJOR_TRIAD, MINOR_TRIAD, strum
from halj.config import Config


@pytest.fixture
def config():
    return Config()


@pytest.fixture
def samplerate(config):
    return config.audio.samplerate


def click_train(times_s, samplerate, duration_s, amplitude=0.6, width_s=0.004):
    """Signal de test : des impulsions bruitees a des dates connues.

    Volontairement plus simple qu'une guitare — sert a verifier que le
    detecteur trouve les attaques *la ou elles sont*, sans marge d'interpretation.
    """
    rng = np.random.default_rng(7)
    signal = np.zeros(int(duration_s * samplerate))
    width = int(width_s * samplerate)
    for time_s in times_s:
        start = int(time_s * samplerate)
        burst = rng.uniform(-1, 1, width) * np.hanning(width) * amplitude
        end = min(start + width, signal.size)
        signal[start:end] += burst[: end - start]
        # queue resonante : sans elle le signal est un dirac, trop facile.
        tail = int(0.12 * samplerate)
        tail_end = min(start + tail, signal.size)
        t = np.arange(tail_end - start) / samplerate
        signal[start:tail_end] += (
            0.3 * amplitude * np.sin(2 * np.pi * 220 * t) * np.exp(-t / 0.05)
        )
    return signal


@pytest.fixture
def guitar_chord(samplerate):
    """Un accord de La mineur gratte, 2 secondes."""
    return strum(45, MINOR_TRIAD, 2.0, samplerate, rng=np.random.default_rng(3))


@pytest.fixture
def guitar_major_chord(samplerate):
    """Un accord de Do majeur gratte, 2 secondes."""
    return strum(48, MAJOR_TRIAD, 2.0, samplerate, rng=np.random.default_rng(4))
