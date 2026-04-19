"""Shared pytest fixtures for Murmur tests."""

from __future__ import annotations

import wave

import numpy as np
import pytest


@pytest.fixture
def white_image() -> np.ndarray:
    """64x128 array of all ones (fully bright)."""
    return np.ones((64, 128), dtype=np.float64)


@pytest.fixture
def black_image() -> np.ndarray:
    """64x128 array of all zeros (fully dark)."""
    return np.zeros((64, 128), dtype=np.float64)


@pytest.fixture
def gradient_image() -> np.ndarray:
    """64x128 float64 array with values from 0 to 1 in raster order."""
    return np.linspace(0, 1, 64 * 128).reshape(64, 128).astype(np.float64)


@pytest.fixture
def checkerboard_image() -> np.ndarray:
    """64x128 alternating 0 and 1 checkerboard pattern."""
    rows, cols = 64, 128
    arr = np.zeros((rows, cols), dtype=np.float64)
    arr[::2, ::2] = 1.0
    arr[1::2, 1::2] = 1.0
    return arr


@pytest.fixture
def sine_audio() -> np.ndarray:
    """1D float64 array: 1 second at 44100 Hz, 440 Hz sine wave."""
    sr = 44100
    t = np.arange(sr, dtype=np.float64) / sr
    return np.sin(2.0 * np.pi * 440.0 * t)


@pytest.fixture
def silence_audio() -> np.ndarray:
    """1D float64 array of 44100 zeros (1 second of silence)."""
    return np.zeros(44100, dtype=np.float64)


@pytest.fixture
def test_wav_path(tmp_path, sine_audio):
    """Temporary WAV file containing the sine_audio fixture; returns the Path."""
    wav_path = tmp_path / "sine.wav"
    # Write manually using stdlib wave so the test has no dependency on murmur.audio_io
    int16 = (np.clip(sine_audio, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(44100)
        wf.writeframes(int16.tobytes())
    return wav_path


@pytest.fixture
def small_white_image() -> np.ndarray:
    """16x32 array of all ones — used in fast unit tests."""
    return np.ones((16, 32), dtype=np.float64)
