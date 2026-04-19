"""Tests for murmur.audio_io.read_audio() and write_audio()."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from murmur.audio_io import read_audio, write_audio


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sine(sr: int = 44100, duration: float = 0.1, freq: float = 440.0) -> np.ndarray:
    t = np.arange(int(sr * duration), dtype=np.float64) / sr
    return np.sin(2.0 * np.pi * freq * t)


def _make_stereo(sr: int = 44100, duration: float = 0.1) -> np.ndarray:
    mono = _make_sine(sr, duration)
    return np.column_stack([mono, mono * 0.5])


# ---------------------------------------------------------------------------
# Roundtrip: mono WAV
# ---------------------------------------------------------------------------

class TestMonoWavRoundtrip:
    def test_samples_close_after_roundtrip(self, tmp_path):
        original = _make_sine()
        path = str(tmp_path / "mono.wav")
        write_audio(path, original, sample_rate=44100)
        loaded, _ = read_audio(path)
        # 16-bit quantization: tolerate ~1/32768 ≈ 0.00003, use 0.001
        assert np.allclose(original, loaded, atol=0.001), (
            "Mono WAV roundtrip: samples differ more than 16-bit quantization tolerance"
        )

    def test_sample_rate_preserved(self, tmp_path):
        original = _make_sine()
        path = str(tmp_path / "rate_test.wav")
        write_audio(path, original, sample_rate=44100)
        _, sr = read_audio(path)
        assert sr == 44100

    def test_output_shape_mono(self, tmp_path):
        original = _make_sine()
        path = str(tmp_path / "shape_mono.wav")
        write_audio(path, original, sample_rate=44100)
        loaded, _ = read_audio(path)
        assert loaded.ndim == 1
        assert len(loaded) == len(original)


# ---------------------------------------------------------------------------
# Roundtrip: stereo WAV
# ---------------------------------------------------------------------------

class TestStereoWavRoundtrip:
    def test_stereo_shape_preserved(self, tmp_path):
        original = _make_stereo()
        path = str(tmp_path / "stereo.wav")
        write_audio(path, original, sample_rate=44100)
        loaded, _ = read_audio(path)
        assert loaded.shape == original.shape

    def test_stereo_values_close(self, tmp_path):
        original = _make_stereo()
        path = str(tmp_path / "stereo_vals.wav")
        write_audio(path, original, sample_rate=44100)
        loaded, _ = read_audio(path)
        assert np.allclose(original, loaded, atol=0.001)

    def test_stereo_sample_rate_preserved(self, tmp_path):
        original = _make_stereo()
        path = str(tmp_path / "stereo_sr.wav")
        write_audio(path, original, sample_rate=22050)
        _, sr = read_audio(path)
        assert sr == 22050


# ---------------------------------------------------------------------------
# Non-WAV without ffmpeg
# ---------------------------------------------------------------------------

class TestNonWavWithoutFfmpeg:
    def test_raises_runtime_error_for_mp3_without_ffmpeg(self, tmp_path):
        """When ffmpeg is absent, reading a non-WAV file must raise RuntimeError."""
        fake_mp3 = tmp_path / "audio.mp3"
        fake_mp3.write_bytes(b"\xff\xfb\x90\x00" * 16)  # fake MP3 header bytes

        with patch("shutil.which", return_value=None):
            with pytest.raises(RuntimeError):
                read_audio(str(fake_mp3))

    def test_raises_runtime_error_for_write_mp3_without_ffmpeg(self, tmp_path):
        """Writing a non-WAV format without ffmpeg must raise RuntimeError."""
        samples = _make_sine()
        out = str(tmp_path / "out.mp3")
        with patch("shutil.which", return_value=None):
            with pytest.raises(RuntimeError):
                write_audio(out, samples, sample_rate=44100)


# ---------------------------------------------------------------------------
# Missing file
# ---------------------------------------------------------------------------

class TestMissingFile:
    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            read_audio(str(tmp_path / "does_not_exist.wav"))
