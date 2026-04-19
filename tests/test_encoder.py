"""Tests for murmur.encoder.encode()."""

from __future__ import annotations

import numpy as np
import pytest

from murmur.encoder import encode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rms(samples: np.ndarray) -> float:
    return float(np.sqrt(np.mean(samples ** 2)))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestEncodeWhiteImage:
    """White image should produce audible, non-silent output."""

    def test_rms_above_threshold(self, white_image):
        audio = encode(white_image, sample_rate=44100, duration=1.0)
        assert _rms(audio) > 0.01, "White image should produce signal with RMS > 0.01"


class TestEncodeBlackImage:
    """Black image (all zeros) should produce near-silence."""

    def test_near_silence(self, black_image):
        audio = encode(black_image, sample_rate=44100, duration=1.0)
        assert np.allclose(audio, 0.0, atol=1e-9), (
            "Black image should produce all-zero audio"
        )


class TestEncodeSingleRow:
    """Single bright row should concentrate energy near that frequency."""

    def test_energy_peaks_near_target_frequency(self):
        freq_bins, time_bins = 32, 16
        image = np.zeros((freq_bins, time_bins), dtype=np.float64)
        # Light only row 0, which maps to freq_max
        image[0, :] = 1.0

        freq_min = 200.0
        freq_max = 4000.0
        sr = 44100
        duration = 1.0

        audio = encode(
            image,
            sample_rate=sr,
            duration=duration,
            freq_min=freq_min,
            freq_max=freq_max,
            log_frequency=True,
        )

        # FFT to check that energy is near freq_max
        fft_mag = np.abs(np.fft.rfft(audio))
        freqs = np.fft.rfftfreq(len(audio), d=1.0 / sr)
        peak_freq = freqs[np.argmax(fft_mag)]

        # The dominant energy should be within 20% of freq_max
        assert abs(peak_freq - freq_max) / freq_max < 0.20, (
            f"Expected peak near {freq_max} Hz, got {peak_freq:.1f} Hz"
        )


class TestEncodeOutputLength:
    """Output length must equal int(sample_rate * duration)."""

    @pytest.mark.parametrize("sr,dur", [
        (44100, 1.0),
        (22050, 2.0),
        (44100, 0.5),
    ])
    def test_output_length(self, small_white_image, sr, dur):
        audio = encode(small_white_image, sample_rate=sr, duration=dur)
        expected = int(sr * dur)
        assert len(audio) == expected, (
            f"Expected {expected} samples, got {len(audio)}"
        )


class TestEncodeFrequencyScaling:
    """Both log and linear frequency scaling should complete without error."""

    def test_log_scaling(self, small_white_image):
        audio = encode(
            small_white_image, sample_rate=44100, duration=0.5,
            log_frequency=True,
        )
        assert audio.ndim == 1
        assert np.all(np.isfinite(audio))

    def test_linear_scaling(self, small_white_image):
        audio = encode(
            small_white_image, sample_rate=44100, duration=0.5,
            log_frequency=False,
        )
        assert audio.ndim == 1
        assert np.all(np.isfinite(audio))


class TestEncodeNormalization:
    """Peak amplitude should be close to the target implied by headroom_db."""

    @pytest.mark.parametrize("headroom_db", [-0.5, -3.0, -6.0])
    def test_peak_amplitude(self, white_image, headroom_db):
        audio = encode(
            white_image, sample_rate=44100, duration=1.0,
            headroom_db=headroom_db,
        )
        target = 10 ** (headroom_db / 20.0)
        peak = float(np.max(np.abs(audio)))
        # Allow 1% relative tolerance
        assert abs(peak - target) / target < 0.01, (
            f"Peak {peak:.4f} too far from target {target:.4f} "
            f"(headroom_db={headroom_db})"
        )


class TestEncodePhaseContinuity:
    """Adjacent sample differences should not exceed 0.5 (no harsh clicks)."""

    def test_no_large_jumps(self, gradient_image):
        audio = encode(
            gradient_image, sample_rate=44100, duration=1.0,
        )
        diffs = np.abs(np.diff(audio))
        max_jump = float(diffs.max())
        assert max_jump < 0.5, f"Large phase discontinuity detected: max jump = {max_jump:.4f}"


class TestEncodeInvalidInput:
    """Invalid inputs should raise ValueError."""

    def test_1d_array_raises(self):
        with pytest.raises(ValueError, match="2D"):
            encode(np.ones(64, dtype=np.float64))

    def test_out_of_range_values_raises(self):
        bad = np.ones((8, 8), dtype=np.float64) * 1.5  # values > 1
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            encode(bad)

    def test_negative_values_raises(self):
        bad = np.full((8, 8), -0.1, dtype=np.float64)
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            encode(bad)

    def test_zero_time_bins_raises(self):
        bad = np.zeros((8, 0), dtype=np.float64)
        with pytest.raises(ValueError, match="at least 1"):
            encode(bad)

    def test_zero_freq_bins_raises(self):
        bad = np.zeros((0, 8), dtype=np.float64)
        with pytest.raises(ValueError, match="at least 1"):
            encode(bad)


class TestEncodeRandomizePhase:
    """randomize_phase should produce valid audio that differs from zero-phase."""

    def test_output_is_finite(self, white_image):
        audio = encode(white_image, sample_rate=44100, duration=0.5, randomize_phase=True)
        assert np.all(np.isfinite(audio)), "randomize_phase output contains non-finite values"

    def test_output_has_signal(self, white_image):
        audio = encode(white_image, sample_rate=44100, duration=0.5, randomize_phase=True)
        assert _rms(audio) > 0.01, "randomize_phase output is unexpectedly silent"

    def test_differs_from_zero_phase(self, white_image):
        """Two randomized-phase encodes should not be identical (different seeds)."""
        a1 = encode(white_image, sample_rate=44100, duration=0.5, randomize_phase=True)
        a2 = encode(white_image, sample_rate=44100, duration=0.5, randomize_phase=True)
        # Both are valid audio; while they could theoretically match it is astronomically unlikely
        assert not np.array_equal(a1, a2), (
            "Two randomize_phase encodes should not produce identical output"
        )


class TestEncodeSeed:
    """--seed makes randomize_phase reproducible."""

    def test_same_seed_produces_identical_output(self, white_image):
        a1 = encode(white_image, sample_rate=44100, duration=0.5, randomize_phase=True, seed=42)
        a2 = encode(white_image, sample_rate=44100, duration=0.5, randomize_phase=True, seed=42)
        assert np.array_equal(a1, a2), "Same seed should produce identical output"

    def test_different_seeds_produce_different_output(self, white_image):
        a1 = encode(white_image, sample_rate=44100, duration=0.5, randomize_phase=True, seed=1)
        a2 = encode(white_image, sample_rate=44100, duration=0.5, randomize_phase=True, seed=2)
        assert not np.array_equal(a1, a2), "Different seeds should produce different output"

    def test_seed_ignored_without_randomize_phase(self, white_image):
        """seed= has no effect when randomize_phase is False."""
        a1 = encode(white_image, sample_rate=44100, duration=0.5, randomize_phase=False, seed=99)
        a2 = encode(white_image, sample_rate=44100, duration=0.5, randomize_phase=False)
        assert np.array_equal(a1, a2), "seed should have no effect without randomize_phase"
