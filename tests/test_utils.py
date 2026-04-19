"""Unit tests for murmur.utils helper functions."""

from __future__ import annotations

import pytest
import numpy as np

from murmur.utils import (
    normalize,
    hann_window,
    db_to_amplitude,
    amplitude_to_db,
    parse_time,
    parse_resolution,
    clamp,
)


# ---------------------------------------------------------------------------
# normalize()
# ---------------------------------------------------------------------------

class TestNormalize:
    def test_peak_matches_headroom(self):
        sig = np.array([0.0, 0.5, -0.5, 0.25], dtype=np.float64)
        out = normalize(sig, headroom_db=-0.5)
        expected_peak = 10 ** (-0.5 / 20.0)
        assert abs(np.max(np.abs(out)) - expected_peak) < 1e-9

    def test_silent_input_returns_zeros(self):
        sig = np.zeros(100, dtype=np.float64)
        out = normalize(sig)
        assert np.all(out == 0.0)

    def test_output_shape_preserved(self):
        sig = np.random.rand(10, 2).astype(np.float64)
        out = normalize(sig)
        assert out.shape == sig.shape

    def test_does_not_mutate_input(self):
        sig = np.array([0.1, -0.2, 0.3], dtype=np.float64)
        orig = sig.copy()
        normalize(sig)
        np.testing.assert_array_equal(sig, orig)

    def test_invalid_target_peak_raises(self):
        sig = np.array([0.5, -0.5], dtype=np.float64)
        with pytest.raises(ValueError):
            normalize(sig, target_peak=0.0)
        with pytest.raises(ValueError):
            normalize(sig, target_peak=1.5)


# ---------------------------------------------------------------------------
# hann_window()
# ---------------------------------------------------------------------------

class TestHannWindow:
    def test_length(self):
        for n in (1, 2, 16, 512, 4096):
            assert len(hann_window(n)) == n

    def test_endpoints_near_zero(self):
        w = hann_window(1024)
        assert abs(w[0]) < 1e-9
        # Last point: cos(2*pi*(n-1)/n) ≈ cos(2*pi) ≈ 1 → w ≈ 0
        assert abs(w[-1]) < 0.01

    def test_peak_near_one(self):
        w = hann_window(1024)
        assert abs(np.max(w) - 1.0) < 1e-9

    def test_values_in_range(self):
        w = hann_window(64)
        assert np.all(w >= 0.0)
        assert np.all(w <= 1.0 + 1e-12)


# ---------------------------------------------------------------------------
# db_to_amplitude() and amplitude_to_db()
# ---------------------------------------------------------------------------

class TestDbConversions:
    def test_0db_is_amplitude_1(self):
        assert abs(db_to_amplitude(0.0) - 1.0) < 1e-12

    def test_minus_6db_is_half(self):
        # -6.0206 dB ≈ 0.5 amplitude
        assert abs(db_to_amplitude(-6.0206) - 0.5) < 0.001

    def test_amplitude_to_db_roundtrip(self):
        for amp in (0.1, 0.5, 1.0, 2.0):
            db = amplitude_to_db(amp)
            back = db_to_amplitude(db)
            assert abs(back - amp) < 1e-9

    def test_zero_amplitude_does_not_raise(self):
        # Uses epsilon internally; should not raise
        result = amplitude_to_db(0.0)
        assert result < -100  # very negative dB


# ---------------------------------------------------------------------------
# parse_time()
# ---------------------------------------------------------------------------

class TestParseTime:
    def test_plain_float(self):
        assert parse_time("154.0") == pytest.approx(154.0)

    def test_plain_int_string(self):
        assert parse_time("30") == pytest.approx(30.0)

    def test_mm_ss(self):
        assert parse_time("2:34") == pytest.approx(2 * 60 + 34)

    def test_hh_mm_ss(self):
        assert parse_time("1:02:34") == pytest.approx(3600 + 2 * 60 + 34)

    def test_zero(self):
        assert parse_time("0") == pytest.approx(0.0)
        assert parse_time("0:00") == pytest.approx(0.0)

    def test_fractional_seconds_in_mm_ss(self):
        assert parse_time("1:30.5") == pytest.approx(90.5)

    def test_invalid_string_raises(self):
        with pytest.raises(ValueError):
            parse_time("abc")

    def test_invalid_mm_ss_raises(self):
        with pytest.raises(ValueError):
            parse_time("x:y")

    def test_whitespace_stripped(self):
        assert parse_time("  10.0  ") == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# parse_resolution()
# ---------------------------------------------------------------------------

class TestParseResolution:
    def test_standard(self):
        assert parse_resolution("256x128") == (256, 128)

    def test_square(self):
        assert parse_resolution("64x64") == (64, 64)

    def test_case_insensitive(self):
        assert parse_resolution("256X128") == (256, 128)

    def test_large_values(self):
        assert parse_resolution("1920x1080") == (1920, 1080)

    def test_missing_x_raises(self):
        with pytest.raises(ValueError):
            parse_resolution("256-128")

    def test_non_numeric_raises(self):
        with pytest.raises(ValueError):
            parse_resolution("axb")

    def test_missing_height_raises(self):
        with pytest.raises(ValueError):
            parse_resolution("256x")


# ---------------------------------------------------------------------------
# clamp()
# ---------------------------------------------------------------------------

class TestClamp:
    def test_within_range(self):
        assert clamp(0.5, 0.0, 1.0) == 0.5

    def test_below_lo(self):
        assert clamp(-0.5, 0.0, 1.0) == 0.0

    def test_above_hi(self):
        assert clamp(1.5, 0.0, 1.0) == 1.0

    def test_at_boundaries(self):
        assert clamp(0.0, 0.0, 1.0) == 0.0
        assert clamp(1.0, 0.0, 1.0) == 1.0

    def test_negative_range(self):
        assert clamp(-5.0, -10.0, -1.0) == -5.0
        assert clamp(0.0, -10.0, -1.0) == -1.0
