"""Tests for murmur.dither.apply_dither()."""

from __future__ import annotations

import numpy as np
import pytest

from murmur.dither import apply_dither


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def gradient():
    """16x32 float64 gradient image."""
    return np.linspace(0.0, 1.0, 16 * 32).reshape(16, 32).astype(np.float64)


@pytest.fixture
def small_gray():
    """8x8 mid-gray image (all 0.5)."""
    return np.full((8, 8), 0.5, dtype=np.float64)


# ---------------------------------------------------------------------------
# 1. floyd-steinberg
# ---------------------------------------------------------------------------

class TestFloydSteinberg:
    def test_output_shape_matches(self, gradient):
        result = apply_dither(gradient, method="floyd-steinberg")
        assert result.shape == gradient.shape

    def test_output_values_in_range(self, gradient):
        result = apply_dither(gradient, method="floyd-steinberg")
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_output_values_binary(self, gradient):
        result = apply_dither(gradient, method="floyd-steinberg")
        unique = set(np.unique(result).tolist())
        assert unique <= {0.0, 1.0}, f"Expected only 0.0/1.0, got {unique}"


# ---------------------------------------------------------------------------
# 2. ordered
# ---------------------------------------------------------------------------

class TestOrdered:
    def test_output_shape_matches(self, gradient):
        result = apply_dither(gradient, method="ordered")
        assert result.shape == gradient.shape

    def test_output_values_binary(self, gradient):
        result = apply_dither(gradient, method="ordered")
        unique = set(np.unique(result).tolist())
        assert unique <= {0.0, 1.0}, f"Expected only 0.0/1.0, got {unique}"

    def test_output_in_range(self, gradient):
        result = apply_dither(gradient, method="ordered")
        assert result.min() >= 0.0
        assert result.max() <= 1.0


# ---------------------------------------------------------------------------
# 3. threshold
# ---------------------------------------------------------------------------

class TestThreshold:
    def test_exact_binary(self, gradient):
        result = apply_dither(gradient, method="threshold")
        unique = set(np.unique(result).tolist())
        assert unique <= {0.0, 1.0}, f"Expected exactly {{0.0, 1.0}}, got {unique}"

    def test_shape_preserved(self, gradient):
        result = apply_dither(gradient, method="threshold")
        assert result.shape == gradient.shape

    def test_custom_threshold(self):
        image = np.array([[0.3, 0.7], [0.4, 0.9]], dtype=np.float64)
        result = apply_dither(image, method="threshold", threshold=0.5)
        expected = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=np.float64)
        assert np.array_equal(result, expected), (
            f"Threshold 0.5 result wrong: {result}"
        )


# ---------------------------------------------------------------------------
# 4. none
# ---------------------------------------------------------------------------

class TestNone:
    def test_returns_equal_array(self, gradient):
        result = apply_dither(gradient, method="none")
        assert np.array_equal(result, gradient), (
            "method='none' should return input unchanged"
        )

    def test_is_a_copy(self, gradient):
        result = apply_dither(gradient, method="none")
        # Modifying result should not affect original
        result[0, 0] = 999.0
        assert gradient[0, 0] != 999.0, "method='none' should return a copy, not a view"


# ---------------------------------------------------------------------------
# 5. All methods: values in [0, 1]
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", ["floyd-steinberg", "ordered", "threshold", "none"])
def test_all_methods_values_in_range(method, gradient):
    result = apply_dither(gradient, method=method)
    assert result.min() >= 0.0
    assert result.max() <= 1.0


# ---------------------------------------------------------------------------
# 6. All methods: input shape is preserved
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", ["floyd-steinberg", "ordered", "threshold", "none"])
def test_all_methods_shape_preserved(method, gradient):
    result = apply_dither(gradient, method=method)
    assert result.shape == gradient.shape


# ---------------------------------------------------------------------------
# 7. Invalid method raises ValueError
# ---------------------------------------------------------------------------

class TestInvalidMethod:
    def test_bad_method_raises(self, gradient):
        with pytest.raises(ValueError, match="Unknown dither method"):
            apply_dither(gradient, method="bogus")

    def test_empty_string_raises(self, gradient):
        with pytest.raises(ValueError):
            apply_dither(gradient, method="")


# ---------------------------------------------------------------------------
# 8. Non-2D input raises ValueError
# ---------------------------------------------------------------------------

class TestNon2DInput:
    def test_1d_raises(self):
        with pytest.raises(ValueError, match="2D"):
            apply_dither(np.ones(64, dtype=np.float64))

    def test_3d_raises(self):
        with pytest.raises(ValueError, match="2D"):
            apply_dither(np.ones((4, 4, 3), dtype=np.float64))
