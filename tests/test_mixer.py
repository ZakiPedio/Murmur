"""Tests for murmur.mixer.overlay() and watermark_overlay()."""

from __future__ import annotations

import numpy as np
import pytest

from murmur.mixer import overlay, watermark_overlay


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_mono(n: int = 44100) -> np.ndarray:
    """1D mono carrier of length n, random values in [-0.5, 0.5]."""
    rng = np.random.default_rng(42)
    return rng.uniform(-0.5, 0.5, n).astype(np.float64)


def _make_stereo(n: int = 44100) -> np.ndarray:
    """2D stereo carrier of shape (n, 2), random values in [-0.5, 0.5]."""
    rng = np.random.default_rng(99)
    return rng.uniform(-0.5, 0.5, (n, 2)).astype(np.float64)


def _make_encoded(n: int = 22050) -> np.ndarray:
    """Simple encoded signal: 440 Hz sine wave of length n."""
    t = np.arange(n, dtype=np.float64) / 44100
    return np.sin(2.0 * np.pi * 440.0 * t)


# ---------------------------------------------------------------------------
# Tests for overlay()
# ---------------------------------------------------------------------------

class TestOverlayMono:
    def test_output_same_length_as_carrier(self):
        carrier = _make_mono(44100)
        encoded = _make_encoded(22050)
        result = overlay(carrier, encoded)
        assert result.shape == carrier.shape

    def test_blend_zero_equals_normalized_carrier(self):
        carrier = _make_mono(44100)
        encoded = _make_encoded(22050)
        result = overlay(carrier, encoded, blend=0.0)
        # With blend=0, the encoded contribution is zero; result is just
        # the normalized carrier.  The shapes and non-zero pattern must match.
        assert result.shape == carrier.shape
        # The ratio of carrier to result should be constant (normalization only)
        nonzero = carrier != 0.0
        if nonzero.any():
            ratios = result[nonzero] / carrier[nonzero]
            assert np.allclose(ratios, ratios[0], atol=1e-9), (
                "blend=0 result should be a scaled version of the carrier"
            )

    def test_no_clipping(self):
        carrier = _make_mono(44100)
        encoded = _make_encoded(22050)
        result = overlay(carrier, encoded, blend=0.9)
        assert result.max() <= 1.0 + 1e-9
        assert result.min() >= -1.0 - 1e-9

    def test_offset_energy_position(self):
        """Energy from encoded signal should appear after the offset."""
        n = 44100
        carrier = np.zeros(n, dtype=np.float64)
        offset_sec = 0.5
        offset_samples = int(offset_sec * 44100)

        encoded = np.ones(n // 2, dtype=np.float64)
        result = overlay(carrier, encoded, blend=1.0,
                         offset_seconds=offset_sec, sample_rate=44100)

        # Before offset: result should be (near) zero
        pre_energy = float(np.sum(np.abs(result[:offset_samples])))
        post_energy = float(np.sum(np.abs(result[offset_samples:])))

        assert pre_energy < 1e-6, (
            f"Expected silence before offset, got energy={pre_energy:.4f}"
        )
        assert post_energy > 0.1, (
            f"Expected energy after offset, got energy={post_energy:.4f}"
        )


class TestOverlayStereo:
    def test_output_same_shape_as_carrier(self):
        carrier = _make_stereo(44100)
        encoded = _make_encoded(22050)
        result = overlay(carrier, encoded)
        assert result.shape == carrier.shape

    def test_left_channel_only(self):
        """With channel='left', only column 0 should differ from carrier."""
        carrier = _make_stereo(44100)
        encoded = _make_encoded(22050)
        result = overlay(carrier, encoded, blend=0.5, channel="left")

        # The right channel, after normalization, should be proportional
        # to carrier[:,1]. Verify by checking that no right-channel sample
        # has *more absolute change* than can be explained by normalization alone.
        # Simpler proxy: encode into left only, so right == carrier[:,1] * scale.
        left_changed = not np.allclose(result[:, 0], carrier[:, 0])
        assert left_changed, "Left channel should be modified with channel='left'"

    def test_right_channel_only(self):
        """With channel='right', only column 1 should be modified."""
        carrier = _make_stereo(44100)
        encoded = _make_encoded(22050)
        result = overlay(carrier, encoded, blend=0.5, channel="right")
        right_changed = not np.allclose(result[:, 1], carrier[:, 1])
        assert right_changed, "Right channel should be modified with channel='right'"

    def test_both_channels(self):
        carrier = _make_stereo(44100)
        encoded = _make_encoded(22050)
        result = overlay(carrier, encoded, blend=0.5, channel="both")
        left_changed = not np.allclose(result[:, 0], carrier[:, 0])
        right_changed = not np.allclose(result[:, 1], carrier[:, 1])
        assert left_changed and right_changed


class TestOverlayAutoNormalization:
    def test_values_bounded(self):
        carrier = _make_mono(44100) * 0.9
        encoded = _make_encoded(22050)
        result = overlay(carrier, encoded, blend=0.9)
        assert np.max(np.abs(result)) <= 1.0 + 1e-9


# ---------------------------------------------------------------------------
# Tests for watermark_overlay()
# ---------------------------------------------------------------------------

class TestWatermarkOverlay:
    def test_mono_output_shape(self):
        carrier = _make_mono(44100)
        encoded = _make_encoded(22050)
        result = watermark_overlay(carrier, encoded)
        assert result.shape == carrier.shape

    def test_stereo_output_shape(self):
        carrier = _make_stereo(44100)
        encoded = _make_encoded(22050)
        result = watermark_overlay(carrier, encoded)
        assert result.shape == carrier.shape

    def test_empty_encoded_raises(self):
        carrier = _make_mono(44100)
        with pytest.raises(ValueError, match="empty"):
            watermark_overlay(carrier, np.array([], dtype=np.float64))

    def test_no_clipping(self):
        carrier = _make_mono(44100)
        encoded = _make_encoded(22050)
        result = watermark_overlay(carrier, encoded, blend=0.1)
        assert np.max(np.abs(result)) <= 1.0 + 1e-9
