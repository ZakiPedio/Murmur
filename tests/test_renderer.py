"""Tests for murmur.renderer renderers."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_test_png(tmp_path, width=64, height=32, value=128):
    """Create a solid-gray PNG in tmp_path and return its path string."""
    img = Image.new("L", (width, height), value)
    path = tmp_path / "test.png"
    img.save(str(path))
    return str(path)


def _make_two_pngs(tmp_path):
    """Create two small test PNGs and return their paths as strings."""
    paths = []
    for i, val in enumerate([80, 180]):
        img = Image.new("L", (32, 32), val)
        p = tmp_path / f"frame{i}.png"
        img.save(str(p))
        paths.append(str(p))
    return paths


# ---------------------------------------------------------------------------
# 1. render_image
# ---------------------------------------------------------------------------

class TestRenderImage:
    def test_shape(self, tmp_path):
        from murmur.renderer import render_image
        path = _make_test_png(tmp_path, width=64, height=32)
        arr = render_image(path, width=64, height=32)
        assert arr.shape == (32, 64)

    def test_values_in_range(self, tmp_path):
        from murmur.renderer import render_image
        path = _make_test_png(tmp_path, width=64, height=32)
        arr = render_image(path, width=64, height=32)
        assert arr.min() >= 0.0
        assert arr.max() <= 1.0

    def test_invert(self, tmp_path):
        from murmur.renderer import render_image
        path = _make_test_png(tmp_path, width=64, height=32, value=200)
        normal = render_image(path, width=64, height=32, invert=False)
        inverted = render_image(path, width=64, height=32, invert=True)
        assert np.allclose(inverted, 1.0 - normal, atol=1e-6)

    def test_missing_file_raises(self, tmp_path):
        from murmur.renderer import render_image
        with pytest.raises(FileNotFoundError):
            render_image(str(tmp_path / "nonexistent.png"), width=64, height=32)


# ---------------------------------------------------------------------------
# 2. render_text
# ---------------------------------------------------------------------------

class TestRenderText:
    def test_non_empty_output(self):
        from murmur.renderer import render_text
        arr = render_text("HELLO")
        assert arr.size > 0

    def test_values_in_range(self):
        from murmur.renderer import render_text
        arr = render_text("TEST")
        assert arr.min() >= 0.0
        assert arr.max() <= 1.0

    def test_2d_output(self):
        from murmur.renderer import render_text
        arr = render_text("HELLO")
        assert arr.ndim == 2

    def test_multiline(self):
        from murmur.renderer import render_text
        arr = render_text("\nHELLO\nWORLD")
        assert arr.size > 0
        assert arr.ndim == 2

    def test_with_explicit_size(self):
        from murmur.renderer import render_text
        arr = render_text("HI", width=128, height=64)
        assert arr.shape == (64, 128)


# ---------------------------------------------------------------------------
# 3. render_ascii
# ---------------------------------------------------------------------------

class TestRenderAscii:
    def test_2d_output(self):
        from murmur.renderer import render_ascii
        arr = render_ascii("Hello\nWorld")
        assert arr.ndim == 2

    def test_values_in_range(self):
        from murmur.renderer import render_ascii
        arr = render_ascii("TEST")
        assert arr.min() >= 0.0
        assert arr.max() <= 1.0

    def test_non_empty(self):
        from murmur.renderer import render_ascii
        arr = render_ascii("ABC")
        assert arr.size > 0


# ---------------------------------------------------------------------------
# 4. render_qr
# ---------------------------------------------------------------------------

class TestRenderQr:
    def test_roughly_square(self):
        from murmur.renderer import render_qr
        arr = render_qr("https://example.com")
        h, w = arr.shape
        aspect = h / w
        assert 0.8 < aspect < 1.2, f"QR output not roughly square: {h}x{w}"

    def test_values_in_range(self):
        from murmur.renderer import render_qr
        arr = render_qr("test")
        assert arr.min() >= 0.0
        assert arr.max() <= 1.0

    def test_not_all_zeros(self):
        from murmur.renderer import render_qr
        arr = render_qr("data")
        assert arr.max() > 0.0, "QR output should not be all zeros"

    def test_explicit_size(self):
        from murmur.renderer import render_qr
        arr = render_qr("hello", width=128, height=128)
        assert arr.shape == (128, 128)

    def test_qr_with_logo_differs_from_qr_without(self, tmp_path):
        from murmur.renderer import render_qr
        from PIL import Image as _Image

        # Create a small solid logo image
        logo_path = tmp_path / "logo.png"
        _Image.new("RGB", (20, 20), (255, 0, 0)).save(str(logo_path))

        arr_plain = render_qr("https://example.com")
        arr_logo = render_qr("https://example.com", logo_path=str(logo_path))

        # Both must be valid 2D float arrays
        assert arr_plain.ndim == 2
        assert arr_logo.ndim == 2
        assert arr_plain.min() >= 0.0 and arr_plain.max() <= 1.0
        assert arr_logo.min() >= 0.0 and arr_logo.max() <= 1.0

        # Logo version must differ from plain version (logo was pasted over center)
        assert not np.array_equal(arr_plain, arr_logo), (
            "QR with logo should differ from QR without logo"
        )


# ---------------------------------------------------------------------------
# 5. render_math
# ---------------------------------------------------------------------------

class TestRenderMath:
    def test_sin_non_empty(self):
        from murmur.renderer import render_math
        arr = render_math("sin(x)", width=128, height=64)
        assert arr.size > 0

    def test_not_constant(self):
        from murmur.renderer import render_math
        arr = render_math("sin(x)", width=128, height=64)
        assert arr.std() > 0.01, "sin(x) plot should not be a constant image"

    def test_values_in_range(self):
        from murmur.renderer import render_math
        arr = render_math("cos(x)", width=64, height=32)
        assert arr.min() >= 0.0
        assert arr.max() <= 1.0

    def test_correct_shape(self):
        from murmur.renderer import render_math
        arr = render_math("sin(x)", width=100, height=50)
        assert arr.shape == (50, 100)


# ---------------------------------------------------------------------------
# 6. render_sequence
# ---------------------------------------------------------------------------

class TestRenderSequence:
    def test_two_images_width(self, tmp_path):
        from murmur.renderer import render_sequence
        paths = _make_two_pngs(tmp_path)
        frame_width = 32
        arr = render_sequence(paths=paths, frame_width=frame_width, height=32)
        expected_width = frame_width * len(paths)
        assert arr.shape[1] == expected_width, (
            f"Expected width {expected_width}, got {arr.shape[1]}"
        )

    def test_output_values_in_range(self, tmp_path):
        from murmur.renderer import render_sequence
        paths = _make_two_pngs(tmp_path)
        arr = render_sequence(paths=paths, frame_width=32, height=32)
        assert arr.min() >= 0.0
        assert arr.max() <= 1.0

    def test_2d_output(self, tmp_path):
        from murmur.renderer import render_sequence
        paths = _make_two_pngs(tmp_path)
        arr = render_sequence(paths=paths, frame_width=32, height=32)
        assert arr.ndim == 2
