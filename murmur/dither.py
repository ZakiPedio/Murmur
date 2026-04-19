"""Dithering algorithms for low-resolution spectrogram images."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

# 4x4 Bayer matrix for ordered dithering (values normalized to [0, 1))
_BAYER_4X4 = np.array([
    [ 0,  8,  2, 10],
    [12,  4, 14,  6],
    [ 3, 11,  1,  9],
    [15,  7, 13,  5],
], dtype=np.float64) / 16.0


def apply_dither(
    image: np.ndarray,
    method: str = "floyd-steinberg",
    threshold: float = 0.5,
) -> np.ndarray:
    """Apply dithering to a grayscale image.

    Args:
        image: 2D float array with values in [0.0, 1.0], shape (H, W).
        method: Dithering algorithm. One of:
            - "floyd-steinberg": Error diffusion (best quality, smooth gradients).
            - "ordered": Bayer 4x4 matrix (characteristic crosshatch pattern).
            - "threshold": Simple binary cutoff (sharpest edges, no gradation).
            - "none": Pass through unchanged.
        threshold: Cutoff value for the "threshold" method. Default 0.5.

    Returns:
        Dithered image, same shape as input, values in [0.0, 1.0].

    Raises:
        ValueError: If method is not recognized or image is not 2D.
    """
    if image.ndim != 2:
        raise ValueError(f"Expected 2D image array, got shape {image.shape}")
    if method not in ("floyd-steinberg", "ordered", "threshold", "none"):
        raise ValueError(
            f"Unknown dither method: {method!r}. "
            "Choose from: floyd-steinberg, ordered, threshold, none"
        )

    if method == "none":
        return image.copy()
    elif method == "threshold":
        return _threshold_dither(image, threshold)
    elif method == "ordered":
        return _ordered_dither(image)
    elif method == "floyd-steinberg":
        return _floyd_steinberg_dither(image)

    # unreachable, but satisfies type checker
    return image.copy()


def _threshold_dither(image: np.ndarray, threshold: float) -> np.ndarray:
    """Binary threshold dithering.

    Args:
        image: 2D float array [0, 1].
        threshold: Cutoff value.

    Returns:
        Binary image with values 0.0 or 1.0.
    """
    return np.where(image >= threshold, 1.0, 0.0).astype(np.float64)


def _ordered_dither(image: np.ndarray) -> np.ndarray:
    """Ordered (Bayer matrix) dithering.

    Tiles the 4x4 Bayer matrix across the image and compares each pixel
    to the corresponding threshold.

    Args:
        image: 2D float array [0, 1].

    Returns:
        Dithered image with values 0.0 or 1.0.
    """
    h, w = image.shape
    # Tile the Bayer matrix to cover the full image
    tile_h = (h + 3) // 4
    tile_w = (w + 3) // 4
    bayer_tiled = np.tile(_BAYER_4X4, (tile_h, tile_w))[:h, :w]
    return np.where(image >= bayer_tiled, 1.0, 0.0).astype(np.float64)


def _floyd_steinberg_dither(image: np.ndarray) -> np.ndarray:
    """Floyd-Steinberg error diffusion dithering.

    Distributes quantization error to neighboring pixels:
        * 7/16 to right
        * 3/16 to bottom-left
        * 5/16 to bottom
        * 1/16 to bottom-right

    Implementation uses a padded buffer so that all boundary checks are
    eliminated from the inner loop — every pixel writes to row[x+1] and
    nxt[x-1..x+1] unconditionally. The scanline (y) loop stays in Python
    because the 7/16 rightward carry is a sequential dependency that
    cannot be expressed as a numpy reduction.

    Args:
        image: 2D float array [0, 1].

    Returns:
        Dithered image with values 0.0 or 1.0.
    """
    h, w = image.shape

    # Pad: 1 extra row on the bottom + 1 extra column on each side.
    # Image pixel (y, x) lives at buf[y, x+1].  The padding absorbs all
    # out-of-bounds writes so the inner loop needs zero boundary checks.
    buf = np.zeros((h + 1, w + 2), dtype=np.float64)
    buf[:h, 1:w + 1] = image

    result = np.empty((h, w), dtype=np.float64)

    for y in range(h):
        row = buf[y]      # 1D view — length w+2
        nxt = buf[y + 1]  # 1D view — length w+2

        for x in range(1, w + 1):  # columns 1..w map to image columns 0..w-1
            old = row[x]
            new = 1.0 if old >= 0.5 else 0.0
            result[y, x - 1] = new
            err = old - new
            if err:
                row[x + 1] += err * 0.4375   # right:        7/16
                nxt[x - 1] += err * 0.1875   # bottom-left:  3/16
                nxt[x]     += err * 0.3125   # bottom:       5/16
                nxt[x + 1] += err * 0.0625   # bottom-right: 1/16

    return result
