"""Spectrogram generation from audio samples.

Produces a PNG visualization of audio frequency content over time using a
manual STFT (short-time Fourier transform) — no scipy or matplotlib.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Colormap helpers
# ---------------------------------------------------------------------------

_COLORMAP_CONTROL_POINTS: dict[str, list[tuple[int, int, int]]] = {
    "inferno": [
        (0, 0, 0),
        (40, 0, 80),
        (120, 0, 100),
        (180, 30, 0),
        (220, 120, 0),
        (255, 220, 100),
        (255, 255, 255),
    ],
    "grayscale": [
        (0, 0, 0),
        (255, 255, 255),
    ],
    "viridis": [
        (68, 1, 84),
        (59, 82, 139),
        (33, 145, 140),
        (94, 201, 98),
        (253, 231, 37),
    ],
}


def _build_colormap(name: str) -> np.ndarray:
    """Build a 256-entry colormap lookup table by linearly interpolating control points.

    Args:
        name: Colormap name — one of ``"inferno"``, ``"grayscale"``, or ``"viridis"``.

    Returns:
        Array of shape ``(256, 3)`` with dtype ``uint8``.

    Raises:
        ValueError: If ``name`` is not a recognised colormap.
    """
    if name not in _COLORMAP_CONTROL_POINTS:
        raise ValueError(
            f"Unknown colormap {name!r}. Choose from: {list(_COLORMAP_CONTROL_POINTS)}"
        )

    control = _COLORMAP_CONTROL_POINTS[name]
    n_points = len(control)
    lut = np.zeros((256, 3), dtype=np.float64)

    # Spread control points evenly across [0, 255]
    indices = np.linspace(0, 255, n_points)

    for i in range(n_points - 1):
        lo_idx = int(round(indices[i]))
        hi_idx = int(round(indices[i + 1]))
        seg_len = hi_idx - lo_idx
        if seg_len == 0:
            lut[lo_idx] = control[i]
            continue
        for channel in range(3):
            lut[lo_idx : hi_idx + 1, channel] = np.linspace(
                control[i][channel], control[i + 1][channel], seg_len + 1
            )

    return np.clip(lut, 0, 255).astype(np.uint8)


def _apply_colormap(data: np.ndarray, colormap: np.ndarray) -> np.ndarray:
    """Map a 2D array of uint8 indices through a colormap LUT.

    Args:
        data: 2D array of shape ``(H, W)`` with dtype ``uint8``.
            Each value is an index into the colormap.
        colormap: Array of shape ``(256, 3)`` with dtype ``uint8``.

    Returns:
        Array of shape ``(H, W, 3)`` with dtype ``uint8``.
    """
    return colormap[data]


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------


def generate_spectrogram(
    samples: np.ndarray,
    sample_rate: int = 44100,
    output_path: str = "spectrogram.png",
    freq_min: Optional[float] = None,
    freq_max: Optional[float] = None,
    fft_size: int = 4096,
    hop_size: int = 512,
    colormap: str = "inferno",
    width: Optional[int] = None,
    height: Optional[int] = None,
    log_freq_warp: bool = False,
) -> None:
    """Generate a spectrogram PNG from an audio signal.

    Performs a manual STFT (no scipy), converts magnitudes to dB, applies a
    colormap, and saves the result as a PNG via Pillow.

    The image axes are: time on X (left = start, right = end) and frequency
    on Y (bottom = low frequencies, top = high frequencies).

    Args:
        samples: 1D float64 audio array, mono only. Stereo should be mixed
            down to mono before calling.
        sample_rate: Sample rate of ``samples`` in Hz. Defaults to 44100.
        output_path: Destination path for the output PNG.
        freq_min: Lower frequency bound in Hz for the displayed range. If
            ``None``, defaults to 0 Hz (DC bin).
        freq_max: Upper frequency bound in Hz for the displayed range. If
            ``None``, defaults to ``sample_rate / 2`` (Nyquist).
        fft_size: FFT window size in samples. Larger values give better
            frequency resolution at the cost of time resolution.
        hop_size: Number of samples to advance between successive frames.
            Smaller values produce smoother time resolution.
        colormap: Name of the colormap to apply. One of ``"inferno"``,
            ``"grayscale"``, or ``"viridis"``.
        width: If provided, resize the output image to this width (pixels)
            using Lanczos resampling.
        height: If provided, resize the output image to this height (pixels)
            using Lanczos resampling.
        log_freq_warp: If True and ``freq_min`` is a positive value, remap
            the linear STFT frequency axis to a logarithmic scale before
            saving.  This matches the encoder's log frequency mapping so that
            each encoded canvas row occupies equal pixel height regardless of
            frequency.  Low-frequency content (letter base strokes, QR bottom
            modules) gets more display pixels instead of being crushed into
            1-2 px by the log→linear mismatch.  Has no effect when
            ``freq_min`` is ``None`` or 0.

    Returns:
        None. Saves the PNG to ``output_path``.

    Raises:
        ValueError: If ``samples`` is not 1D or if ``colormap`` is unknown.
    """
    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim != 1:
        raise ValueError(
            f"samples must be 1D (mono). Got shape {samples.shape}. "
            "Mix stereo to mono before calling generate_spectrogram()."
        )

    n_samples = len(samples)
    logger.info(
        "Generating spectrogram: %d samples @ %d Hz, fft_size=%d, hop_size=%d, colormap=%r",
        n_samples,
        sample_rate,
        fft_size,
        hop_size,
        colormap,
    )

    # -----------------------------------------------------------------------
    # 1. Build Hann window
    # -----------------------------------------------------------------------
    n = fft_size
    window = 0.5 * (1 - np.cos(2 * np.pi * np.arange(n) / n))

    # -----------------------------------------------------------------------
    # 2. Compute STFT frames
    # -----------------------------------------------------------------------
    # Pad the signal so the last frame is complete
    n_frames = max(1, (n_samples - fft_size) // hop_size + 1)
    pad_length = (n_frames - 1) * hop_size + fft_size - n_samples
    if pad_length > 0:
        samples = np.concatenate([samples, np.zeros(pad_length, dtype=np.float64)])

    # Build frame matrix: shape (n_frames, fft_size)
    frame_starts = np.arange(n_frames) * hop_size
    frames = np.stack([samples[s : s + fft_size] for s in frame_starts])  # (T, fft_size)

    # Apply Hann window to every frame simultaneously
    frames *= window  # broadcast across rows

    # -----------------------------------------------------------------------
    # 3. FFT + magnitude
    # -----------------------------------------------------------------------
    fft_result = np.fft.rfft(frames, n=fft_size, axis=1)  # (T, fft_size//2+1)
    magnitude = np.abs(fft_result)  # (T, F)

    # -----------------------------------------------------------------------
    # 4. Convert to dB
    # -----------------------------------------------------------------------
    db = 20.0 * np.log10(magnitude + 1e-10)  # (T, F)

    # -----------------------------------------------------------------------
    # 5. Frequency-axis crop
    # -----------------------------------------------------------------------
    nyquist = sample_rate / 2.0
    n_bins = fft_size // 2 + 1  # number of rfft frequency bins

    # Bin index for a given Hz value
    def _hz_to_bin(hz: float) -> int:
        return int(round(hz / nyquist * (n_bins - 1)))

    f_lo = 0 if freq_min is None else max(0, _hz_to_bin(freq_min))
    f_hi = n_bins - 1 if freq_max is None else min(n_bins - 1, _hz_to_bin(freq_max))

    if f_lo >= f_hi:
        logger.warning(
            "freq_min/freq_max (%s/%s) produce an empty or degenerate bin range [%d, %d]; "
            "using full range.",
            freq_min,
            freq_max,
            f_lo,
            f_hi,
        )
        f_lo, f_hi = 0, n_bins - 1

    db = db[:, f_lo : f_hi + 1]  # (T, F_cropped)
    logger.debug("Frequency bins kept: %d..%d (of %d)", f_lo, f_hi, n_bins)

    # -----------------------------------------------------------------------
    # 6. Normalize to [0, 255]
    # -----------------------------------------------------------------------
    db_min = db.min()
    db_max = db.max()
    if db_max == db_min:
        # Silence or constant signal — produce a uniform image
        normalized = np.zeros_like(db, dtype=np.uint8)
    else:
        normalized = ((db - db_min) / (db_max - db_min) * 255).astype(np.uint8)

    # normalized shape: (T, F_cropped) — T = time axis, F = frequency axis

    # -----------------------------------------------------------------------
    # 7. Orient image: X = time, Y = frequency (low freq at bottom)
    #    Transpose so rows = freq bins, cols = time frames → (F, T)
    #    Then flip vertically so row 0 = highest frequency (top of image)
    #    but we want low freq at bottom visually, so flip so row 0 = highest.
    # -----------------------------------------------------------------------
    img_data = normalized.T  # (F_cropped, T) — rows are freq, cols are time
    img_data = np.flipud(img_data)  # flip so low freq is at bottom of PNG

    # -----------------------------------------------------------------------
    # 7.5 Optional log-frequency warp
    #     After flipud, img_data row 0 = freq_max (top) and row -1 = freq_min
    #     (bottom).  The STFT bins are linearly spaced in Hz, but the encoder
    #     uses a log frequency scale — so each encoded canvas row spans an
    #     exponentially growing number of STFT bins as frequency increases.
    #     At 200 Hz with fft_size=4096 / sr=44100 this means the bottom ~8
    #     canvas rows each occupy only 1-2 STFT bins, which after downscaling
    #     to the target height collapse to <1 px and become invisible.
    #
    #     The log warp stretches the image rows so that equal canvas rows take
    #     equal pixel height — low-frequency content (letter bases, QR modules)
    #     gets the same thickness as high-frequency content.
    # -----------------------------------------------------------------------
    if log_freq_warp and freq_min is not None and freq_min > 0:
        _f_lo = float(freq_min)
        _f_hi = float(freq_max) if freq_max is not None else float(nyquist)
        F_rows = img_data.shape[0]

        # For output row i (0 = top = f_hi, F-1 = bottom = f_lo):
        #   log-spaced frequency: f = f_hi * (f_lo/f_hi)^(i/(F-1))
        #   linear source row:    src = (f_hi - f) / (f_hi - f_lo) * (F-1)
        t = np.arange(F_rows, dtype=np.float64) / max(F_rows - 1, 1)
        freq_at_row = _f_hi * (_f_lo / _f_hi) ** t      # log interpolation
        src_rows = (_f_hi - freq_at_row) / (_f_hi - _f_lo) * (F_rows - 1)
        src_rows = np.clip(src_rows, 0.0, F_rows - 1.0)

        i0 = np.floor(src_rows).astype(np.intp)
        i1 = np.minimum(i0 + 1, F_rows - 1)
        alpha = (src_rows - i0)[:, np.newaxis]           # (F_rows, 1)

        img_data = np.round(
            (1.0 - alpha) * img_data[i0].astype(np.float64)
            + alpha * img_data[i1].astype(np.float64)
        ).astype(np.uint8)
        logger.debug(
            "Log-frequency warp applied: %.1f-%.1f Hz, %d rows",
            _f_lo, _f_hi, F_rows,
        )

    # -----------------------------------------------------------------------
    # 8. Apply colormap → (H, W, 3) uint8
    # -----------------------------------------------------------------------
    lut = _build_colormap(colormap)
    rgb = _apply_colormap(img_data, lut)  # (H, W, 3)

    # -----------------------------------------------------------------------
    # 9. Build PIL image, optionally resize, save
    # -----------------------------------------------------------------------
    pil_img = Image.fromarray(rgb, mode="RGB")

    if width is not None or height is not None:
        orig_w, orig_h = pil_img.size
        new_w = width if width is not None else orig_w
        new_h = height if height is not None else orig_h
        logger.debug("Resizing spectrogram from %dx%d to %dx%d", orig_w, orig_h, new_w, new_h)
        pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pil_img.save(str(out), format="PNG")
    logger.info("Spectrogram saved to %s (%dx%d px)", out, pil_img.width, pil_img.height)
