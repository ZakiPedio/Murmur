"""Encoder: convert a grayscale image into audio via additive sine synthesis.

The spectrogram of the resulting audio reproduces the input image.
"""

from __future__ import annotations

import logging

import numpy as np

from murmur.utils import normalize, hann_window

logger = logging.getLogger(__name__)


def encode(
    image: np.ndarray,
    sample_rate: int = 44100,
    duration: float = 10.0,
    freq_min: float = 200.0,
    freq_max: float = 8000.0,
    log_frequency: bool = True,
    headroom_db: float = -0.5,
    randomize_phase: bool = False,
    seed: int | None = None,
) -> np.ndarray:
    """Convert a grayscale image into audio whose spectrogram reproduces the image.

    Each row of the image maps to a frequency band; each column maps to a time
    slice. Pixel brightness controls the amplitude of the corresponding sine
    wave. The synthesis uses overlap-add with a Hann window to avoid clicks at
    frame boundaries.

    Args:
        image: 2D float64 array with values in [0, 1]. Shape is
            (freq_bins, time_bins). Row 0 corresponds to freq_max,
            the last row to freq_min.
        sample_rate: Output sample rate in Hz.
        duration: Total audio duration in seconds.
        freq_min: Lowest frequency mapped to the last image row, in Hz.
        freq_max: Highest frequency mapped to the first image row, in Hz.
        log_frequency: If True, use logarithmic frequency spacing (perceptually
            uniform). If False, use linear spacing.
        headroom_db: Peak headroom of the normalized output, in dB. For example
            -0.5 yields a peak of ~0.944 FS.
        randomize_phase: If True, initialize per-frequency phases to random
            values in [0, 2*pi) instead of zero. This makes the output sound
            like band-limited noise rather than a buzzy pure-tone chord, at the
            cost of reproducibility (output differs on each call).
        seed: Integer seed for the random number generator used by
            ``randomize_phase``. When ``None`` (default), a fresh random seed
            is drawn each call. When an integer is given, the output is fully
            reproducible across runs.

    Returns:
        1D float64 numpy array of audio samples with length
        ``int(sample_rate * duration)``, normalized to [-1.0, 1.0].

    Raises:
        ValueError: If ``image`` is not 2-dimensional or contains values
            outside [0, 1].
    """
    if image.ndim != 2:
        raise ValueError(
            f"image must be a 2D array, got shape {image.shape}"
        )
    freq_bins, time_bins = image.shape
    if freq_bins == 0 or time_bins == 0:
        raise ValueError(
            f"image must have at least 1 row and 1 column, got shape {image.shape}"
        )
    if image.min() < 0.0 or image.max() > 1.0:
        raise ValueError(
            "image values must be in [0, 1]; "
            f"got min={image.min():.4f}, max={image.max():.4f}"
        )
    total_samples = int(sample_rate * duration)

    logger.info(
        "encode: %d freq bins, %d time bins, %d Hz, %.2f s -> %d samples",
        freq_bins,
        time_bins,
        sample_rate,
        duration,
        total_samples,
    )

    # --- Frequency mapping -------------------------------------------------
    # Row 0 = freq_max, last row = freq_min.
    if log_frequency:
        freqs = np.logspace(
            np.log10(freq_min), np.log10(freq_max), freq_bins
        )[::-1]
    else:
        freqs = np.linspace(freq_max, freq_min, freq_bins)

    logger.debug(
        "encode: frequency range %.1f Hz – %.1f Hz (%s spacing)",
        freqs[-1],
        freqs[0],
        "log" if log_frequency else "linear",
    )

    # --- Overlap-add setup -------------------------------------------------
    hop_size = total_samples // time_bins
    window_size = hop_size * 2
    window = hann_window(window_size)  # shape (window_size,)

    # Output buffer — slightly longer to accommodate the last frame overflow.
    output = np.zeros(total_samples + window_size, dtype=np.float64)

    # Phase accumulators: one per frequency.  Stored as a running phase value
    # (in radians) so that each frame starts exactly where the previous one
    # left off, preventing phase discontinuities.
    if randomize_phase:
        logger.debug("encode: randomizing initial phases (seed=%s)", seed)
        rng = np.random.default_rng(seed)
        phases = rng.uniform(0.0, 2.0 * np.pi, freq_bins)
    else:
        phases = np.zeros(freq_bins, dtype=np.float64)

    # Pre-compute local sample indices (0 … window_size-1) — reused every frame.
    local_indices = np.arange(window_size, dtype=np.float64)

    # Pre-compute per-frequency phase advance per hop (advanced every frame).
    phase_advance = 2.0 * np.pi * freqs * hop_size / sample_rate  # (freq_bins,)

    # Pre-compute per-frequency angular rate (2*pi*freq / sample_rate) used
    # inside the frame to build the instantaneous phase matrix.
    angular_rate = 2.0 * np.pi * freqs / sample_rate  # (freq_bins,)

    # --- Per-frame synthesis (vectorized) ----------------------------------
    # Each iteration synthesizes all freq_bins simultaneously via broadcasting:
    #   phi  : (freq_bins, window_size) — instantaneous phase matrix
    #   frame: (window_size,)           — sum over frequency axis
    # This replaces the inner `for f_idx` loop and is ~20-50x faster.
    _progress_step = max(1, time_bins // 10)  # report ~10 progress milestones
    for t_idx in range(time_bins):
        t_start = t_idx * hop_size

        if t_idx % _progress_step == 0:
            pct = int(100 * t_idx / time_bins)
            logger.info("encode: %3d%%  [frame %d/%d]", pct, t_idx, time_bins)

        amplitudes = image[:, t_idx]  # (freq_bins,)

        # phi[f, n] = phases[f] + angular_rate[f] * n
        # shape: (freq_bins, window_size)
        phi = phases[:, np.newaxis] + angular_rate[:, np.newaxis] * local_indices[np.newaxis, :]

        # Weighted sum over frequency bins → (window_size,).
        # Using matmul (amplitudes @ sin(phi)) is equivalent to
        # np.sum(amplitudes[:, None] * np.sin(phi), axis=0) but uses BLAS.
        frame = amplitudes @ np.sin(phi)

        # Advance all phases by hop_size samples (phase continuity).
        phases = (phases + phase_advance) % (2.0 * np.pi)

        # Apply Hann window and overlap-add into output buffer.
        output[t_start : t_start + window_size] += frame * window

    # Trim to exact length.
    output = output[:total_samples]

    # --- Normalize ----------------------------------------------------------
    logger.debug("encode: normalizing with headroom_db=%.1f", headroom_db)
    output = normalize(output, headroom_db=headroom_db)

    logger.info("encode: done — peak amplitude %.4f", np.max(np.abs(output)))
    return output
