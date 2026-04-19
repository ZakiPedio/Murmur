"""Shared utilities: normalization, windowing, and helper functions."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def normalize(
    samples: np.ndarray,
    target_peak: float = 0.95,
    headroom_db: Optional[float] = None,
) -> np.ndarray:
    """Normalize audio samples to a target peak amplitude.

    Args:
        samples: 1D or 2D float64 audio array.
        target_peak: Target peak amplitude in (0.0, 1.0]. Used when headroom_db is None.
                     Defaults to 0.95.
        headroom_db: If provided, overrides target_peak. Peak headroom in dB
                     (e.g. -0.5 means peak = 10^(-0.5/20) ≈ 0.944).

    Returns:
        Normalized array with same shape as input.

    Raises:
        ValueError: If the resolved target_peak is not in (0, 1].
    """
    if headroom_db is not None:
        target_peak = 10 ** (headroom_db / 20.0)
    if not (0 < target_peak <= 1.0):
        raise ValueError(f"target_peak must be in (0, 1], got {target_peak}")

    peak = np.max(np.abs(samples))
    if peak == 0.0:
        logger.debug("normalize: input is silent (all zeros), returning unchanged")
        return samples.copy()

    return samples * (target_peak / peak)


def hann_window(n: int) -> np.ndarray:
    """Return a Hann window of length n.

    Args:
        n: Window length in samples.

    Returns:
        1D float64 array of length n.
    """
    return 0.5 * (1 - np.cos(2 * np.pi * np.arange(n) / n))


def db_to_amplitude(db: float) -> float:
    """Convert decibels to linear amplitude.

    Args:
        db: Value in decibels.

    Returns:
        Linear amplitude.
    """
    return 10 ** (db / 20.0)


def amplitude_to_db(amplitude: float, epsilon: float = 1e-10) -> float:
    """Convert linear amplitude to decibels.

    Args:
        amplitude: Linear amplitude value.
        epsilon: Small offset to avoid log(0).

    Returns:
        Value in decibels.
    """
    return 20.0 * np.log10(amplitude + epsilon)


def parse_time(time_str: str) -> float:
    """Parse a time string into seconds.

    Supports formats:
        - Plain float: "154.0"
        - MM:SS: "2:34"
        - HH:MM:SS: "1:02:34"

    Args:
        time_str: Time string to parse.

    Returns:
        Time in seconds as float.

    Raises:
        ValueError: If the format is unrecognized.
    """
    time_str = time_str.strip()
    if ":" in time_str:
        parts = time_str.split(":")
        try:
            parts_f = [float(p) for p in parts]
        except ValueError as e:
            raise ValueError(f"Invalid time format: {time_str!r}") from e
        if len(parts) == 2:
            return parts_f[0] * 60 + parts_f[1]
        elif len(parts) == 3:
            return parts_f[0] * 3600 + parts_f[1] * 60 + parts_f[2]
        else:
            raise ValueError(f"Invalid time format: {time_str!r}")
    else:
        try:
            return float(time_str)
        except ValueError as e:
            raise ValueError(f"Invalid time format: {time_str!r}") from e


def parse_resolution(res_str: str) -> tuple[int, int]:
    """Parse a resolution string like '256x128' into (width, height).

    Args:
        res_str: Resolution string in WxH format.

    Returns:
        (width, height) tuple.

    Raises:
        ValueError: If the format is invalid.
    """
    try:
        w, h = res_str.lower().split("x")
        return int(w), int(h)
    except (ValueError, AttributeError) as e:
        raise ValueError(f"Invalid resolution format: {res_str!r}, expected WxH (e.g. '256x128')") from e


def clamp(value: float, lo: float, hi: float) -> float:
    """Clamp a value to the range [lo, hi].

    Args:
        value: Value to clamp.
        lo: Lower bound.
        hi: Upper bound.

    Returns:
        Clamped value.
    """
    return max(lo, min(hi, value))
