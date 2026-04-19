"""Audio mixing utilities for Murmur.

Provides functions to overlay an encoded steganographic signal into a carrier
audio track, with optional channel targeting and blend control.
"""

from __future__ import annotations

import logging

import numpy as np

from murmur.utils import normalize

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def overlay(
    carrier: np.ndarray,
    encoded: np.ndarray,
    blend: float = 0.3,
    offset_seconds: float = 0.0,
    sample_rate: int = 44100,
    channel: str = "both",
    headroom_db: float = -0.5,
) -> np.ndarray:
    """Mix an encoded signal into carrier audio.

    The encoded signal is added into the carrier starting at ``offset_seconds``.
    If the encoded signal extends beyond the end of the carrier it is truncated.
    The result is normalized to prevent clipping.

    Args:
        carrier: 1D array of shape ``(N,)`` for mono, or 2D array of shape
            ``(N, 2)`` for stereo. dtype should be float64 in ``[-1.0, 1.0]``.
        encoded: 1D mono signal of shape ``(M,)``.
        blend: Amplitude scaling factor applied to the encoded signal before
            mixing. Must be in ``[0.0, 1.0]``.
        offset_seconds: Time offset in seconds at which to begin inserting the
            encoded signal into the carrier. Must be non-negative.
        sample_rate: Sample rate in Hz used to convert ``offset_seconds`` to a
            sample index. Defaults to 44100.
        channel: Which channel(s) to inject the encoded signal into when the
            carrier is stereo. One of ``"both"``, ``"left"`` (column 0), or
            ``"right"`` (column 1). Ignored for mono carriers.
        headroom_db: Peak headroom in dB applied during final normalization.
            Defaults to -0.5 dB.

    Returns:
        Mixed audio array with the same shape and dtype as ``carrier``.

    Raises:
        ValueError: If ``carrier`` has an unsupported number of dimensions, if
            ``channel`` is not one of the accepted values, or if
            ``offset_seconds`` is negative.
    """
    if offset_seconds < 0:
        raise ValueError(f"offset_seconds must be non-negative, got {offset_seconds}")
    if channel not in ("both", "left", "right"):
        raise ValueError(f"channel must be 'both', 'left', or 'right', got {channel!r}")
    if carrier.ndim not in (1, 2):
        raise ValueError(f"carrier must be 1D or 2D, got shape {carrier.shape}")
    if carrier.ndim == 2 and carrier.shape[1] != 2:
        raise ValueError(
            f"Stereo carrier must have shape (N, 2), got {carrier.shape}"
        )

    # Work on a copy to avoid mutating the caller's array
    result = carrier.astype(np.float64, copy=True)
    encoded = np.asarray(encoded, dtype=np.float64)

    offset_samples = int(offset_seconds * sample_rate)
    carrier_len = result.shape[0]

    if offset_samples >= carrier_len:
        logger.warning(
            "offset_samples (%d) >= carrier length (%d); encoded signal not mixed.",
            offset_samples,
            carrier_len,
        )
        return result

    # Determine how many encoded samples can fit
    available = carrier_len - offset_samples
    enc_to_use = encoded[:available]  # truncate if longer than remaining space
    n_enc = len(enc_to_use)

    scaled = enc_to_use * blend
    sl = slice(offset_samples, offset_samples + n_enc)

    if result.ndim == 1:
        # Mono carrier
        result[sl] += scaled
        logger.debug(
            "overlay (mono): blended %d samples at offset %d with blend=%.3f",
            n_enc,
            offset_samples,
            blend,
        )
    else:
        # Stereo carrier
        if channel in ("both", "left"):
            result[sl, 0] += scaled
        if channel in ("both", "right"):
            result[sl, 1] += scaled
        logger.debug(
            "overlay (stereo, channel=%r): blended %d samples at offset %d with blend=%.3f",
            channel,
            n_enc,
            offset_samples,
            blend,
        )

    result = normalize(result, headroom_db=headroom_db)
    return result


def watermark_overlay(
    carrier: np.ndarray,
    encoded: np.ndarray,
    blend: float = 0.03,
    sample_rate: int = 44100,
    headroom_db: float = -0.5,
) -> np.ndarray:
    """Spread an encoded signal across the full carrier duration and mix it in.

    The encoded signal is time-stretched (via linear interpolation of sample
    indices) to exactly match the length of the carrier, then mixed in at a
    very low blend level so the watermark is imperceptible but retrievable via
    spectrogram analysis.

    Args:
        carrier: 1D array of shape ``(N,)`` for mono, or 2D array of shape
            ``(N, 2)`` for stereo. dtype should be float64 in ``[-1.0, 1.0]``.
        encoded: 1D mono signal of shape ``(M,)``.
        blend: Amplitude scaling factor for the stretched encoded signal.
            Defaults to ``0.03`` (very quiet, ~-30 dB relative to full scale).
        sample_rate: Sample rate in Hz passed through to :func:`overlay`.
        headroom_db: Peak headroom in dB applied during final normalization.
            Defaults to -0.5 dB.

    Returns:
        Mixed audio array with the same shape as ``carrier``.

    Raises:
        ValueError: If ``encoded`` is empty or if ``carrier`` has an
            unsupported shape.
    """
    encoded = np.asarray(encoded, dtype=np.float64)
    if len(encoded) == 0:
        raise ValueError("encoded signal is empty; nothing to watermark.")

    carrier_len = carrier.shape[0]

    if len(encoded) == carrier_len:
        stretched = encoded
        logger.debug("watermark_overlay: encoded already matches carrier length, no stretching needed.")
    else:
        # Linear interpolation: map carrier_len output samples onto encoded indices
        orig_indices = np.arange(len(encoded), dtype=np.float64)
        new_indices = np.linspace(0.0, len(encoded) - 1, carrier_len)
        stretched = np.interp(new_indices, orig_indices, encoded)
        logger.debug(
            "watermark_overlay: time-stretched encoded from %d → %d samples (factor=%.3fx)",
            len(encoded),
            carrier_len,
            carrier_len / len(encoded),
        )

    return overlay(
        carrier=carrier,
        encoded=stretched,
        blend=blend,
        offset_seconds=0.0,
        sample_rate=sample_rate,
        channel="both",
        headroom_db=headroom_db,
    )
