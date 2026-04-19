"""Audio I/O: read and write audio files using stdlib ``wave`` and ffmpeg.

Supported natively (no external tools required):
    - WAV (16-bit and 24-bit PCM)

All other formats (MP3, FLAC, OGG, AAC, …) require ``ffmpeg`` to be
installed and available on the system PATH.  ffmpeg is used to transcode to
or from a temporary WAV file which is then handled by the stdlib ``wave``
module.

Internal audio representation
------------------------------
All samples are returned / accepted as numpy float64 arrays normalized to the
range [-1.0, 1.0].

    - Mono  : shape (num_samples,)
    - Stereo: shape (num_samples, 2)
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal constants
# ---------------------------------------------------------------------------

_FFMPEG_INSTALL_MSG = (
    "ffmpeg not found. Install it: "
    "Windows: `winget install ffmpeg` | "
    "macOS: `brew install ffmpeg` | "
    "Linux: `apt install ffmpeg`"
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def read_audio(path: str) -> Tuple[np.ndarray, int]:
    """Read an audio file and return float64 samples and the sample rate.

    WAV files (16-bit and 24-bit PCM) are decoded with the stdlib ``wave``
    module.  All other formats are decoded via ffmpeg to a temporary WAV file
    first.

    Args:
        path: Filesystem path to the audio file.

    Returns:
        A tuple ``(samples, sample_rate)`` where ``samples`` is a float64
        numpy array in [-1.0, 1.0].  Mono files have shape
        ``(num_samples,)``; stereo files have shape ``(num_samples, 2)``.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        RuntimeError: If a non-WAV format is requested and ffmpeg is not
            available.
        ValueError: If the WAV bit depth is unsupported (not 16 or 24 bit).
    """
    src = Path(path)
    if not src.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    if src.suffix.lower() == ".wav":
        logger.debug("read_audio: reading WAV directly: %s", path)
        return _read_wav(src)

    # Non-WAV: transcode to temp WAV via ffmpeg, then read.
    logger.info("read_audio: non-WAV format detected (%s), using ffmpeg", src.suffix)
    _require_ffmpeg()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        _ffmpeg_decode(src, tmp_path)
        samples, sample_rate = _read_wav(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    return samples, sample_rate


def write_audio(path: str, samples: np.ndarray, sample_rate: int = 44100) -> None:
    """Write audio samples to a file.

    WAV output uses the stdlib ``wave`` module (16-bit PCM).  Other formats
    are written by first encoding to a temporary WAV, then transcoding via
    ffmpeg.

    Note on sample rate conversion: this function writes the given samples
    at the given sample rate without resampling.  If the target container or
    codec requires a specific sample rate, ffmpeg will handle the conversion
    automatically during transcoding.

    Args:
        path: Destination filesystem path.  The extension determines the
            output format (e.g. ``.mp3``, ``.flac``).
        samples: Float64 numpy array in [-1.0, 1.0].  Shape ``(N,)`` for
            mono or ``(N, 2)`` for stereo.
        sample_rate: Sample rate in Hz.

    Raises:
        ValueError: If ``samples`` has an unsupported shape or dtype.
        RuntimeError: If a non-WAV format is requested and ffmpeg is not
            available.
    """
    dst = Path(path)

    if dst.suffix.lower() == ".wav":
        logger.debug("write_audio: writing WAV directly: %s", path)
        _write_wav(dst, samples, sample_rate)
        return

    logger.info(
        "write_audio: non-WAV format (%s), using ffmpeg for transcoding",
        dst.suffix,
    )
    _require_ffmpeg()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        _write_wav(tmp_path, samples, sample_rate)
        _ffmpeg_encode(tmp_path, dst)
    finally:
        tmp_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# WAV helpers
# ---------------------------------------------------------------------------


def _read_wav(path: Path) -> Tuple[np.ndarray, int]:
    """Read a WAV file and return (float64 samples, sample_rate).

    Supports 16-bit and 24-bit PCM.  Both mono and stereo channels are
    handled.

    Args:
        path: Path to the WAV file.

    Returns:
        ``(samples, sample_rate)`` — see :func:`read_audio` for conventions.

    Raises:
        ValueError: If the bit depth is not 16 or 24.
    """
    with wave.open(str(path), "rb") as wf:
        num_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()  # bytes per sample
        sample_rate = wf.getframerate()
        num_frames = wf.getnframes()
        raw_bytes = wf.readframes(num_frames)

    bit_depth = sample_width * 8
    logger.debug(
        "_read_wav: %d ch, %d Hz, %d-bit, %d frames",
        num_channels,
        sample_rate,
        bit_depth,
        num_frames,
    )

    if sample_width == 2:
        # 16-bit little-endian signed integers.
        int_samples = np.frombuffer(raw_bytes, dtype="<i2").astype(np.float64)
        float_samples = int_samples / 32768.0

    elif sample_width == 3:
        # 24-bit: no native numpy dtype, so unpack manually.
        num_total_samples = num_frames * num_channels
        raw = bytearray(raw_bytes)
        int_samples = np.empty(num_total_samples, dtype=np.int32)
        for i in range(num_total_samples):
            b0 = raw[i * 3]
            b1 = raw[i * 3 + 1]
            b2 = raw[i * 3 + 2]
            # Sign-extend: the most-significant byte carries the sign bit.
            value = b0 | (b1 << 8) | (b2 << 16)
            if value & 0x800000:
                value -= 0x1000000
            int_samples[i] = value
        float_samples = int_samples.astype(np.float64) / 8388608.0

    else:
        raise ValueError(
            f"Unsupported WAV bit depth: {bit_depth}-bit. "
            "Only 16-bit and 24-bit PCM are supported."
        )

    # Clip to [-1, 1] to guard against rare encoder errors at full-scale.
    float_samples = np.clip(float_samples, -1.0, 1.0)

    if num_channels == 2:
        float_samples = float_samples.reshape(-1, 2)
    elif num_channels > 2:
        # Reshape to (frames, channels); caller can decide what to do.
        float_samples = float_samples.reshape(-1, num_channels)

    return float_samples, sample_rate


def _write_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    """Write float64 samples as a 16-bit PCM WAV file.

    Args:
        path: Destination path.
        samples: Float64 array, shape ``(N,)`` or ``(N, channels)``.
        sample_rate: Sample rate in Hz.

    Raises:
        ValueError: If ``samples`` has more than 2 dimensions.
    """
    if samples.ndim == 1:
        num_channels = 1
    elif samples.ndim == 2:
        num_channels = samples.shape[1]
    else:
        raise ValueError(
            f"samples must be 1D or 2D, got shape {samples.shape}"
        )

    # Convert float64 -> int16 (multiply by 32767, not 32768, to avoid
    # wrapping at exactly +1.0).
    int16_samples = (
        np.clip(samples, -1.0, 1.0) * 32767
    ).astype(np.int16)

    # Flatten interleaved for multi-channel.
    raw_bytes = int16_samples.flatten().tobytes()

    logger.debug(
        "_write_wav: writing %d ch, %d Hz, 16-bit -> %s",
        num_channels,
        sample_rate,
        path,
    )

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(num_channels)
        wf.setsampwidth(2)  # 16-bit = 2 bytes
        wf.setframerate(sample_rate)
        wf.writeframes(raw_bytes)


# ---------------------------------------------------------------------------
# ffmpeg helpers
# ---------------------------------------------------------------------------


def _require_ffmpeg() -> None:
    """Raise RuntimeError if ffmpeg is not available on the PATH.

    Raises:
        RuntimeError: With a user-friendly installation message.
    """
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(_FFMPEG_INSTALL_MSG)


def _ffmpeg_decode(src: Path, dst_wav: Path) -> None:
    """Decode an arbitrary audio file to a WAV file using ffmpeg.

    The output WAV uses the source file's native sample rate and channel
    layout; no resampling or channel conversion is applied.

    Args:
        src: Source audio file path.
        dst_wav: Destination WAV file path (will be overwritten if it exists).

    Raises:
        subprocess.CalledProcessError: If ffmpeg exits with a non-zero code.
    """
    cmd = [
        "ffmpeg",
        "-y",           # overwrite output without asking
        "-i", str(src),
        "-f", "wav",
        "-acodec", "pcm_s16le",
        str(dst_wav),
    ]
    logger.debug("_ffmpeg_decode: %s", " ".join(cmd))
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        stderr_text = result.stderr.decode(errors="replace")
        raise RuntimeError(
            f"ffmpeg failed to decode '{src}' (exit {result.returncode}):\n"
            f"{stderr_text}"
        )


def _ffmpeg_encode(src_wav: Path, dst: Path) -> None:
    """Transcode a WAV file to the target format using ffmpeg.

    The output format is inferred from ``dst``'s file extension.  ffmpeg
    applies its default codec for that container.  If the target codec or
    container requires a different sample rate, ffmpeg performs the conversion
    automatically.

    Args:
        src_wav: Source WAV file path.
        dst: Destination file path; extension determines the output format.

    Raises:
        subprocess.CalledProcessError: If ffmpeg exits with a non-zero code.
    """
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(src_wav),
        str(dst),
    ]
    logger.debug("_ffmpeg_encode: %s", " ".join(cmd))
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        stderr_text = result.stderr.decode(errors="replace")
        raise RuntimeError(
            f"ffmpeg failed to encode '{dst}' (exit {result.returncode}):\n"
            f"{stderr_text}"
        )
