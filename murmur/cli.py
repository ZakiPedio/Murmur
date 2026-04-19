"""Command-line interface for Murmur.

Provides six subcommands:
  encode   – render an image/text/QR and synthesise encoded audio
  overlay  – embed image signal into an existing carrier audio file
  verify   – generate a spectrogram PNG for visual inspection
  decode   – extract a QR code from a spectrogram image
  presets  – list available named presets
  probe    – analyze a carrier file and recommend encoding parameters
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

from murmur.utils import hann_window

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Argument-parser helpers
# ---------------------------------------------------------------------------

_DEFAULT_WIDTH = 256
_DEFAULT_HEIGHT = 128


def _add_input_group(parser: argparse.ArgumentParser) -> None:
    """Add a mutually exclusive input-type group to *parser*."""
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", metavar="PATH", help="Input image file")
    group.add_argument("--text", metavar="STRING", help="Render text as image")
    group.add_argument("--ascii", metavar="STRING", help="Render ASCII art as image")
    group.add_argument("--qr", metavar="STRING", help="Render QR code")
    group.add_argument("--svg", metavar="PATH", help="Render SVG file")
    group.add_argument("--math", metavar="EXPRESSION", help="Render math expression")
    group.add_argument("--gif", metavar="PATH", help="Use first frame of GIF")
    group.add_argument(
        "--images",
        metavar="PATH",
        nargs="+",
        help="One or more images to tile into a sequence",
    )


def _add_encoding_options(parser: argparse.ArgumentParser) -> None:
    """Add shared encoding parameters to *parser*."""
    parser.add_argument(
        "--qr-logo",
        metavar="PATH",
        default=None,
        help="Logo image to embed in the centre of a QR code (use with --qr)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        metavar="FLOAT",
        help="Audio duration in seconds (default: 10.0)",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=44100,
        metavar="INT",
        help="Audio sample rate in Hz (default: 44100)",
    )
    parser.add_argument(
        "--freq-min",
        type=float,
        default=200.0,
        metavar="FLOAT",
        help="Lowest frequency in Hz (default: 200.0)",
    )
    parser.add_argument(
        "--freq-max",
        type=float,
        default=8000.0,
        metavar="FLOAT",
        help="Highest frequency in Hz (default: 8000.0)",
    )
    parser.add_argument(
        "--freq-scale",
        choices=["log", "lin"],
        default="log",
        help="Frequency axis scaling (default: log)",
    )
    parser.add_argument(
        "--resolution",
        metavar="WxH",
        default=None,
        help="Image resolution, e.g. '256x128' (default: 256x128)",
    )
    parser.add_argument(
        "--dither",
        choices=["floyd-steinberg", "ordered", "threshold", "none"],
        default="none",
        help="Dithering algorithm (default: none)",
    )
    parser.add_argument(
        "--invert",
        action="store_true",
        help="Invert the image before encoding",
    )
    parser.add_argument(
        "--headroom-db",
        type=float,
        default=-0.5,
        metavar="FLOAT",
        help="Peak headroom in dB (default: -0.5)",
    )
    parser.add_argument(
        "--preset",
        metavar="NAME",
        default=None,
        help="Named preset to load as defaults",
    )
    parser.add_argument(
        "--font",
        metavar="PATH",
        default=None,
        help="Path to a TTF/OTF font file for text rendering",
    )
    parser.add_argument(
        "--font-size",
        type=int,
        default=48,
        metavar="INT",
        help="Font size for text rendering (default: 48)",
    )
    parser.add_argument(
        "--x-range",
        metavar="MIN:MAX",
        default="-10:10",
        help="X-axis range for math plots (default: -10:10)",
    )
    parser.add_argument(
        "--plot-thickness",
        type=int,
        default=3,
        metavar="INT",
        help="Line thickness for math plot rendering (default: 3)",
    )
    parser.add_argument(
        "--tile",
        type=int,
        default=1,
        metavar="INT",
        help="Repeat the image N times across the duration (default: 1)",
    )
    parser.add_argument(
        "--auto-contrast",
        action="store_true",
        help=(
            "Histogram-stretch the input image so its darkest pixel maps to 0 "
            "and brightest to 1. Useful for faint or washed-out source images."
        ),
    )
    parser.add_argument(
        "--randomize-phase",
        action="store_true",
        help=(
            "Randomize per-frequency initial phases. The output sounds like "
            "band-limited noise instead of a buzzy pure-tone chord. "
            "Off by default for reproducibility."
        ),
    )
    parser.add_argument(
        "--mp3-safe",
        action="store_true",
        help=(
            "Clamp --freq-min/--freq-max to the MP3-safe window (~200-16000 Hz) "
            "and warn if your requested range falls outside it."
        ),
    )
    parser.add_argument(
        "--verify-after-encode",
        action="store_true",
        help=(
            "After writing the WAV, automatically generate a spectrogram PNG "
            "alongside it (same stem, .spec.png). Lets you see the embedded "
            "content without a separate 'murmur verify' call."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        metavar="INT",
        help=(
            "Integer seed for --randomize-phase. Makes the randomized output "
            "reproducible across runs. Has no effect without --randomize-phase."
        ),
    )


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

_MP3_FREQ_MIN = 200.0
_MP3_FREQ_MAX = 16000.0


def _validate_encoding_args(args: argparse.Namespace) -> None:
    """Validate freq, blend, and duration values; sys.exit(1) on failure."""
    # --mp3-safe: clamp range and warn before other validation
    if getattr(args, "mp3_safe", False):
        warned = False
        if args.freq_min < _MP3_FREQ_MIN:
            print(
                f"Warning: --freq-min {args.freq_min:.0f} Hz is below the MP3-safe floor "
                f"({_MP3_FREQ_MIN:.0f} Hz) — clamping.",
                file=sys.stderr,
            )
            args.freq_min = _MP3_FREQ_MIN
            warned = True
        if args.freq_max > _MP3_FREQ_MAX:
            print(
                f"Warning: --freq-max {args.freq_max:.0f} Hz exceeds the MP3-safe ceiling "
                f"({_MP3_FREQ_MAX:.0f} Hz) — clamping.",
                file=sys.stderr,
            )
            args.freq_max = _MP3_FREQ_MAX
            warned = True
        if warned:
            print(
                "Note: the embedded image will survive MP3 re-encoding within this range.",
                file=sys.stderr,
            )

    if args.freq_min >= args.freq_max:
        _die(
            f"--freq-min ({args.freq_min}) must be less than "
            f"--freq-max ({args.freq_max})"
        )
    nyquist = args.sample_rate / 2.0
    if args.freq_max > nyquist:
        _die(
            f"--freq-max ({args.freq_max}) exceeds Nyquist limit "
            f"({nyquist} Hz) for sample rate {args.sample_rate} Hz"
        )
    if args.duration <= 0:
        _die(f"--duration must be positive, got {args.duration}")


def _die(message: str) -> None:
    """Print *message* to stderr and exit with code 1."""
    print(f"Error: {message}", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Preset application
# ---------------------------------------------------------------------------

def _apply_preset(args: argparse.Namespace) -> None:
    """If --preset is set, load it and apply values as defaults.

    Values from the preset are written to *args* only when the corresponding
    attribute still holds its parser-default value, i.e. the user did not
    explicitly set that flag.
    """
    if args.preset is None:
        return

    from murmur.presets import get_preset

    try:
        preset = get_preset(args.preset)
    except ValueError as exc:
        _die(str(exc))

    # Mapping from preset keys to args attribute names
    _KEY_MAP: dict[str, str] = {
        "freq_min": "freq_min",
        "freq_max": "freq_max",
        "duration": "duration",
        "dither": "dither",
        "blend": "blend",
        # Note: "blend" only exists on the overlay parser, not encode. The
        # hasattr() guard below means blend preset values are silently skipped
        # for `murmur encode` — that is intentional, not a bug.
        "log_frequency": "_preset_log_frequency",  # handled specially below
    }

    for key, value in preset.items():
        if key == "description":
            continue
        if key == "log_frequency":
            # Convert preset's log_frequency bool → freq_scale string
            args.freq_scale = "log" if value else "lin"
        elif hasattr(args, key):
            # Only set if the attribute exists on this subcommand's parser.
            # Keys absent from the subcommand (e.g. blend for encode) are
            # intentionally skipped rather than injected as unknown attributes.
            setattr(args, key, value)

    logger.debug("Applied preset %r: %s", args.preset, preset)


# ---------------------------------------------------------------------------
# Rendering dispatch
# ---------------------------------------------------------------------------

def _resolve_resolution(args: argparse.Namespace) -> tuple[int, int]:
    """Return (width, height) from --resolution or defaults."""
    if args.resolution is not None:
        from murmur.utils import parse_resolution

        try:
            return parse_resolution(args.resolution)
        except ValueError as exc:
            _die(str(exc))
    return _DEFAULT_WIDTH, _DEFAULT_HEIGHT


def _render_input(args: argparse.Namespace) -> np.ndarray:
    """Dispatch to the correct renderer based on *args* and return a 2D float64
    image array with shape (height, width) and values in [0, 1].
    """
    width, height = _resolve_resolution(args)

    try:
        from murmur import renderer
    except ImportError:
        renderer = None  # type: ignore[assignment]

    def _need_renderer(name: str) -> None:
        if renderer is None:
            _die(
                f"murmur.renderer module not found. "
                f"Cannot use --{name.replace('_', '-')} without it."
            )

    if args.image is not None:
        _need_renderer("image")
        logger.debug("Rendering image: %s", args.image)
        return renderer.render_image(args.image, width=width, height=height)

    if args.text is not None:
        _need_renderer("text")
        logger.debug("Rendering text: %r", args.text)
        font = getattr(args, "font", None)
        font_size = getattr(args, "font_size", 48)
        return renderer.render_text(
            args.text, width=width, height=height,
            font_path=font, font_size=font_size,
        )

    if args.ascii is not None:
        _need_renderer("ascii")
        logger.debug("Rendering ASCII: %r", args.ascii)
        return renderer.render_ascii(args.ascii, width=width, height=height)

    if args.qr is not None:
        _need_renderer("qr")
        logo = getattr(args, "qr_logo", None)
        logger.debug("Rendering QR: %r (logo=%s)", args.qr, logo)
        return renderer.render_qr(args.qr, width=width, height=height, logo_path=logo)

    if args.svg is not None:
        _need_renderer("svg")
        logger.debug("Rendering SVG: %s", args.svg)
        return renderer.render_svg(args.svg, width=width, height=height)

    if args.math is not None:
        _need_renderer("math")
        x_range_str = getattr(args, "x_range", "-10:10")
        thickness = getattr(args, "plot_thickness", 3)
        logger.debug("Rendering math: %r", args.math)
        # Parse "MIN:MAX" string → (float, float) tuple expected by render_math
        try:
            _lo, _hi = x_range_str.split(":")
            x_range_tuple = (float(_lo), float(_hi))
        except (ValueError, AttributeError):
            _die(f"--x-range must be in MIN:MAX format (e.g. '-10:10'), got {x_range_str!r}")
            raise RuntimeError("unreachable")
        return renderer.render_math(
            args.math, width=width, height=height,
            x_range=x_range_tuple, thickness=thickness,
        )

    if args.gif is not None:
        _need_renderer("gif")
        logger.debug("Rendering GIF (first frame): %s", args.gif)
        return renderer.render_image(args.gif, width=width, height=height)

    if args.images is not None:
        _need_renderer("images")
        logger.debug("Rendering image sequence: %s", args.images)
        return renderer.render_sequence(
            paths=args.images, frame_width=width, height=height,
        )

    # Should be unreachable — argparse enforces exactly one input.
    _die("No input type specified.")
    raise RuntimeError("unreachable")  # satisfy type checker


# ---------------------------------------------------------------------------
# Encoding pipeline helpers
# ---------------------------------------------------------------------------

def _apply_encoding_options(
    image: np.ndarray, args: argparse.Namespace
) -> tuple[np.ndarray, int, int]:
    """Apply auto-contrast, dither, tile, and invert to *image*.

    Returns:
        (processed_image, freq_bins, time_bins) where freq_bins = image height
        and time_bins = image width after all transforms.
    """
    from murmur.dither import apply_dither

    # Auto-contrast: histogram-stretch to full [0, 1] range
    if getattr(args, "auto_contrast", False):
        lo, hi = image.min(), image.max()
        if hi > lo:
            logger.debug("auto-contrast: stretching [%.4f, %.4f] -> [0, 1]", lo, hi)
            image = (image - lo) / (hi - lo)
        else:
            logger.debug("auto-contrast: image is uniform (lo=hi=%.4f), skipping", lo)

    # Dither
    if args.dither != "none":
        logger.debug("Applying dither: %s", args.dither)
        image = apply_dither(image, method=args.dither)

    # Tile horizontally
    tile = getattr(args, "tile", 1)
    if tile > 1:
        logger.debug("Tiling image %dx horizontally", tile)
        image = np.tile(image, (1, tile))

    # Invert
    if getattr(args, "invert", False):
        logger.debug("Inverting image")
        image = 1.0 - image

    freq_bins, time_bins = image.shape
    return image, freq_bins, time_bins


def _encode_image(image: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    """Run the full encoding pipeline and return audio samples."""
    from murmur.encoder import encode

    log_frequency = (args.freq_scale == "log")
    # image shape is (freq_bins, time_bins) = (height, width); log as WxH (standard convention)
    logger.info(
        "Encoding: %dx%d image (WxH), %.1f Hz - %.1f Hz, %.2f s, %d Hz SR",
        image.shape[1], image.shape[0],
        args.freq_min, args.freq_max,
        args.duration, args.sample_rate,
    )
    return encode(
        image=image,
        sample_rate=args.sample_rate,
        duration=args.duration,
        freq_min=args.freq_min,
        freq_max=args.freq_max,
        log_frequency=log_frequency,
        headroom_db=args.headroom_db,
        randomize_phase=getattr(args, "randomize_phase", False),
        seed=getattr(args, "seed", None),
    )


# ---------------------------------------------------------------------------
# Subcommand: encode
# ---------------------------------------------------------------------------

def _preflight_ffmpeg(output: str) -> None:
    """Fail immediately if a non-WAV output path requires ffmpeg that isn't on PATH."""
    import shutil as _shutil
    if Path(output).suffix.lower() != ".wav":
        if _shutil.which("ffmpeg") is None:
            _die(
                f"Output format '{Path(output).suffix}' requires ffmpeg, but ffmpeg was not "
                "found on PATH. Install it first:\n"
                "  Windows : winget install ffmpeg\n"
                "  macOS   : brew install ffmpeg\n"
                "  Linux   : sudo apt install ffmpeg"
            )


def cmd_encode(args: argparse.Namespace) -> None:
    """Handle the ``encode`` subcommand."""
    _apply_preset(args)
    _validate_encoding_args(args)
    _preflight_ffmpeg(args.output)

    logger.debug("cmd_encode: output=%s", args.output)

    # 1. Render image; remember original frame count for the frame map
    image = _render_input(args)
    original_width = image.shape[1]  # columns before tiling = frame count for sequences

    # 2. Apply auto-contrast / dither / tile / invert
    image, freq_bins, time_bins = _apply_encoding_options(image, args)

    # 3. Encode
    samples = _encode_image(image, args)

    # 4. Apply channel selection — create stereo if left/right specified
    channel = getattr(args, "channel", "both")
    if channel in ("left", "right"):
        stereo = np.zeros((len(samples), 2), dtype=np.float64)
        ch_idx = 0 if channel == "left" else 1
        stereo[:, ch_idx] = samples
        samples = stereo
        logger.debug("encode: channel=%s -> stereo output", channel)

    # 5. Write audio
    from murmur.audio_io import write_audio

    output_path = args.output
    write_audio(output_path, samples, sample_rate=args.sample_rate)

    n_samples = samples.shape[0] if samples.ndim == 2 else len(samples)
    duration_actual = n_samples / args.sample_rate
    tile = getattr(args, "tile", 1)
    print(
        f"Encoded image ({time_bins}x{freq_bins}) -> {output_path} "
        f"({duration_actual:.1f}s, {args.sample_rate}Hz, "
        f"{args.freq_min:.0f}Hz-{args.freq_max:.0f}Hz)"
    )
    if tile > 1:
        print(
            f"Tip: --tile {tile} repeats the content {tile}x within the same duration. "
            f"Use --verify-after-encode for a proportionally correct spectrogram preview; "
            f"external viewers will show each copy compressed {tile}x in time.",
            file=sys.stderr,
        )

    # 6. Frame map — printed when a multi-frame input (--gif / --images) was used
    is_sequence = args.gif is not None or args.images is not None
    if is_sequence and original_width > 1:
        tile = getattr(args, "tile", 1)
        n_frames = original_width  # before tiling
        frame_duration = args.duration / (n_frames * tile)
        print(f"\nFrame time map ({n_frames} frame{'s' if n_frames != 1 else ''}):")
        print(f"  {'Frame':>5}  {'Start':>8}  {'End':>8}")
        print(f"  {'-'*5}  {'-'*8}  {'-'*8}")
        _MAP_HEAD = 5   # rows shown at top and bottom when truncating
        if n_frames <= _MAP_HEAD * 2:
            rows = range(n_frames)
        else:
            rows = list(range(_MAP_HEAD)) + [-1] + list(range(n_frames - _MAP_HEAD, n_frames))
        for i in rows:
            if i == -1:
                print(f"  {'...':>5}  {'...':>8}  {'...':>8}")
                continue
            start = i * frame_duration
            end = start + frame_duration
            print(f"  {i + 1:>5}  {start:>7.2f}s  {end:>7.2f}s")

    # 7. Auto-generate spectrogram after encode
    if getattr(args, "verify_after_encode", False):
        from murmur.spectrogram import generate_spectrogram

        spec_path = str(Path(output_path).with_suffix(".spec.png"))
        # Use mono for spectrogram: take first channel if stereo
        mono = samples[:, 0] if samples.ndim == 2 else samples

        # Compute a spectrogram height so each encoded image pixel appears at a
        # 1:1 (square) aspect ratio in the verification PNG.  The natural
        # spectrogram has ~n_samples/hop_size columns and ~(freq_range × fft_size
        # / sample_rate) rows.  For square pixels we need:
        #   target_H / target_W  =  freq_bins / time_bins
        # Keeping the natural width and solving for height:
        #   target_H = natural_cols × freq_bins / time_bins
        # This corrects the default 256×128 canvas distortion (1.69× tall pixels)
        # so that text, QR codes, and tiled content all appear at their true
        # proportions when the user opens the .spec.png.
        _HOP = 512   # default hop_size used by generate_spectrogram
        _FFT = 4096  # default fft_size used by generate_spectrogram
        _natural_cols = max(1, (len(mono) - _FFT) // _HOP + 1)
        _verify_height = max(32, round(_natural_cols * freq_bins / time_bins))

        # Enforce a minimum of 2 display pixels per encoded frequency row so
        # that letter strokes stay visible.  When --tile compresses the time
        # axis heavily, the square-pixel formula can produce a very short image
        # (e.g. 142 px for tile=3) where each row is sub-pixel and the bottom
        # of letter shapes collapses to nothing.  Scale both dimensions up
        # proportionally so the aspect ratio (square pixels) is preserved.
        _MIN_PX_PER_ROW = 2
        _min_h = _MIN_PX_PER_ROW * freq_bins
        if _verify_height < _min_h:
            _scale_up = _min_h / _verify_height
            _verify_height = _min_h
            _verify_width = round(_natural_cols * _scale_up)
        else:
            _verify_width = None  # use natural spectrogram width

        generate_spectrogram(
            samples=mono,
            sample_rate=args.sample_rate,
            output_path=spec_path,
            freq_min=args.freq_min,
            freq_max=args.freq_max,
            width=_verify_width,
            height=_verify_height,
            log_freq_warp=True,   # Remap linear STFT → log frequency so each
                                  # encoded canvas row occupies equal pixel height.
                                  # Without this, the STFT's 10.77 Hz/bin resolution
                                  # gives the bottom canvas rows (near freq_min) only
                                  # 1-2 px after downscaling, making letter bases and
                                  # QR bottom modules invisible.
        )
        print(f"Spectrogram saved to {spec_path}")


# ---------------------------------------------------------------------------
# Subcommand: overlay
# ---------------------------------------------------------------------------

def cmd_overlay(args: argparse.Namespace) -> None:
    """Handle the ``overlay`` subcommand."""
    _apply_preset(args)
    _validate_encoding_args(args)
    _preflight_ffmpeg(args.output)

    blend = args.blend
    if not (0.0 <= blend <= 1.0):
        _die(f"--blend must be in [0, 1], got {blend}")

    from murmur.audio_io import read_audio, write_audio
    from murmur.mixer import overlay, watermark_overlay
    from murmur.utils import parse_time

    logger.debug("cmd_overlay: carrier=%s, output=%s", args.carrier, args.output)

    # 1. Read carrier; adopt its sample rate so the output always matches the carrier.
    carrier_samples, carrier_sr = read_audio(args.carrier)
    args.sample_rate = carrier_sr

    # 2. Render + encode
    image = _render_input(args)
    image, freq_bins, time_bins = _apply_encoding_options(image, args)
    encoded = _encode_image(image, args)

    # 3. Repeat: tile the encoded signal to fill carrier length
    if getattr(args, "repeat", False) and not args.watermark:
        carrier_len = carrier_samples.shape[0]
        if len(encoded) < carrier_len:
            reps = -(-carrier_len // len(encoded))  # ceiling division
            encoded = np.tile(encoded, reps)[:carrier_len]
            logger.info("repeat: tiled encoded signal to %d samples", carrier_len)

    # 4. Mix
    offset_seconds = parse_time(args.offset)

    if args.watermark:
        logger.info("Applying watermark overlay")
        result = watermark_overlay(
            carrier=carrier_samples,
            encoded=encoded,
            blend=blend,
            sample_rate=args.sample_rate,
            headroom_db=args.headroom_db,
        )
    else:
        logger.info(
            "Applying overlay: blend=%.3f, offset=%.2fs, channel=%s",
            blend, offset_seconds, args.channel,
        )
        result = overlay(
            carrier=carrier_samples,
            encoded=encoded,
            blend=blend,
            offset_seconds=offset_seconds,
            sample_rate=args.sample_rate,
            channel=args.channel,
            headroom_db=args.headroom_db,
        )

    # 5. Write
    write_audio(args.output, result, sample_rate=args.sample_rate)
    print(f"Overlay written to {args.output}")


# ---------------------------------------------------------------------------
# Subcommand: verify
# ---------------------------------------------------------------------------

def cmd_verify(args: argparse.Namespace) -> None:
    """Handle the ``verify`` subcommand."""
    from murmur.audio_io import read_audio
    from murmur.spectrogram import generate_spectrogram

    logger.debug("cmd_verify: input=%s, output=%s", args.input, args.output)

    samples, sample_rate = read_audio(args.input)

    # Mix to mono if stereo
    if samples.ndim == 2:
        logger.debug("Mixing stereo to mono for spectrogram")
        samples = samples.mean(axis=1)

    generate_spectrogram(
        samples=samples,
        sample_rate=sample_rate,
        output_path=args.output,
        freq_min=args.freq_min,
        freq_max=args.freq_max,
        fft_size=args.fft_size,
        hop_size=args.hop_size,
        colormap=args.colormap,
        width=args.width,
        height=args.height,
    )

    print(f"Spectrogram saved to {args.output}")


# ---------------------------------------------------------------------------
# Subcommand: decode
# ---------------------------------------------------------------------------

def cmd_decode(args: argparse.Namespace) -> None:
    """Handle the ``decode`` subcommand."""
    from murmur.audio_io import read_audio
    from murmur.spectrogram import generate_spectrogram

    logger.debug("cmd_decode: input=%s, output=%s", args.input, args.output)

    samples, sample_rate = read_audio(args.input)

    # Mix to mono if stereo
    if samples.ndim == 2:
        logger.debug("Mixing stereo to mono for decode")
        samples = samples.mean(axis=1)

    generate_spectrogram(
        samples=samples,
        sample_rate=sample_rate,
        output_path=args.output,
        freq_min=args.freq_min,
        freq_max=args.freq_max,
        fft_size=args.fft_size,
        hop_size=512,
        colormap="grayscale",
    )

    # Attempt automatic QR/barcode decoding
    try:
        from pyzbar import pyzbar  # type: ignore[import]
        from PIL import Image

        img = Image.open(args.output).convert("L")
        decoded_objects = pyzbar.decode(img)

        if decoded_objects:
            for obj in decoded_objects:
                data = obj.data.decode("utf-8", errors="replace")
                kind = obj.type
                print(f"Decoded ({kind}): {data}")
        else:
            print(
                "No QR/barcode detected automatically. "
                f"Spectrogram saved to {args.output}. "
                "Use a QR scanner app to read it."
            )
    except ImportError:
        print("Install pyzbar for automatic decoding: pip install pyzbar")
        print(
            f"Spectrogram saved to {args.output}. "
            "Use a QR scanner app to read it."
        )


# ---------------------------------------------------------------------------
# Subcommand: presets
# ---------------------------------------------------------------------------

def cmd_probe(args: argparse.Namespace) -> None:
    """Handle the ``probe`` subcommand.

    Analyzes carrier audio spectral content and recommends encoding parameters.
    """
    import json as _json

    from murmur.audio_io import read_audio

    samples, sample_rate = read_audio(args.input)

    # Mix to mono
    if samples.ndim == 2:
        samples = samples.mean(axis=1)

    nyquist = sample_rate / 2.0

    # --- Octave band RMS analysis ------------------------------------------
    # Define octave bands from ~31 Hz up to Nyquist
    center_freqs = []
    f = 31.25
    while f < nyquist:
        center_freqs.append(f)
        f *= 2.0

    # Compute a power spectrum via windowed FFT over the whole signal
    # Use up to ~4 s of audio for speed; longer files are chunked and averaged.
    chunk_size = min(len(samples), int(sample_rate * 4.0))
    fft_size = 1 << (chunk_size - 1).bit_length()  # next power of 2
    fft_size = min(fft_size, 131072)  # cap at 128k

    # Compute magnitude spectrum
    window = hann_window(min(chunk_size, fft_size))
    chunk = samples[:len(window)]
    chunk = chunk * window
    spectrum = np.abs(np.fft.rfft(chunk, n=fft_size))
    freqs_axis = np.fft.rfftfreq(fft_size, d=1.0 / sample_rate)

    # RMS per octave band
    bands = []
    for fc in center_freqs:
        f_lo = fc / (2 ** 0.5)
        f_hi = fc * (2 ** 0.5)
        f_hi = min(f_hi, nyquist)
        mask = (freqs_axis >= f_lo) & (freqs_axis < f_hi)
        if mask.sum() == 0:
            rms_db = -120.0
        else:
            rms = float(np.sqrt(np.mean(spectrum[mask] ** 2)))
            rms_db = 20.0 * np.log10(max(rms, 1e-12))
        bands.append({"center_hz": round(fc, 1), "lo_hz": round(f_lo, 1),
                      "hi_hz": round(f_hi, 1), "rms_db": round(rms_db, 1)})

    # --- Recommendations ---------------------------------------------------
    # Find the loudest and quietest octave bands for masking/visibility hints
    valid_bands = [b for b in bands if b["center_hz"] >= 200 and b["center_hz"] <= nyquist]
    if valid_bands:
        loudest = max(valid_bands, key=lambda b: b["rms_db"])
        quietest = min(valid_bands, key=lambda b: b["rms_db"])

        # Suggest a 3-octave window (factor of 8) around the loudest band for masking
        mask_min = max(200.0, loudest["center_hz"] / 2.0)
        mask_max = min(nyquist, loudest["center_hz"] * 4.0)

        # Overall signal loudness → blend suggestion
        overall_rms = float(np.sqrt(np.mean(samples ** 2)))
        if overall_rms < 0.05:
            blend_hint = 0.5
        elif overall_rms < 0.2:
            blend_hint = 0.3
        else:
            blend_hint = 0.15

        recommendations = {
            "best_masking": {
                "freq_min": round(mask_min),
                "freq_max": round(mask_max),
                "note": f"Carrier is loudest near {loudest['center_hz']:.0f} Hz — "
                        "image will be masked by existing energy",
            },
            "most_visible": {
                "freq_min": round(max(200.0, quietest["center_hz"] / 2.0)),
                "freq_max": round(min(nyquist, quietest["center_hz"] * 4.0)),
                "note": f"Carrier is quietest near {quietest['center_hz']:.0f} Hz — "
                        "image will stand out clearly",
            },
            "blend": round(blend_hint, 2),
            "blend_note": f"Carrier RMS {overall_rms:.3f} -> suggested blend {blend_hint}",
        }
    else:
        recommendations = {"note": "No suitable frequency bands found above 200 Hz"}

    result = {
        "file": args.input,
        "sample_rate": sample_rate,
        "duration_s": round(len(samples) / sample_rate, 2),
        "nyquist_hz": nyquist,
        "octave_bands": bands,
        "recommendations": recommendations,
    }

    if args.json:
        print(_json.dumps(result, indent=2))
        return

    # Human-readable table
    print(f"\nProbe: {args.input}")
    print(f"  Sample rate : {sample_rate} Hz  |  Duration: {result['duration_s']:.1f}s")
    print(f"  Nyquist     : {nyquist:.0f} Hz\n")
    print(f"  {'Band (Hz)':>16}  {'RMS (dB)':>10}  {'Level':}")
    print(f"  {'-'*16}  {'-'*10}  {'-'*20}")
    for b in bands:
        bar_len = max(0, int((b["rms_db"] + 80) / 4))
        bar = "#" * bar_len
        print(f"  {b['lo_hz']:>7.0f}-{b['hi_hz']:<7.0f}  {b['rms_db']:>+9.1f}  {bar}")

    print()
    if "best_masking" in recommendations:
        bm = recommendations["best_masking"]
        mv = recommendations["most_visible"]
        print("  Recommendations:")
        print(f"    Best masking (image hidden): --freq-min {bm['freq_min']} --freq-max {bm['freq_max']}")
        print(f"      {bm['note']}")
        print(f"    Most visible (image clear):  --freq-min {mv['freq_min']} --freq-max {mv['freq_max']}")
        print(f"      {mv['note']}")
        print(f"    Suggested blend:             --blend {recommendations['blend']}")
        print(f"      {recommendations['blend_note']}")
    print()


def cmd_presets(args: argparse.Namespace) -> None:  # noqa: ARG001
    """Handle the ``presets`` subcommand."""
    from murmur.presets import list_presets

    presets = list_presets()

    name_width = max(len(name) for name, _ in presets) + 2
    print(f"{'Preset':<{name_width}}  Description")
    print("-" * (name_width + 2) + "-" * 50)
    for name, description in presets:
        print(f"{name:<{name_width}}  {description}")


# ---------------------------------------------------------------------------
# Parser construction
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    """Construct and return the top-level argument parser."""

    parser = argparse.ArgumentParser(
        prog="murmur",
        description="Embed images into audio spectrograms",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable DEBUG logging",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress all output below WARNING",
    )

    subparsers = parser.add_subparsers(title="subcommands", dest="subcommand")

    # ------------------------------------------------------------------
    # encode
    # ------------------------------------------------------------------
    parser_encode = subparsers.add_parser(
        "encode",
        help="Generate audio with an image embedded in its spectrogram",
        description=(
            "Render an image (from file, text, QR, etc.) and synthesise audio "
            "whose spectrogram reproduces that image."
        ),
    )
    _add_input_group(parser_encode)
    _add_encoding_options(parser_encode)
    parser_encode.add_argument(
        "--output", "-o",
        required=True,
        metavar="PATH",
        help="Output audio file path (e.g. out.wav)",
    )
    parser_encode.add_argument(
        "--channel",
        choices=["both", "left", "right"],
        default="both",
        help=(
            "Stereo channel for the encoded signal (default: both). "
            "Use 'left' or 'right' to create a stereo WAV with content "
            "in only one channel — enables the stereo layer technique."
        ),
    )
    parser_encode.set_defaults(func=cmd_encode)

    # ------------------------------------------------------------------
    # overlay
    # ------------------------------------------------------------------
    parser_overlay = subparsers.add_parser(
        "overlay",
        help="Embed an image signal into an existing carrier audio file",
        description=(
            "Mix the encoded image signal into a carrier audio track at "
            "a specified blend level, time offset, and channel."
        ),
    )
    _add_input_group(parser_overlay)
    _add_encoding_options(parser_overlay)
    parser_overlay.add_argument(
        "--carrier",
        required=True,
        metavar="PATH",
        help="Carrier audio file to embed into",
    )
    parser_overlay.add_argument(
        "--output", "-o",
        required=True,
        metavar="PATH",
        help="Output audio file path",
    )
    parser_overlay.add_argument(
        "--blend",
        type=float,
        default=0.3,
        metavar="FLOAT",
        help="Blend factor for the encoded signal in [0, 1] (default: 0.3)",
    )
    parser_overlay.add_argument(
        "--offset",
        metavar="TIME",
        default="0",
        help=(
            "Time offset at which to insert the signal "
            "(seconds or MM:SS or HH:MM:SS, default: 0)"
        ),
    )
    parser_overlay.add_argument(
        "--channel",
        choices=["both", "left", "right"],
        default="both",
        help="Stereo channel to inject into (default: both)",
    )
    parser_overlay.add_argument(
        "--watermark",
        action="store_true",
        help=(
            "Spread signal across the full carrier duration "
            "(ultra-low amplitude watermark mode)"
        ),
    )
    parser_overlay.add_argument(
        "--repeat",
        action="store_true",
        help=(
            "Loop the encoded signal to fill the full carrier duration. "
            "Useful for watermarking long audio files with a repeating pattern."
        ),
    )
    parser_overlay.set_defaults(func=cmd_overlay)

    # ------------------------------------------------------------------
    # verify
    # ------------------------------------------------------------------
    parser_verify = subparsers.add_parser(
        "verify",
        help="Generate a spectrogram PNG to visually verify embedded content",
        description=(
            "Read an audio file and produce a spectrogram image so you can "
            "confirm the embedded pattern is visible."
        ),
    )
    parser_verify.add_argument(
        "--input", "-i",
        required=True,
        metavar="PATH",
        help="Input audio file",
    )
    parser_verify.add_argument(
        "--output", "-o",
        metavar="PATH",
        default="spectrogram.png",
        help="Output PNG path (default: spectrogram.png)",
    )
    parser_verify.add_argument(
        "--freq-min",
        type=float,
        default=None,
        metavar="FLOAT",
        help="Lower frequency limit for the spectrogram display (Hz)",
    )
    parser_verify.add_argument(
        "--freq-max",
        type=float,
        default=None,
        metavar="FLOAT",
        help="Upper frequency limit for the spectrogram display (Hz)",
    )
    parser_verify.add_argument(
        "--fft-size",
        type=int,
        default=4096,
        metavar="INT",
        help="FFT window size (default: 4096)",
    )
    parser_verify.add_argument(
        "--hop-size",
        type=int,
        default=512,
        metavar="INT",
        help="Hop size between STFT frames (default: 512)",
    )
    parser_verify.add_argument(
        "--colormap",
        choices=["inferno", "grayscale", "viridis"],
        default="inferno",
        help="Spectrogram colour map (default: inferno)",
    )
    parser_verify.add_argument(
        "--width",
        type=int,
        default=None,
        metavar="INT",
        help="Resize spectrogram image to this width (px)",
    )
    parser_verify.add_argument(
        "--height",
        type=int,
        default=None,
        metavar="INT",
        help="Resize spectrogram image to this height (px)",
    )
    parser_verify.set_defaults(func=cmd_verify)

    # ------------------------------------------------------------------
    # decode
    # ------------------------------------------------------------------
    parser_decode = subparsers.add_parser(
        "decode",
        help="Attempt to extract a QR/barcode from an audio spectrogram",
        description=(
            "Generate a greyscale spectrogram from the audio, then try to "
            "decode any QR codes or barcodes visible in it (requires pyzbar)."
        ),
    )
    parser_decode.add_argument(
        "--input", "-i",
        required=True,
        metavar="PATH",
        help="Input audio file",
    )
    parser_decode.add_argument(
        "--freq-min",
        type=float,
        default=None,
        metavar="FLOAT",
        help="Lower frequency limit for the spectrogram (Hz)",
    )
    parser_decode.add_argument(
        "--freq-max",
        type=float,
        default=None,
        metavar="FLOAT",
        help="Upper frequency limit for the spectrogram (Hz)",
    )
    parser_decode.add_argument(
        "--fft-size",
        type=int,
        default=4096,
        metavar="INT",
        help="FFT window size (default: 4096)",
    )
    parser_decode.add_argument(
        "--output", "-o",
        metavar="PATH",
        default="decoded.png",
        help="Spectrogram image output path (default: decoded.png)",
    )
    parser_decode.set_defaults(func=cmd_decode)

    # ------------------------------------------------------------------
    # presets
    # ------------------------------------------------------------------
    parser_presets = subparsers.add_parser(
        "presets",
        help="List all available named presets",
        description="Print a table of preset names and their descriptions.",
    )
    parser_presets.set_defaults(func=cmd_presets)

    # ------------------------------------------------------------------
    # probe
    # ------------------------------------------------------------------
    parser_probe = subparsers.add_parser(
        "probe",
        help="Analyze a carrier audio file and recommend encoding parameters",
        description=(
            "Read an audio file, show spectral energy by octave band, and "
            "suggest --freq-min, --freq-max, and --blend values for optimal "
            "spectrogram visibility."
        ),
    )
    parser_probe.add_argument(
        "--input", "-i",
        required=True,
        metavar="PATH",
        help="Audio file to analyze",
    )
    parser_probe.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON (for scripting)",
    )
    parser_probe.set_defaults(func=cmd_probe)

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Main CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args()

    # Logging setup
    if args.verbose and args.quiet:
        print(
            "Error: --verbose and --quiet are mutually exclusive.",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.verbose:
        log_level = logging.DEBUG
    elif args.quiet:
        log_level = logging.WARNING
    else:
        log_level = logging.INFO

    logging.basicConfig(
        level=log_level,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(0)

    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:  # pylint: disable=broad-except
        logger.debug("Unhandled exception", exc_info=True)
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
