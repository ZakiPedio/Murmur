# Murmur — Technical Documentation

This document covers the internal architecture, algorithms, and extension points of Murmur.
For user-facing usage, see [README.md](README.md).

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Algorithm Deep Dive](#2-algorithm-deep-dive)
3. [Renderer Pipeline](#3-renderer-pipeline)
4. [Mixing Theory](#4-mixing-theory)
5. [Spectrogram Generation](#5-spectrogram-generation)
6. [Dithering](#6-dithering)
7. [Format Support](#7-format-support)
8. [Stereo Encoding](#8-stereo-encoding)
9. [New Encoding Flags](#9-new-encoding-flags)
10. [Carrier Analysis (probe)](#10-carrier-analysis-probe)
11. [QR Code Reliability](#11-qr-code-reliability)
12. [Performance Notes](#12-performance-notes)
13. [Extending Murmur](#13-extending-murmur)
14. [Known Limitations](#14-known-limitations)
15. [Troubleshooting](#15-troubleshooting)

---

## 1. Architecture Overview

### Module Dependency Diagram

```
cli.py
  |
  +-- presets.py          (no deps; pure config dicts)
  |
  +-- renderer.py         (PIL, qrcode; optional cairosvg)
  |     |
  |     +-- PIL (Pillow)
  |     +-- qrcode[pil]
  |     +-- cairosvg      [optional]
  |     +-- ast           (stdlib; safe math parsing)
  |
  +-- dither.py           (numpy only)
  |
  +-- encoder.py          (numpy only)
  |     |
  |     +-- utils.py      (normalize, hann_window)
  |
  +-- mixer.py            (numpy only)
  |     |
  |     +-- utils.py
  |
  +-- audio_io.py         (wave stdlib; optional ffmpeg subprocess)
  |
  +-- spectrogram.py      (numpy, PIL)
  |
  +-- utils.py            (numpy only; no project deps)
```

Every module depends only on modules below it in this tree. There are no circular
imports. `utils.py` is a leaf with no project dependencies.

### Data Flow

```
User Input (file / string / expression)
        |
        v
  renderer.py  -->  grayscale float64 array  (H, W)  in [0, 1]
        |
        v  [optional: --auto-contrast stretches range to [0, 1]]
        |
        v
  dither.py    -->  optionally binarised array  (H, W)  in {0, 1}
        |
        v
  encoder.py   -->  mono audio samples  (N,)  float64  in [-1, 1]
        |
        +-- for overlay: mixer.py  -->  mixed samples  (N,) or (N, 2)
        |
        v
  audio_io.py  -->  WAV file (16-bit PCM)  or  ffmpeg -> other format
        |
        v  [optional: --verify-after-encode generates spectrogram PNG]
```

The probe path runs a separate lightweight analysis pipeline:

```
audio_io.py  -->  samples  -->  cmd_probe (FFT + octave band RMS)  -->  stdout / JSON
```

The verification path runs a separate pipeline:

```
audio_io.py  -->  samples  -->  spectrogram.py  -->  PNG (via PIL)
```

All internal audio is represented as **numpy float64 arrays** normalized to
`[-1.0, 1.0]`. Stereo arrays have shape `(N, 2)` where column 0 is the left
channel and column 1 is the right channel.

---

## 2. Algorithm Deep Dive

### 2.1 Additive Sine Synthesis

The core encoding algorithm is **additive synthesis**: the audio signal is
constructed as a weighted sum of sinusoids, one per frequency bin:

```
          F-1
x(t) =   sum   A[f] * sin(2*pi*freq[f]*t + phi[f])
          f=0
```

Where:

- `F` is the number of frequency bins (= image height)
- `A[f]` is the amplitude of bin `f` at the current time frame (= pixel brightness)
- `freq[f]` is the frequency assigned to row `f` in Hz
- `phi[f]` is the running phase accumulator for bin `f`
- `t` is local sample time within the frame (0 … window_size-1 samples)

### 2.2 Vectorized Synthesis

The inner loop over frequency bins is fully vectorized using numpy broadcasting.
For each time frame, the phase matrix is constructed as an outer product:

```
phi[f, n] = phases[f] + angular_rate[f] * n

where:
  angular_rate[f] = 2*pi*freq[f] / sample_rate   (shape: freq_bins)
  n = local_indices                               (shape: window_size)
  phi                                             (shape: freq_bins x window_size)
```

The frame is then computed as a single matrix-vector multiply (BLAS `dgemv`):

```python
frame = amplitudes @ np.sin(phi)   # (freq_bins,) @ (freq_bins, window_size) -> (window_size,)
```

This replaces the former pure-Python `for f_idx in range(freq_bins)` inner loop
and is approximately 20–50x faster for typical image sizes (128 freq bins ×
256 time bins). Three quantities are pre-computed once before the time loop:

- `angular_rate` — `2*pi*freqs / sample_rate` (shape: `freq_bins`)
- `phase_advance` — `2*pi*freqs * hop_size / sample_rate` (shape: `freq_bins`)
- `local_indices` — `np.arange(window_size)` (shape: `window_size`)

### 2.3 Frequency-to-Row Mapping

Each of the `H` image rows is assigned a frequency. Two modes are supported:

**Logarithmic (default, `--freq-scale log`):**

```
freq[f] = freq_min * (freq_max / freq_min)^(f / (H-1))
```

Equivalently, using `numpy.logspace`:

```python
freqs = np.logspace(log10(freq_min), log10(freq_max), H)[::-1]
```

The reversal places `freq_max` at row 0 (top of image = high frequency) and
`freq_min` at row `H-1` (bottom of image = low frequency). This matches the
visual convention used by most spectrogram viewers.

**Linear (`--freq-scale lin`):**

```
freq[f] = freq_max - f * (freq_max - freq_min) / (H - 1)
```

Logarithmic spacing is strongly preferred for most use cases because it allocates
more rows to the lower octaves (where human hearing is more sensitive) and fewer
rows to the upper octaves. This produces a perceptually uniform frequency axis
that matches what spectrogram viewers typically display by default.

### 2.4 Phase Continuity

Without phase continuity, each frame starts from `phi = 0`. This creates a
click at every frame boundary because the sine wave for bin `f` is abruptly
reset. In the spectrogram domain this manifests as vertical striping artefacts
at each frame boundary.

Murmur avoids this with a **running phase accumulator** per bin. After each
frame, the accumulator is advanced by exactly the phase accumulated during one
hop (not one window):

```
phi[f] += 2*pi * freq[f] * hop_size / sample_rate
phi[f] %= 2*pi
```

Using `hop_size` (not `window_size`) ensures that the phase at the start of
frame `t+1` is exactly continuous with the end of the hop region of frame `t`.
With vectorized synthesis this advance is applied to all bins simultaneously:

```python
phases = (phases + phase_advance) % (2.0 * np.pi)
```

### 2.5 Randomized Phase (`--randomize-phase`)

By default all phase accumulators start at 0. Every frequency bin therefore
starts its sine wave at exactly the same point in its cycle, and when summed
across hundreds of bins the resulting waveform has a characteristic buzzy,
chord-like timbre that is recognizable as synthetic.

When `--randomize-phase` is set, the initial phase for each bin is drawn
uniformly from `[0, 2π)` using `np.random.default_rng`:

```python
rng = np.random.default_rng(seed)   # seed=None -> fresh random; int -> reproducible
phases = rng.uniform(0.0, 2.0 * np.pi, freq_bins)
```

Summing many sine waves with independent random phases produces output that
sounds like band-limited noise (cf. the central limit theorem). The spectrogram
is identical to the zero-phase version — only the waveform and perceptual
quality differ.

The `--seed INT` flag pins the RNG seed, making the output bit-for-bit
reproducible across calls. Without it, a fresh seed is drawn each time via
`np.random.default_rng(None)`.

### 2.6 Overlap-Add Windowing

Each synthesised frame is multiplied by a **Hann window** before being added
to the output buffer:

```
w(n) = 0.5 * (1 - cos(2*pi*n / (N-1)))   for n = 0 ... N-1
```

The Hann window tapers smoothly to zero at both ends. With 50% overlap
(hop size = window size / 2), successive Hann windows sum to a constant:

```
w(n) + w(n + N/2) ≈ 1.0   for all n
```

This **overlap-add** (OLA) property guarantees that the total energy contribution
from each sample position in the output is uniform, regardless of the window
shape. Without OLA, the tapered edges of each window would create periodic
amplitude dips in the output every `hop_size` samples.

### 2.7 Normalization and Headroom

After synthesis, the signal is peak-normalized:

```
target_peak = 10^(headroom_db / 20)
output = output * (target_peak / max(|output|))
```

With the default `headroom_db = -0.5`, the peak amplitude is:

```
target_peak = 10^(-0.5/20) ≈ 0.944  (full scale)
```

This leaves a small safety margin below digital full scale. The normalization
is applied after overlap-add so the full dynamic range of the synthesized
signal is preserved, not each individual frame.

---

## 3. Renderer Pipeline

Every renderer in `murmur/renderer.py` produces a 2D `float64` numpy array
with shape `(height, width)` and values in `[0.0, 1.0]`. This array is passed
directly to the encoder where height becomes the number of frequency bins and
width becomes the number of time frames.

### Conversion pipeline for all renderers

1. Generate or load the image at the requested pixel size
2. Convert to greyscale (mode `"L"` in Pillow)
3. Convert to `float64` and divide by 255 to get values in `[0, 1]`
4. Return the array with shape `(height, width)`

### Auto-contrast (`--auto-contrast`)

After the renderer returns the image array, the CLI optionally applies a
histogram stretch before dithering:

```python
lo, hi = image.min(), image.max()
if hi > lo:
    image = (image - lo) / (hi - lo)
```

This maps the darkest pixel to 0.0 and the brightest to 1.0, giving the full
dynamic range to the encoder regardless of the source image's original contrast.
It is applied after rendering but before dithering, so dithering operates on
the full-range image.

**When to use it:** Any source image that is uniformly dim or bright will
produce a spectrogram with reduced contrast. A scanned photograph, a faded
image, or a synthetic gradient that only covers a narrow slice of [0, 1] will
all benefit. Avoid it for images that already fill [0, 1], as it is a no-op
for those.

### Per-renderer details

| Input flag | Module function | Key behaviour |
|------------|----------------|--------------|
| `--image` | `render_image` | Load via PIL; resize with Lanczos; RGBA composited on white |
| `--text` | `render_text` | System font auto-detected per platform; text centred on black background |
| `--ascii` | `render_ascii` | Monospace font required; line-wrapped to fit width |
| `--qr` | `render_qr` | `qrcode` library, error correction H; optional logo composited centred |
| `--svg` | `render_svg` | Requires `cairosvg`; raises `ImportError` with install hint otherwise |
| `--math` | `render_math` | `ast.parse` for safe evaluation; plots `y = f(x)` using thickness-controlled drawing |
| `--gif` | `render_image` | PIL opens GIF; `seek(0)` extracts frame 0; processed identically to `--image` |
| `--images` | `render_sequence` | Each image rendered at `(height, frame_width)` and concatenated horizontally; frame count printed as a time map after encoding |

### Platform font detection

Text renderers attempt to load a system TrueType font using a hard-coded path
list keyed by `platform.system()`:

```
Windows : C:/Windows/Fonts/arial.ttf, verdana.ttf, ...
macOS   : /System/Library/Fonts/Helvetica.ttc, ...
Linux   : /usr/share/fonts/truetype/dejavu/DejaVuSans.ttf, ...
```

If no system font is found, Pillow's built-in bitmap font is used as a
fallback (small, fixed size, no bold/italic). Use `--font PATH` to specify a
custom TTF/OTF file and bypass auto-detection entirely.

### Safe math evaluation

The `--math` renderer uses `ast.parse` and a whitelist of safe node types to
evaluate expressions. `eval()` is never called. Supported nodes include
`Expression`, `BinOp`, `UnaryOp`, `Call`, `Name`, `Constant`, and the
standard arithmetic operators. Supported functions: `sin`, `cos`, `tan`,
`exp`, `log`, `sqrt`, `abs`, `pi`, `e`.

The CLI parses the `--x-range MIN:MAX` string (e.g. `"-6.28:6.28"`) into a
`(float, float)` tuple before passing it to `render_math`.

---

## 4. Mixing Theory

### Overlay mechanics

`mixer.overlay` adds a scaled copy of the encoded signal into the carrier:

```
result[t_start : t_start + N] += encoded * blend
result = normalize(result)
```

The blend factor is a **linear amplitude multiplier**, not a power ratio.
At `blend = 0.3`, the encoded signal is injected at 30% of its synthesized
peak amplitude before normalization. After normalization, the actual
amplitude ratio depends on the carrier's RMS level.

Practical amplitude relationships:

| Blend | Approx dB below full scale | Typical audibility |
|-------|--------------------------|-------------------|
| 0.5 | −6 dB | Clearly audible |
| 0.3 | −10 dB | Audible with attention |
| 0.1 | −20 dB | Subtle |
| 0.05 | −26 dB | Hard to hear |
| 0.03 | −30 dB | Near-inaudible (watermark) |

### Stereo channel injection

When the carrier is stereo (shape `(N, 2)`), the encoded mono signal is added
into column 0 (`left`), column 1 (`right`), or both, depending on `--channel`.
The carrier's other channel is untouched.

This enables the dual-layer technique: encode two independent images by running
`overlay` twice with `--channel left` then `--channel right`. When the stereo
file is loaded in a spectrogram viewer with split channels, each channel
shows a different image.

### Watermark overlay

The watermark path uses `numpy.interp` to time-stretch the encoded signal from
its natural length to exactly the carrier length before mixing. This is linear
sample interpolation (not resampling with anti-aliasing), which is sufficient
for the very low-amplitude, spread-spectrum purpose:

```python
new_indices = np.linspace(0.0, len(encoded) - 1, carrier_len)
stretched = np.interp(new_indices, orig_indices, encoded)
```

The stretched signal is then mixed with `blend = 0.03` (default) across the
entire carrier. Recovery requires a large FFT size (8192–16384 samples) and
matching the original frequency band parameters exactly.

### Repeat (`--repeat`)

The `--repeat` flag in `overlay` mode tiles the encoded signal (using
`np.tile`) to fill the full carrier length before mixing, without time-stretching.
This is distinct from watermark mode:

| Mode | Mechanism | Amplitude | Pitch |
|------|-----------|-----------|-------|
| Default | Inject once at offset | Normal | Normal |
| `--watermark` | Stretch to carrier length | Very low (~0.03) | Compressed |
| `--repeat` | Tile to carrier length | Normal | Normal |

Use `--repeat` when you want the same pattern to repeat at its natural tempo
throughout a long carrier. Use `--watermark` when you want a single ultra-quiet
forensic marker spread across the full file.

---

## 5. Spectrogram Generation

`murmur/spectrogram.py` implements a manual **Short-Time Fourier Transform
(STFT)** without scipy or matplotlib.

### STFT implementation

1. **Pad** the input to ensure the last frame is complete
2. **Frame matrix**: extract overlapping windows as a 2D array using numpy
   strides (`frames[t, :] = samples[t*hop : t*hop + fft_size]`)
3. **Window**: multiply every row by a Hann window (broadcast)
4. **FFT**: `np.fft.rfft(frames, n=fft_size, axis=1)` — produces shape
   `(T, fft_size//2 + 1)` where T is the number of frames
5. **Magnitude**: `np.abs(fft_result)` — discard phase
6. **dB conversion**: `20 * log10(magnitude + 1e-10)` — add epsilon to
   avoid log(0)
7. **Crop**: extract only the frequency bins corresponding to `[freq_min, freq_max]`
8. **Normalize** to uint8 in `[0, 255]` using global min/max of the dB array
9. **Orient**: transpose to `(F, T)`, flip vertically so low frequencies
   appear at the bottom of the image (standard spectrogram convention)
10. **Colormap**: apply a 256-entry LUT (inferno, grayscale, or viridis)
11. **Resize** (optional): Lanczos resampling via PIL
12. **Save** as PNG

### FFT size tradeoffs

| FFT size | Freq resolution | Time resolution | Memory | Recommended for |
|----------|----------------|----------------|--------|----------------|
| 1024 | ~43 Hz/bin @ 44100 Hz | Good | Low | Quick preview |
| 2048 | ~21 Hz/bin | Moderate | Low | General use |
| 4096 | ~10.8 Hz/bin | Moderate | Medium | Default; good for verify |
| 8192 | ~5.4 Hz/bin | Coarser | Medium | QR decoding, detailed inspection |
| 16384 | ~2.7 Hz/bin | Coarse | High | Watermark recovery |

Frequency resolution = `sample_rate / fft_size`. Time resolution
(per frame) = `hop_size / sample_rate` seconds.

### Colourmap implementation

The three built-in colourmaps (inferno, grayscale, viridis) are defined as
lists of RGB control points. A 256-entry lookup table is generated by linearly
interpolating between control points spaced evenly across [0, 255]:

```python
lut = np.zeros((256, 3), dtype=float)
# for each segment between control points:
lut[lo:hi+1, ch] = np.linspace(control[i][ch], control[i+1][ch], seg_len+1)
```

The LUT is applied via direct numpy indexing (`lut[uint8_array]`), which is
effectively a vectorised table lookup and runs in a single pass with no Python
loop.

---

## 6. Dithering

Dithering converts a continuous-tone greyscale image into an approximation
using only binary (black/white) pixels. This is useful when the encoding
resolution is low and individual sine waves either are fully on or fully off,
rather than at intermediate amplitudes.

### Threshold dithering

Each pixel is compared to a fixed cutoff (default 0.5):

```
output[y, x] = 1.0 if image[y, x] >= 0.5 else 0.0
```

Produces the sharpest possible edges. Best for QR codes, barcodes, and text,
where the hard boundaries of the symbols are more important than tonal accuracy.
Fully vectorized with `np.where`.

### Ordered (Bayer) dithering

A 4x4 Bayer matrix of threshold values is tiled across the image:

```
Bayer 4x4 (normalized to [0, 1)):
 0/16   8/16   2/16  10/16
12/16   4/16  14/16   6/16
 3/16  11/16   1/16   9/16
15/16   7/16  13/16   5/16
```

Each pixel is compared to the corresponding Bayer threshold value. Produces a
structured crosshatch pattern that is very regular and visually distinctive.
Fast to compute: no error diffusion loop required. Fully vectorized with
`np.tile` and `np.where`.

### Floyd-Steinberg dithering

Error diffusion: after quantizing each pixel to black or white, the quantization
error is distributed to neighbours:

```
error = old_value - new_value

right        += error * 7/16
bottom-left  += error * 3/16
bottom       += error * 5/16
bottom-right += error * 1/16
```

This produces the most visually faithful approximation of continuous-tone images.
Gradients appear smooth rather than banded.

**Implementation:** The inner pixel loop cannot be fully vectorized because the
7/16 rightward error carries a sequential dependency — each pixel depends on
accumulated errors from its left neighbour. Murmur uses a **padded-buffer
scanline** approach to eliminate all per-pixel boundary checks:

1. The image is copied into a zero-padded buffer of shape `(H+1, W+2)` — one
   extra row on the bottom and one extra column on each side.
2. The outer row loop iterates in Python, taking 1D views of the current and
   next rows.
3. The inner column loop writes to `row[x+1]`, `nxt[x-1]`, `nxt[x]`,
   `nxt[x+1]` unconditionally, because the padding absorbs all out-of-bounds
   writes.

This removes the four `if` boundary checks per pixel that the naive
implementation requires, giving a meaningful speedup for large images. The row
loop stays in Python because of the sequential column dependency.

### Visual quality comparison

| Input | Threshold | Ordered | Floyd-Steinberg |
|-------|-----------|---------|----------------|
| QR code | Best (sharpest) | Acceptable | Acceptable |
| Text | Best | Acceptable | Acceptable |
| Photo | Poor (flat) | Banded | Best |
| Gradient | Poor | Banded | Excellent |
| Logo | Good | Good | Good |

---

## 7. Format Support

### WAV (native)

WAV files are read and written directly using Python's `stdlib wave` module.
No external dependencies required.

- **Read**: 16-bit (`<i2`) and 24-bit (manual 3-byte unpacking) PCM are
  supported. Samples are converted to `float64` in `[-1, 1]`.
- **Write**: Always outputs 16-bit PCM (`int16`, multiply float64 by 32767).
  The 32767 multiplier (not 32768) avoids overflow at exactly `+1.0`.
- **Channels**: Mono `(N,)` and stereo `(N, 2)` arrays are handled.
  Multi-channel WAV files beyond stereo are read but the caller must handle
  the channel layout.

### Non-WAV formats via ffmpeg

Any format not recognized as `.wav` is transcoded through a temporary WAV file:

**Decoding (read):**
```
source_file  -->  [ffmpeg -i source -f wav -acodec pcm_s16le]  -->  temp.wav  -->  _read_wav()
```

**Encoding (write):**
```
_write_wav()  -->  temp.wav  -->  [ffmpeg -y -i temp.wav output.mp3/flac/ogg]  -->  output
```

ffmpeg is invoked via `subprocess.run` with its output captured. If ffmpeg
exits with a non-zero code, the stderr output is included in the `RuntimeError`
message to aid debugging.

`shutil.which("ffmpeg")` is used to detect availability. If ffmpeg is not
found, a user-friendly error message with platform-specific install instructions
is raised before any transcoding is attempted.

#### Pre-flight ffmpeg validation

In addition to the at-write check inside `audio_io.py`, the CLI performs a
**pre-flight check** at the very start of `cmd_encode` and `cmd_overlay` (before
any rendering or synthesis):

```python
def _preflight_ffmpeg(output: str) -> None:
    if Path(output).suffix.lower() != ".wav":
        if shutil.which("ffmpeg") is None:
            _die("Output format '...' requires ffmpeg, but ffmpeg was not found on PATH. ...")
```

This means a missing ffmpeg is detected immediately — you get a clear error
with install instructions before waiting for synthesis to complete. Without the
pre-flight check, a user requesting a 10-minute MP3 encode would wait the full
synthesis time only to see the error at the very end.

### MP3 artifacts and `--mp3-safe`

MP3 uses lossy compression that removes spectral components it deems
perceptually unimportant. This directly degrades spectrogram quality:

- High-frequency content above ~16 kHz is heavily filtered or removed
- Tonal content at specific frequencies may be attenuated or masked
- The quantization noise floor rises, obscuring low-amplitude encoded regions
- Encoded images in the spectrogram will appear blurry or partially erased

**`--mp3-safe` flag:** Clamps `--freq-min` and `--freq-max` to the empirically
safe window of 200–16000 Hz. If either value falls outside this window, a warning
is printed to stderr and the value is adjusted:

```
Warning: --freq-max 20000 Hz exceeds the MP3-safe ceiling (16000 Hz) — clamping.
Note: the embedded image will survive MP3 re-encoding within this range.
```

The clamping is applied before all other validation, so subsequent checks
(e.g. Nyquist limit) operate on the adjusted values. The constants
`_MP3_FREQ_MIN = 200.0` and `_MP3_FREQ_MAX = 16000.0` are defined at module
level in `cli.py` for easy adjustment.

**Recommendation:** Use WAV for all steganographic work. Convert to MP3 only
if the carrier format requires it. If MP3 output is required, use `--mp3-safe`,
increase `--blend` to compensate for amplitude losses, and prefer high bitrate
settings (320 kbps via ffmpeg `-b:a 320k`).

---

## 8. Stereo Encoding

### L/R channel separation

Murmur encodes mono audio internally. The `mixer.overlay` function handles
stereo carriers by selecting which column of the `(N, 2)` array to modify:

```python
if channel in ("both", "left"):
    result[sl, 0] += scaled   # left channel
if channel in ("both", "right"):
    result[sl, 1] += scaled   # right channel
```

When `--channel both` is used (default), the identical encoded mono signal is
added to both channels. When `--channel left` or `--channel right` is used,
only that channel is modified. The other channel retains its original carrier
signal untouched.

The `encode` subcommand also accepts `--channel left/right`: it produces a
stereo WAV where one channel contains the encoded signal and the other is
silent, without needing a carrier:

```python
stereo = np.zeros((len(samples), 2), dtype=np.float64)
stereo[:, 0 if channel == "left" else 1] = samples
```

### Dual-layer technique in practice

```bash
# Pass 1: left channel
murmur overlay --image imageA.png --carrier carrier.wav \
    --channel left --blend 0.5 --output pass1.wav

# Pass 2: right channel of pass1.wav
murmur overlay --image imageB.png --carrier pass1.wav \
    --channel right --blend 0.5 --output dual.wav
```

Separating the channels in Audacity (*Track > Split Stereo Track*) reveals
two independent spectrograms, one per channel.

### Practical considerations

- Normalization in `overlay` is applied to the full stereo array after mixing.
  If one channel has a very high amplitude, normalization may reduce the other
  channel's level. Balance carrier levels before overlaying.
- `murmur verify` mixes stereo to mono (`samples.mean(axis=1)`) before
  generating the spectrogram. Use your viewer's channel isolation to inspect
  individual layers.

---

## 9. New Encoding Flags

This section documents the four encoding flags added alongside the vectorized
encoder. All four flags are available to both `encode` and `overlay`.

### `--auto-contrast`

**Problem it solves:** A source image with low dynamic range (all pixels
clustered near mid-grey, or a washed-out photograph) will produce a spectrogram
where the embedded content is barely visible — the sine wave amplitudes are all
close to the same value, so there is little contrast between bright and dark
regions.

**How it works:** After rendering the image but before dithering, the pipeline
applies a linear histogram stretch:

```python
lo, hi = image.min(), image.max()
image = (image - lo) / (hi - lo)   # maps [lo, hi] -> [0, 1]
```

If the image is already uniform (`lo == hi`), the stretch is skipped. The
operation is logged at DEBUG level.

**When to use:** Any faint, washed-out, or low-contrast source image. Has no
effect on images that already span the full [0, 1] range, and does not change
the shape or dithering of the image.

---

### `--randomize-phase`

**Problem it solves:** The default zero-phase synthesis produces output with a
recognizable buzzy timbre — a chord of hundreds of simultaneous pure tones.
This can be a giveaway that something unusual has been encoded, and it sounds
unpleasant when the frequency range is audible.

**How it works:** The per-frequency initial phase vector is drawn from a
uniform distribution instead of being set to zero:

```python
rng = np.random.default_rng(seed)                        # seed=None → fresh; int → reproducible
phases = rng.uniform(0.0, 2.0 * np.pi, freq_bins)        # when randomize_phase=True
phases = np.zeros(freq_bins)                             # default
```

By the central limit theorem, summing many sinusoids with independent random
phases approximates a Gaussian noise process. The output sounds like
band-limited white noise rather than a chord.

**Trade-offs:**
- The spectrogram appearance is identical — this only affects the audio waveform
- Output is non-reproducible by default (fresh seed each call); use `--seed` to fix it
- Particularly useful for audible frequency ranges (200–8000 Hz) where the
  chord effect is most noticeable
- Less important for stealth bands (14–20 kHz) where the output is barely audible

---

### `--seed INT`

**Problem it solves:** `--randomize-phase` draws a fresh random seed each call,
so two encodes of the same image produce different audio files. This is intentional
for production use (each copy sounds slightly different), but breaks reproducibility
for testing, CI pipelines, and A/B comparisons.

**How it works:** The seed is forwarded directly to `np.random.default_rng`:

```python
rng = np.random.default_rng(seed)   # int -> reproducible; None -> fresh
phases = rng.uniform(0.0, 2.0 * np.pi, freq_bins)
```

Using the Generator API (`default_rng`) rather than the legacy `np.random.seed()`
keeps the randomness isolated to this call and does not affect global numpy state.

**When to use:** Any time you need deterministic output from `--randomize-phase` —
regression tests, automated pipelines, demos where you want to distribute
pre-computed audio that others can reproduce.

**Note:** `--seed` has no effect without `--randomize-phase`. When
`randomize_phase=False`, phases are initialized to zeros regardless of `seed`.

---

### `--mp3-safe`

**Problem it solves:** When the output WAV will be re-encoded as MP3, content
outside approximately 200–16000 Hz will be attenuated or removed by the codec.
Users who set `--freq-max 20000` would silently produce a file that degrades
badly under MP3 encoding.

**How it works:** Applied during argument validation, before encoding begins:

```python
_MP3_FREQ_MIN = 200.0
_MP3_FREQ_MAX = 16000.0

if args.freq_min < _MP3_FREQ_MIN:
    # warn and clamp
if args.freq_max > _MP3_FREQ_MAX:
    # warn and clamp
```

Warnings are printed to stderr so they are visible even when output is piped.
A note about MP3 survival is also printed.

**Relationship to `--freq-min`/`--freq-max`:** The clamping replaces the
user-specified values in `args` before any further processing, so subsequent
validation (Nyquist check, `freq_min < freq_max`) operates on the clamped values.

---

### `--verify-after-encode`

**Problem it solves:** The natural next step after `murmur encode` is always
`murmur verify` to confirm the image is visible in the spectrogram. Having to
run a second command with the right input/output paths is friction.

**How it works:** After `write_audio` completes, the CLI computes a spectrogram
from the in-memory samples and saves it as `<stem>.spec.png` next to the output
WAV:

```python
spec_path = Path(output_path).with_suffix(".spec.png")
generate_spectrogram(
    samples=mono,
    sample_rate=args.sample_rate,
    output_path=str(spec_path),
    freq_min=args.freq_min,
    freq_max=args.freq_max,
)
```

The spectrogram uses the same `freq_min`/`freq_max` as the encoding, so the
displayed frequency range exactly matches what was encoded. If the output is
stereo (from `--channel left/right`), the first channel is used for the
spectrogram. The default spectrogram settings (FFT size 4096, inferno colormap)
are used; for custom settings, use `murmur verify` separately.

---

## 10. Carrier Analysis (`probe`)

The `probe` subcommand performs spectral analysis on a carrier audio file and
produces recommendations for encoding parameters.

### Algorithm

1. **Read audio** and mix to mono.
2. **Windowed FFT**: take up to 4 seconds of audio, zero-pad to the next power
   of 2 (capped at 128k samples), apply a Hann window, compute `np.fft.rfft`.
3. **Octave bands**: for each octave center frequency `fc` from 31.25 Hz up to
   Nyquist (doubling each step), compute the RMS of the magnitude spectrum within
   `[fc/√2, fc*√2]`:
   ```python
   rms_db = 20 * log10(sqrt(mean(spectrum[mask] ** 2)))
   ```
4. **Loudest band** — the octave band with the highest RMS in the 200 Hz+ range.
5. **Quietest band** — the octave band with the lowest RMS in the 200 Hz+ range.
6. **Blend estimate** — derived from overall signal RMS:
   - RMS < 0.05 → blend 0.5
   - RMS < 0.20 → blend 0.3
   - RMS ≥ 0.20 → blend 0.15

### Masking vs. visibility recommendations

The probe produces two `--freq-min`/`--freq-max` recommendations:

**Best masking** — places the encoded image inside the loudest band.
The existing carrier energy acts as a "mask" that makes the encoded signal
less perceptible to listeners and harder to distinguish visually in the
spectrogram. Use this when you want the encoding to be covert.

**Most visible** — places the encoded image in the quietest band.
The signal has no competing carrier energy and will appear with maximum
brightness contrast in the spectrogram. Use this for artistic work or when
you need the image to be as readable as possible.

### JSON output (`--json`)

All analysis results are available as structured JSON for scripting:

```json
{
  "file": "music.wav",
  "sample_rate": 44100,
  "duration_s": 30.0,
  "nyquist_hz": 22050,
  "octave_bands": [
    {"center_hz": 31.2, "lo_hz": 22.1, "hi_hz": 44.2, "rms_db": -42.1},
    ...
  ],
  "recommendations": {
    "best_masking": {"freq_min": 354, "freq_max": 2828, "note": "..."},
    "most_visible": {"freq_min": 8000, "freq_max": 16000, "note": "..."},
    "blend": 0.3,
    "blend_note": "Carrier RMS 0.142 -> suggested blend 0.3"
  }
}
```

---

## 11. QR Code Reliability

### Error correction levels

Murmur uses `qrcode.ERROR_CORRECT_H` when generating QR codes (30% error
correction). This is the highest available level and allows up to 30% of the
QR code data modules to be damaged or unreadable while the code remains
decodable.

### Logo size limits

When `--qr-logo` is used, the logo is composited over the centre of the QR
code. The maximum recommended logo size is **30% of the QR code area**
(i.e., the logo should not cover more than 30% of the module grid). Exceeding
this limit can push the error rate above the correction capacity.

Murmur does not enforce a logo size limit; if the logo is too large and the
QR code fails to scan, reduce the logo with `--resolution` or resize the
logo image before using it.

### Spectrogram resolution effects

The FFT size used during verification (`--fft-size` in `murmur verify`) must
be large enough to resolve individual QR code modules as distinct bright/dark
regions. Each module must map to at least 2–4 frequency bins to be reliably
distinguished.

Given:
- Encoded resolution: `H` rows (frequency bins)
- Frequency range: `freq_min` to `freq_max`
- FFT size: `fft_size`
- Sample rate: `sample_rate`

The number of FFT bins covering the encoded range is:

```
visible_bins = fft_size * (freq_max - freq_min) / (sample_rate / 2)
```

For QR scanning, `visible_bins` should be at least `4 * H`. So:

```
fft_size >= 4 * H * (sample_rate / 2) / (freq_max - freq_min)
```

Example: H=64, SR=44100, freq range 200–8000 Hz:
```
fft_size >= 4 * 64 * 22050 / 7800 ≈ 724  (use 1024 or larger)
```

---

## 12. Performance Notes

### Encoder performance

The encoder's synthesis loop iterates over `time_bins` frames (one per image
column). Within each frame, all `freq_bins` sine waves are computed in a
single vectorized operation:

```
phi    : (freq_bins, window_size)  float64  — instantaneous phase matrix
frame  : (window_size,)            float64  — weighted sum via BLAS matmul
```

Memory per frame: `freq_bins * window_size * 8 bytes`. For 128 bins and a
typical `window_size` of ~1720 samples: ~1.8 MB per frame, allocated and
discarded each iteration (numpy manages this efficiently via temporary arrays).

### Expected processing times

Times are approximate and depend on CPU, Python version, and numpy BLAS backend.

| Operation | Parameters | Approx time (post-vectorization) |
|-----------|-----------|----------------------------------|
| `encode --text` | 256×128, 10s, 44100 Hz | ~0.5–1 s |
| `encode --image` | 256×128, 10s, 44100 Hz | ~0.5–1 s |
| `encode --image` | 512×256, 20s, 44100 Hz | ~3–5 s |
| `overlay` | same as encode + WAV read | encode time + <1 s |
| `verify` | 30s WAV, fft_size=4096 | < 2 s |
| `probe` | 30s WAV | < 1 s |
| Floyd-Steinberg dither | 256×128 | < 0.2 s |

The vectorized encoder is approximately 20–50x faster than the former
pure-Python inner loop, making real-time interactive use practical for
standard resolutions.

### Memory usage

| Data type | Size formula | Example (256×128, 10s) |
|-----------|-------------|----------------------|
| Image array | H × W × 8 bytes | 262 KB |
| Audio (mono) | N × 8 bytes (float64) | 3.4 MB (441k samples) |
| Phase matrix (per frame) | freq_bins × window_size × 8 | ~1.8 MB (transient) |
| STFT frame matrix (verify) | T × fft_size × 8 bytes | variable |

Total peak memory for a typical encode is under 50 MB.

---

## 13. Extending Murmur

### Adding a new renderer

1. Add a function to `murmur/renderer.py` with signature:

```python
def render_mytype(
    data: str,
    width: int = 256,
    height: int = 128,
    **kwargs,
) -> np.ndarray:
    """Return a (height, width) float64 array in [0, 1]."""
    ...
```

2. Add a `--mytype` argument to `_add_input_group` in `murmur/cli.py`.

3. Add a dispatch branch in `_render_input` in `murmur/cli.py`:

```python
if args.mytype is not None:
    _need_renderer("mytype")
    return renderer.render_mytype(args.mytype, width=width, height=height)
```

4. Add tests in `tests/test_renderer.py`.

The renderer must always return a 2D `float64` array with values clamped to
`[0, 1]`. Shape must be `(height, width)` — height is the frequency axis,
width is the time axis.

### Adding a new preset

1. Add an entry to the `PRESETS` dict in `murmur/presets.py`:

```python
PRESETS["mypreset"] = {
    "description": "One-line description shown by murmur presets",
    "freq_min": 500,
    "freq_max": 6000,
    "duration": 12.0,
    "dither": "none",
    "blend": 0.25,
    "log_frequency": True,
}
```

Valid keys and their types:

| Key | Type | Maps to CLI flag |
|-----|------|-----------------|
| `description` | str | (display only) |
| `freq_min` | float | `--freq-min` |
| `freq_max` | float | `--freq-max` |
| `duration` | float | `--duration` |
| `dither` | str | `--dither` |
| `blend` | float | `--blend` (overlay only) |
| `log_frequency` | bool | `--freq-scale log/lin` |

2. Add a test case in `tests/test_presets.py`.

---

## 14. Known Limitations

### Frequency resolution

The number of distinct frequencies that can be reliably distinguished in
the output audio is bounded by the FFT resolution:

```
min_bin_spacing = sample_rate / fft_size
```

At the default FFT size of 4096 and 44100 Hz sample rate, two frequencies
must differ by at least ~10.8 Hz to appear as separate bins. If the image
height (`freq_bins`) is large and the frequency range is narrow, adjacent
rows may alias into the same FFT bin and appear merged in the spectrogram.

### MP3 and lossy compression artifacts

MP3, AAC, and OGG Vorbis all use perceptual audio coding that intentionally
removes or reduces spectral components deemed inaudible. Always encode to WAV
and convert to lossy formats only if the use case allows visible degradation.
Use `--mp3-safe` when MP3 output is required — it constrains the frequency
range to what MP3 can preserve.

### High frequencies near Nyquist

At very high frequencies (above ~18 kHz at 44100 Hz), the sine synthesis
frequency approaches the Nyquist limit. Near Nyquist, the Hann window applied
during synthesis can cause amplitude modulation of the synthesized tone. Most
audio hardware also rolls off sharply between 18–22 kHz. Mitigation: keep
`--freq-max` at or below 18000 Hz for reliable results.

### QR code scanning reliability from spectrograms

A spectrogram is a visual approximation of frequency content, not a perfect
reproduction of the source image. Several factors reduce QR scan reliability:

- **Amplitude smearing**: The Hann window spreads energy into adjacent bins,
  blurring module boundaries.
- **FFT resolution mismatch**: If the spectrogram FFT size does not produce
  enough bins to resolve each QR module, modules will merge.
- **Carrier bleed**: In overlay mode, the carrier contributes spectral energy
  that can obscure the QR pattern.

Mitigation: use `murmur encode` (not overlay) for maximum QR reliability, use
`--dither threshold`, and verify with a large `--fft-size` matching the encoded
frequency range.

---

## 15. Troubleshooting

### Clicks or pops in the output audio

**Cause:** Phase discontinuities at frame boundaries. This should not happen
with the current implementation, which maintains per-bin phase accumulators.
If clicks occur:

- Check that `headroom_db` is negative (e.g. `-0.5`) so the normalized signal
  does not clip at `1.0`.
- Verify that the carrier (for overlay mode) is not already clipping.
- Ensure the frequency range does not exceed Nyquist (`sample_rate / 2`).

### Blurry or low-contrast spectrogram

**Cause:** FFT size too small to resolve individual frequency bins.

- Increase `--fft-size` in `murmur verify` to 4096, 8192, or 16384.
- Narrow the display range with `--freq-min` and `--freq-max` to match
  encoding parameters.
- If using overlay mode, reduce `--blend` on the carrier.
- Use `--auto-contrast` when encoding to ensure the image fills the full
  amplitude range.

### QR code not scanning

1. Verify the spectrogram is visible: `murmur verify --input out.wav --fft-size 8192 --output spec.png`
2. If the pattern is visible but not scanning, increase `--resolution` when encoding.
3. Use `--dither threshold` for binary QR modules.
4. Increase `--blend` if using overlay mode (try 0.5–0.8).
5. Crop the spectrogram image to the QR area before scanning.
6. Check that `libzbar` is installed (Linux: `sudo apt install libzbar0`).

### Spectrogram is visible but tonally obvious

**Symptom:** The encoded audio has a recognizable buzzy or chord-like sound that
makes the steganography obvious on casual listening.

**Fix:** Use `--randomize-phase` to replace the buzzy chord with band-limited
noise. The spectrogram content is identical, but the audio waveform sounds
natural. Particularly effective for audible frequency ranges (200–8000 Hz).

### Output sounds wrong after MP3 round-trip

**Symptom:** The spectrogram looks distorted or the embedded image is barely
visible after encoding to MP3.

**Fix:** Use `--mp3-safe` to constrain the frequency range to 200–16000 Hz.
Re-encode with a higher MP3 bitrate (`ffmpeg -b:a 320k`). If the image is still
degraded, increase `--blend` to compensate for MP3 amplitude losses.

### ffmpeg not found

**Symptom:** `RuntimeError: ffmpeg not found. Install it: ...`

```bash
# Windows
winget install Gyan.FFmpeg

# macOS
brew install ffmpeg

# Debian / Ubuntu
sudo apt update && sudo apt install ffmpeg
```

After installation, verify with `ffmpeg -version`. If installed in a
non-standard location, add its directory to your `PATH`.

### cairosvg ImportError

Install with `pip install "cairosvg>=2.7"`. On Windows, `cairosvg` requires
the Cairo runtime DLL — use the GTK for Windows Runtime or convert SVG to PNG
outside Murmur and use `--image` instead.

### pyzbar ImportError or no QR codes found

Install with `pip install pyzbar`. On Linux: `sudo apt install libzbar0`.
If pyzbar is installed but no QR codes are found, the spectrogram resolution
is likely insufficient. Follow the QR scanning tips above.
