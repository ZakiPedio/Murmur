"""Integration tests for the Murmur CLI (python -m murmur)."""

from __future__ import annotations

import subprocess
import sys
import wave
from pathlib import Path

import pytest
from PIL import Image


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _run(*args, cwd=None):
    """Run `python -m murmur <args>` and return the CompletedProcess."""
    cmd = [sys.executable, "-m", "murmur"] + list(args)
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(cwd) if cwd else None,
    )


def _is_valid_wav(path: Path) -> bool:
    """Return True if path points to a valid WAV file."""
    try:
        with wave.open(str(path), "rb") as wf:
            return wf.getnframes() > 0
    except Exception:
        return False


def _is_valid_png(path: Path) -> bool:
    """Return True if path points to a valid PNG file."""
    try:
        img = Image.open(str(path))
        img.verify()
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# 1. encode --text → valid WAV
# ---------------------------------------------------------------------------

class TestEncodeText:
    @pytest.mark.slow
    def test_encode_text_produces_wav(self, tmp_path):
        out = tmp_path / "hello.wav"
        result = _run(
            "encode", "--text", "HELLO",
            "--duration", "1.0",
            "--output", str(out),
        )
        assert result.returncode == 0, (
            f"encode --text failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )
        assert out.exists(), "Output WAV file was not created"
        assert _is_valid_wav(out), "Output is not a valid WAV file"


# ---------------------------------------------------------------------------
# 2. encode --qr → valid WAV
# ---------------------------------------------------------------------------

class TestEncodeQr:
    @pytest.mark.slow
    def test_encode_qr_produces_wav(self, tmp_path):
        out = tmp_path / "qr.wav"
        result = _run(
            "encode", "--qr", "https://example.com",
            "--duration", "1.0",
            "--output", str(out),
        )
        assert result.returncode == 0, (
            f"encode --qr failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )
        assert out.exists()
        assert _is_valid_wav(out)


# ---------------------------------------------------------------------------
# 3. encode --text --preset aphex → valid WAV
# ---------------------------------------------------------------------------

class TestEncodeWithPreset:
    @pytest.mark.slow
    def test_encode_text_with_aphex_preset(self, tmp_path):
        out = tmp_path / "aphex.wav"
        result = _run(
            "encode", "--text", "TEST",
            "--preset", "aphex",
            "--duration", "1.0",
            "--output", str(out),
        )
        assert result.returncode == 0, (
            f"encode --preset aphex failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )
        assert out.exists()
        assert _is_valid_wav(out)


# ---------------------------------------------------------------------------
# 4. verify --input → valid PNG
# ---------------------------------------------------------------------------

class TestVerify:
    @pytest.mark.slow
    def test_verify_produces_png(self, tmp_path):
        # First create a WAV to verify
        wav = tmp_path / "input.wav"
        _run(
            "encode", "--text", "X",
            "--duration", "1.0",
            "--output", str(wav),
        )
        assert wav.exists(), "Setup WAV not created; cannot test verify"

        spec = tmp_path / "spec.png"
        result = _run(
            "verify",
            "--input", str(wav),
            "--output", str(spec),
        )
        assert result.returncode == 0, (
            f"verify failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )
        assert spec.exists(), "Spectrogram PNG was not created"
        assert _is_valid_png(spec), "Output is not a valid PNG"


# ---------------------------------------------------------------------------
# 5. presets → stdout contains known preset names
# ---------------------------------------------------------------------------

class TestPresetsCommand:
    def test_presets_lists_known_names(self):
        result = _run("presets")
        assert result.returncode == 0, (
            f"presets command failed:\n{result.stderr}"
        )
        output = result.stdout
        for name in ("aphex", "cicada", "stealth"):
            assert name in output, (
                f"Preset '{name}' not found in presets output:\n{output}"
            )


# ---------------------------------------------------------------------------
# 6. encode with no input → exit code 1
# ---------------------------------------------------------------------------

class TestEncodeNoInput:
    def test_encode_without_input_exits_1(self, tmp_path):
        out = tmp_path / "out.wav"
        result = _run("encode", "--output", str(out))
        assert result.returncode != 0, (
            "encode without any input source should fail"
        )


# ---------------------------------------------------------------------------
# 7. encode with invalid freq range → exit code 1
# ---------------------------------------------------------------------------

class TestInvalidFreqRange:
    def test_freq_min_gt_freq_max_exits_1(self, tmp_path):
        out = tmp_path / "bad.wav"
        result = _run(
            "encode", "--text", "X",
            "--freq-min", "1000",
            "--freq-max", "500",
            "--output", str(out),
        )
        assert result.returncode == 1, (
            f"Expected exit code 1 for invalid freq range, got {result.returncode}\n"
            f"stderr: {result.stderr}"
        )


# ---------------------------------------------------------------------------
# 8. encode with mutually exclusive --image and --text → exit code non-zero
# ---------------------------------------------------------------------------

class TestMutuallyExclusiveInputs:
    def test_image_and_text_together_exits_nonzero(self, tmp_path):
        # Create a dummy image file so argparse gets past file validation
        dummy_img = tmp_path / "dummy.png"
        from PIL import Image as _Image
        _Image.new("L", (4, 4), 128).save(str(dummy_img))

        out = tmp_path / "out.wav"
        result = _run(
            "encode",
            "--image", str(dummy_img),
            "--text", "HELLO",
            "--output", str(out),
        )
        assert result.returncode != 0, (
            "Providing both --image and --text should fail (mutually exclusive)"
        )


# ---------------------------------------------------------------------------
# 9. overlay command integration tests
# ---------------------------------------------------------------------------

class TestOverlayCommand:
    @pytest.mark.slow
    def test_overlay_produces_valid_wav(self, tmp_path):
        """overlay embeds a text image into an existing carrier WAV."""
        carrier = tmp_path / "carrier.wav"
        output = tmp_path / "mixed.wav"

        # Create carrier
        r = _run("encode", "--text", "CARRIER", "--duration", "2.0", "--output", str(carrier))
        assert r.returncode == 0, f"carrier encode failed: {r.stderr}"
        assert carrier.exists()

        # Overlay another image
        r2 = _run(
            "overlay",
            "--text", "HIDDEN",
            "--carrier", str(carrier),
            "--blend", "0.3",
            "--output", str(output),
        )
        assert r2.returncode == 0, f"overlay failed:\nstdout: {r2.stdout}\nstderr: {r2.stderr}"
        assert output.exists(), "Overlay output WAV was not created"
        assert _is_valid_wav(output), "Overlay output is not a valid WAV"

    @pytest.mark.slow
    def test_overlay_output_same_length_as_carrier(self, tmp_path):
        """Overlay does not change the duration of the carrier."""
        import wave as _wave

        carrier = tmp_path / "carrier.wav"
        output = tmp_path / "mixed.wav"

        _run("encode", "--text", "C", "--duration", "2.0", "--output", str(carrier))

        _run(
            "overlay",
            "--text", "H",
            "--carrier", str(carrier),
            "--output", str(output),
        )

        with _wave.open(str(carrier), "rb") as wf:
            carrier_frames = wf.getnframes()
        with _wave.open(str(output), "rb") as wf:
            output_frames = wf.getnframes()

        assert carrier_frames == output_frames, (
            f"Overlay changed WAV length: carrier={carrier_frames}, output={output_frames}"
        )

    @pytest.mark.slow
    def test_overlay_missing_carrier_exits_1(self, tmp_path):
        """overlay exits with code 1 if carrier file does not exist."""
        result = _run(
            "overlay",
            "--text", "X",
            "--carrier", str(tmp_path / "nonexistent.wav"),
            "--output", str(tmp_path / "out.wav"),
        )
        assert result.returncode == 1, "Expected exit code 1 for missing carrier"

    @pytest.mark.slow
    def test_overlay_blend_out_of_range_exits_1(self, tmp_path):
        """overlay exits with code 1 if --blend > 1.0."""
        carrier = tmp_path / "carrier.wav"
        _run("encode", "--text", "C", "--duration", "1.0", "--output", str(carrier))

        result = _run(
            "overlay",
            "--text", "X",
            "--carrier", str(carrier),
            "--blend", "2.0",
            "--output", str(tmp_path / "out.wav"),
        )
        assert result.returncode == 1, "Expected exit code 1 for blend > 1.0"

    @pytest.mark.slow
    def test_overlay_headroom_db_is_respected(self, tmp_path):
        """--headroom-db is passed through to the mixer and affects output peak."""
        import wave as _wave
        import struct as _struct

        carrier = tmp_path / "carrier.wav"
        out_loud = tmp_path / "loud.wav"
        out_quiet = tmp_path / "quiet.wav"

        _run("encode", "--text", "C", "--duration", "2.0", "--output", str(carrier))

        _run("overlay", "--text", "H", "--carrier", str(carrier),
             "--headroom-db", "-0.5", "--output", str(out_loud))
        _run("overlay", "--text", "H", "--carrier", str(carrier),
             "--headroom-db", "-20.0", "--output", str(out_quiet))

        def _peak(path):
            with _wave.open(str(path), "rb") as wf:
                raw = wf.readframes(wf.getnframes())
            samples = [_struct.unpack_from("<h", raw, i)[0] for i in range(0, len(raw), 2)]
            return max(abs(s) for s in samples)

        assert _peak(out_loud) > _peak(out_quiet), (
            "--headroom-db -0.5 should produce louder output than -20.0"
        )


# ---------------------------------------------------------------------------
# 10. decode command integration tests
# ---------------------------------------------------------------------------

class TestDecodeCommand:
    @pytest.mark.slow
    def test_decode_produces_spectrogram_image(self, tmp_path):
        """decode saves a spectrogram PNG even without pyzbar."""
        wav = tmp_path / "input.wav"
        out_img = tmp_path / "decoded.png"

        _run("encode", "--text", "X", "--duration", "1.0", "--output", str(wav))
        assert wav.exists()

        result = _run(
            "decode",
            "--input", str(wav),
            "--output", str(out_img),
        )
        assert result.returncode == 0, (
            f"decode failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )
        assert out_img.exists(), "Decoded spectrogram image was not created"
        assert _is_valid_png(out_img), "Decoded output is not a valid PNG"

    def test_decode_missing_input_exits_1(self, tmp_path):
        """decode exits with code 1 if input file does not exist."""
        result = _run(
            "decode",
            "--input", str(tmp_path / "nonexistent.wav"),
        )
        assert result.returncode == 1, "Expected exit code 1 for missing input"


# ---------------------------------------------------------------------------
# 11. encode --dither options
# ---------------------------------------------------------------------------

class TestEncodeDitherOptions:
    @pytest.mark.slow
    @pytest.mark.parametrize("method", ["none", "threshold", "ordered", "floyd-steinberg"])
    def test_dither_method_produces_valid_wav(self, tmp_path, method):
        out = tmp_path / f"dither_{method}.wav"
        result = _run(
            "encode", "--text", "HI",
            "--duration", "1.0",
            "--dither", method,
            "--output", str(out),
        )
        assert result.returncode == 0, (
            f"encode --dither {method} failed:\n{result.stderr}"
        )
        assert _is_valid_wav(out)


# ---------------------------------------------------------------------------
# 12. New encoding flags
# ---------------------------------------------------------------------------

class TestNewEncodingFlags:
    @pytest.mark.slow
    def test_randomize_phase_produces_wav(self, tmp_path):
        out = tmp_path / "rand_phase.wav"
        result = _run(
            "encode", "--text", "HI",
            "--duration", "1.0",
            "--randomize-phase",
            "--output", str(out),
        )
        assert result.returncode == 0, f"--randomize-phase failed:\n{result.stderr}"
        assert _is_valid_wav(out)

    @pytest.mark.slow
    def test_auto_contrast_produces_wav(self, tmp_path):
        out = tmp_path / "autocontrast.wav"
        result = _run(
            "encode", "--text", "HI",
            "--duration", "1.0",
            "--auto-contrast",
            "--output", str(out),
        )
        assert result.returncode == 0, f"--auto-contrast failed:\n{result.stderr}"
        assert _is_valid_wav(out)

    @pytest.mark.slow
    def test_mp3_safe_clamps_range(self, tmp_path):
        """--mp3-safe should clamp an out-of-range freq-max and still succeed."""
        out = tmp_path / "mp3safe.wav"
        result = _run(
            "encode", "--text", "HI",
            "--duration", "1.0",
            "--mp3-safe",
            "--freq-max", "20000",  # above MP3-safe ceiling; should be clamped
            "--output", str(out),
        )
        assert result.returncode == 0, f"--mp3-safe failed:\n{result.stderr}"
        assert _is_valid_wav(out)
        # Warning about clamping should appear in stderr
        assert "clamping" in result.stderr.lower(), (
            f"Expected clamping warning in stderr, got:\n{result.stderr}"
        )

    @pytest.mark.slow
    def test_verify_after_encode_produces_spec_png(self, tmp_path):
        out = tmp_path / "verifyme.wav"
        spec = tmp_path / "verifyme.spec.png"
        result = _run(
            "encode", "--text", "HI",
            "--duration", "1.0",
            "--verify-after-encode",
            "--output", str(out),
        )
        assert result.returncode == 0, f"--verify-after-encode failed:\n{result.stderr}"
        assert out.exists(), "WAV file was not created"
        assert spec.exists(), ".spec.png was not created by --verify-after-encode"
        assert _is_valid_png(spec), ".spec.png is not a valid PNG"


# ---------------------------------------------------------------------------
# 13. probe subcommand
# ---------------------------------------------------------------------------

class TestProbeCommand:
    @pytest.mark.slow
    def test_probe_exits_0(self, tmp_path):
        """probe runs successfully on a valid WAV."""
        wav = tmp_path / "probe_input.wav"
        _run("encode", "--text", "X", "--duration", "2.0", "--output", str(wav))
        assert wav.exists()

        result = _run("probe", "--input", str(wav))
        assert result.returncode == 0, f"probe failed:\n{result.stderr}"

    @pytest.mark.slow
    def test_probe_json_output(self, tmp_path):
        """probe --json produces valid JSON with expected keys."""
        import json as _json

        wav = tmp_path / "probe_json.wav"
        _run("encode", "--text", "X", "--duration", "2.0", "--output", str(wav))

        result = _run("probe", "--input", str(wav), "--json")
        assert result.returncode == 0, f"probe --json failed:\n{result.stderr}"

        data = _json.loads(result.stdout)
        assert "octave_bands" in data, "probe JSON missing 'octave_bands'"
        assert "recommendations" in data, "probe JSON missing 'recommendations'"
        assert "sample_rate" in data, "probe JSON missing 'sample_rate'"

    @pytest.mark.slow
    def test_probe_stdout_contains_recommendations(self, tmp_path):
        """probe human-readable output mentions recommendations."""
        wav = tmp_path / "probe_hr.wav"
        _run("encode", "--text", "X", "--duration", "2.0", "--output", str(wav))

        result = _run("probe", "--input", str(wav))
        assert "Recommendations" in result.stdout or "Probe" in result.stdout, (
            f"probe output missing expected headings:\n{result.stdout}"
        )


# ---------------------------------------------------------------------------
# 14. --seed flag for --randomize-phase
# ---------------------------------------------------------------------------

class TestSeedFlag:
    @pytest.mark.slow
    def test_seed_makes_output_reproducible(self, tmp_path):
        """Two encodes with the same --seed produce bit-identical WAVs."""
        import hashlib

        out1 = tmp_path / "seed_a.wav"
        out2 = tmp_path / "seed_b.wav"

        for out in (out1, out2):
            r = _run(
                "encode", "--text", "HI",
                "--duration", "1.0",
                "--randomize-phase",
                "--seed", "123",
                "--output", str(out),
            )
            assert r.returncode == 0, f"encode with --seed failed:\n{r.stderr}"

        h1 = hashlib.md5(out1.read_bytes()).hexdigest()
        h2 = hashlib.md5(out2.read_bytes()).hexdigest()
        assert h1 == h2, "Same --seed should produce identical WAV files"

    @pytest.mark.slow
    def test_different_seeds_produce_different_output(self, tmp_path):
        """Two encodes with different --seed values are not identical."""
        import hashlib

        out1 = tmp_path / "seed1.wav"
        out2 = tmp_path / "seed2.wav"

        for seed, out in (("1", out1), ("2", out2)):
            _run(
                "encode", "--text", "HI",
                "--duration", "1.0",
                "--randomize-phase",
                "--seed", seed,
                "--output", str(out),
            )

        h1 = hashlib.md5(out1.read_bytes()).hexdigest()
        h2 = hashlib.md5(out2.read_bytes()).hexdigest()
        assert h1 != h2, "Different seeds should produce different WAV files"


# ---------------------------------------------------------------------------
# 15. Pre-flight ffmpeg check
# ---------------------------------------------------------------------------

class TestPreflightFfmpeg:
    def test_nonwav_output_without_ffmpeg_exits_1(self, tmp_path, monkeypatch):
        """encode with a non-WAV output should exit 1 when ffmpeg is absent."""
        import shutil as _shutil

        # Only run this test if ffmpeg is actually absent on this system
        if _shutil.which("ffmpeg") is not None:
            pytest.skip("ffmpeg is installed; cannot test pre-flight failure")

        out = tmp_path / "out.mp3"
        result = _run(
            "encode", "--text", "HI",
            "--duration", "1.0",
            "--output", str(out),
        )
        assert result.returncode == 1, (
            "Expected exit 1 when ffmpeg is absent and output is non-WAV"
        )
        assert "ffmpeg" in result.stderr.lower(), (
            f"Expected ffmpeg mention in stderr, got:\n{result.stderr}"
        )
