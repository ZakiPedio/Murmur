"""Named parameter presets for common Murmur use cases."""

from __future__ import annotations

from typing import Any

# Each preset is a dict of CLI argument overrides.
# Keys match the long-form argument names (with underscores, not dashes).
PRESETS: dict[str, dict[str, Any]] = {
    "aphex": {
        "description": "Classic Aphex Twin style — face visible in full audible range",
        "freq_min": 200,
        "freq_max": 12000,
        "duration": 8.0,
        "dither": "none",
        "log_frequency": True,
    },
    "cicada": {
        "description": "Cicada 3301 puzzle style — QR code in stealth frequency band",
        "freq_min": 14000,
        "freq_max": 20000,
        "duration": 15.0,
        "dither": "threshold",
        "blend": 0.15,
    },
    "stealth": {
        "description": "Near-inaudible — high frequency, low amplitude, long duration",
        "freq_min": 16000,
        "freq_max": 20000,
        "duration": 30.0,
        "blend": 0.05,
        "dither": "none",
    },
    "loud": {
        "description": "Maximum visibility — full range, high contrast, short duration",
        "freq_min": 100,
        "freq_max": 16000,
        "duration": 5.0,
        "dither": "threshold",
        "log_frequency": False,
    },
    "watermark": {
        "description": "Forensic watermark — extremely subtle, spread across full duration",
        "freq_min": 8000,
        "freq_max": 19000,
        "blend": 0.03,
        "dither": "none",
    },
    "musical": {
        "description": "Embed in mid-range — blends with typical music content",
        "freq_min": 500,
        "freq_max": 4000,
        "duration": 10.0,
        "blend": 0.2,
        "dither": "floyd-steinberg",
    },
}

_REQUIRED_RANGE_KEYS = {"freq_min", "freq_max"}
_VALID_DITHER_METHODS = {"floyd-steinberg", "ordered", "threshold", "none"}


def get_preset(name: str) -> dict[str, Any]:
    """Retrieve a preset by name.

    Args:
        name: Preset name (case-insensitive).

    Returns:
        Dict of parameter overrides for the preset.

    Raises:
        ValueError: If the preset name is not recognized.
    """
    key = name.lower()
    if key not in PRESETS:
        available = ", ".join(sorted(PRESETS))
        raise ValueError(f"Unknown preset: {name!r}. Available presets: {available}")
    return PRESETS[key].copy()


def list_presets() -> list[tuple[str, str]]:
    """Return a list of (name, description) tuples for all presets.

    Returns:
        Sorted list of (preset_name, description) tuples.
    """
    return [(name, info["description"]) for name, info in sorted(PRESETS.items())]
