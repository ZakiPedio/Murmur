"""Tests for murmur.presets.get_preset() and list_presets()."""

from __future__ import annotations

import pytest

from murmur.presets import PRESETS, get_preset, list_presets


# ---------------------------------------------------------------------------
# 1. All preset names are valid
# ---------------------------------------------------------------------------

class TestGetPreset:
    @pytest.mark.parametrize("name", list(PRESETS.keys()))
    def test_all_presets_retrievable(self, name):
        result = get_preset(name)
        assert isinstance(result, dict)

    @pytest.mark.parametrize("name", list(PRESETS.keys()))
    def test_preset_is_a_copy(self, name):
        """Mutations to the returned dict must not affect the canonical PRESETS."""
        result = get_preset(name)
        result["__test__"] = True
        assert "__test__" not in PRESETS[name]

    def test_case_insensitive(self):
        """Preset names should be case-insensitive."""
        result_lower = get_preset("aphex")
        result_upper = get_preset("APHEX")
        assert result_lower == result_upper


# ---------------------------------------------------------------------------
# 2. Each preset has 'description' key
# ---------------------------------------------------------------------------

class TestPresetDescription:
    @pytest.mark.parametrize("name", list(PRESETS.keys()))
    def test_has_description(self, name):
        result = get_preset(name)
        assert "description" in result, f"Preset '{name}' missing 'description' key"
        assert isinstance(result["description"], str)
        assert len(result["description"]) > 0


# ---------------------------------------------------------------------------
# 3. freq_min < freq_max where both are present
# ---------------------------------------------------------------------------

class TestPresetFrequencyRange:
    @pytest.mark.parametrize("name", list(PRESETS.keys()))
    def test_freq_min_lt_freq_max(self, name):
        result = get_preset(name)
        if "freq_min" in result and "freq_max" in result:
            assert result["freq_min"] < result["freq_max"], (
                f"Preset '{name}': freq_min ({result['freq_min']}) "
                f">= freq_max ({result['freq_max']})"
            )


# ---------------------------------------------------------------------------
# 4. Unknown preset raises ValueError
# ---------------------------------------------------------------------------

class TestUnknownPreset:
    def test_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown preset"):
            get_preset("totally_nonexistent_preset_xyz")

    def test_error_message_includes_name(self):
        with pytest.raises(ValueError, match="bad_name"):
            get_preset("bad_name")


# ---------------------------------------------------------------------------
# 5. list_presets() returns list of tuples (name, description)
# ---------------------------------------------------------------------------

class TestListPresets:
    def test_returns_list(self):
        result = list_presets()
        assert isinstance(result, list)

    def test_each_item_is_tuple_of_two_strings(self):
        result = list_presets()
        for item in result:
            assert isinstance(item, tuple), f"Expected tuple, got {type(item)}"
            assert len(item) == 2, f"Expected 2-tuple, got length {len(item)}"
            name, desc = item
            assert isinstance(name, str)
            assert isinstance(desc, str)

    def test_includes_all_preset_names(self):
        result = list_presets()
        returned_names = {name for name, _ in result}
        for name in PRESETS:
            assert name in returned_names, f"Preset '{name}' missing from list_presets()"

    def test_descriptions_match(self):
        result = list_presets()
        for name, desc in result:
            assert desc == PRESETS[name]["description"], (
                f"Description mismatch for preset '{name}'"
            )

    def test_known_presets_present(self):
        result = list_presets()
        names = [name for name, _ in result]
        for expected in ("aphex", "cicada", "stealth"):
            assert expected in names, f"Expected preset '{expected}' in list_presets()"
