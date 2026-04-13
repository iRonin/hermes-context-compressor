"""Unit tests for ironin-compressor config parsing (no Hermes dependency)."""

import pytest
from hermes_context_compressor.config import parse_config, DEFAULTS


class TestParseConfigDefaults:
    def test_all_defaults_present(self):
        for key in ["keep_threshold", "drop_threshold", "max_retries_window",
                     "summarize_low", "preserve_tool_integrity", "skip_scoring_under_msgs"]:
            assert key in DEFAULTS

    def test_sensible_defaults(self):
        assert DEFAULTS["keep_threshold"] == 7
        assert DEFAULTS["drop_threshold"] == 4
        assert DEFAULTS["max_retries_window"] == 3
        assert DEFAULTS["summarize_low"] is False
        assert DEFAULTS["skip_scoring_under_msgs"] == 6


class TestParseConfigOverrides:
    def test_single_override(self):
        result = parse_config({"keep_threshold": 8})
        assert result["keep_threshold"] == 8
        assert result["drop_threshold"] == 4  # unchanged

    def test_multiple_overrides(self):
        result = parse_config({"keep_threshold": 9, "drop_threshold": 5, "max_retries_window": 5})
        assert result["keep_threshold"] == 9
        assert result["drop_threshold"] == 5
        assert result["max_retries_window"] == 5


class TestParseConfigClamping:
    @pytest.mark.parametrize("value,expected", [
        (15, 10),  # above max
        (1, 4),    # below min
        (7, 7),    # in range
        (4, 4),    # at min
        (10, 10),  # at max
    ])
    def test_keep_threshold_clamp(self, value, expected):
        result = parse_config({"keep_threshold": value})
        assert result["keep_threshold"] == expected

    @pytest.mark.parametrize("value,expected", [
        (10, 6),   # above max
        (-5, 0),   # below min
        (3, 3),    # in range
        (0, 0),    # at min
        (6, 6),    # at max
    ])
    def test_drop_threshold_clamp(self, value, expected):
        result = parse_config({"drop_threshold": value})
        assert result["drop_threshold"] == expected


class TestParseConfigTypeHandling:
    def test_string_number_conversion(self):
        result = parse_config({"keep_threshold": "8"})
        assert result["keep_threshold"] == 8
        assert isinstance(result["keep_threshold"], int)

    def test_invalid_string_keeps_default(self):
        result = parse_config({"keep_threshold": "abc"})
        assert result["keep_threshold"] == DEFAULTS["keep_threshold"]

    def test_bool_handling(self):
        assert parse_config({"summarize_low": True})["summarize_low"] is True
        assert parse_config({"summarize_low": False})["summarize_low"] is False
        assert parse_config({"summarize_low": 1})["summarize_low"] is True  # int->bool

    def test_unknown_key_ignored(self):
        result = parse_config({"unknown_key": 42})
        assert result == DEFAULTS


class TestParseConfigInvariants:
    def test_drop_always_less_than_keep(self):
        result = parse_config({"keep_threshold": 5, "drop_threshold": 5})
        assert result["drop_threshold"] < result["keep_threshold"]

    def test_drop_adjusted_when_equal_at_clamp(self):
        result = parse_config({"keep_threshold": 4, "drop_threshold": 6})
        assert result["keep_threshold"] == 4
        assert result["drop_threshold"] == 3
