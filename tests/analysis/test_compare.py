"""Tests for compare.py — seed detection, region discovery, layer detection, table display."""

import io
import sys

from prompt_mechinterp.analysis.compare import (
    _auto_discover_regions,
    _detect_num_layers,
    _detect_seed,
    table_terminal_attention,
)
from prompt_mechinterp.constants import SKIP_REGIONS


class TestDetectSeed:
    def test_single_letter_suffix(self):
        assert _detect_seed("sample_01_a") == "seed_a"

    def test_another_letter_suffix(self):
        assert _detect_seed("sample_01_b") == "seed_b"

    def test_multi_char_final_segment(self):
        """Only a single-letter final segment counts as a seed identifier."""
        assert _detect_seed("sample_01") == "seed_default"

    def test_multi_char_suffix(self):
        """Multi-character suffix should not be treated as a seed."""
        assert _detect_seed("sample_01_ab") == "seed_default"

    def test_numeric_suffix(self):
        """A single digit should not be detected (isalpha check)."""
        assert _detect_seed("sample_01_1") == "seed_default"

    def test_deeply_nested_id(self):
        """Seed letter should always be the final character after the last underscore."""
        assert _detect_seed("exp_v3_sample_01_c") == "seed_c"


class TestAutoDiscoverRegions:
    def test_filters_skip_regions(self):
        sample = {
            "region_map": {
                "system_prompt": {},
                "user_message": {},
                "directive": {},
                "entity_rules": {},
                "chat_template": {},
            }
        }
        result = _auto_discover_regions([sample])
        assert "directive" in result
        assert "entity_rules" in result
        for skip in SKIP_REGIONS:
            assert skip not in result

    def test_returns_sorted(self):
        sample = {
            "region_map": {
                "zebra_region": {},
                "alpha_region": {},
            }
        }
        result = _auto_discover_regions([sample])
        assert result == ["alpha_region", "zebra_region"]

    def test_empty_samples(self):
        assert _auto_discover_regions([]) == []

    def test_sample_without_region_map(self):
        result = _auto_discover_regions([{}])
        assert result == []


class TestDetectNumLayers:
    def test_detects_from_attention_data(self):
        sample = {
            "attention": {
                "terminal": {
                    "per_layer": [
                        {"layer": i} for i in range(32)
                    ]
                }
            }
        }
        assert _detect_num_layers([sample]) == 32

    def test_fallback_to_64_without_data(self):
        assert _detect_num_layers([{}]) == 64

    def test_fallback_to_64_empty_list(self):
        assert _detect_num_layers([]) == 64

    def test_uses_max_layer_index(self):
        """Should use the maximum layer index + 1, not count of entries."""
        sample = {
            "attention": {
                "terminal": {
                    "per_layer": [
                        {"layer": 0},
                        {"layer": 15},
                        {"layer": 63},
                    ]
                }
            }
        }
        assert _detect_num_layers([sample]) == 64


class TestTableTerminalAttention:
    """The table header should reflect the actual model's layer range, not hardcoded 64."""

    def _make_32_layer_sample(self):
        """Build a sample with 32-layer attention data."""
        per_layer = [
            {"per_region_mean": {"rules": 0.1 + i * 0.01}}
            for i in range(32)
        ]
        return {
            "attention": {"terminal": {"per_layer": per_layer}},
            "region_map": {"rules": {}},
        }

    def test_header_reflects_actual_layer_count(self):
        """For a 32-layer model, the header should NOT say 'L60-63'.

        The computation correctly uses the last N layers, but the display
        hardcodes '64' and '63' — misleading for any non-64-layer model.
        """
        samples = [self._make_32_layer_sample()]
        variants = {"Test": samples}

        captured = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        try:
            table_terminal_attention(variants, ["rules"], "terminal", num_layers=32)
        finally:
            sys.stdout = old_stdout

        output = captured.getvalue()
        # The header should NOT contain "L60-63" for a 32-layer model
        assert "L60" not in output, (
            f"Header says 'L60-63' for a 32-layer model. "
            f"Should reflect actual layer range (L28-31).\n"
            f"Output: {output}"
        )
