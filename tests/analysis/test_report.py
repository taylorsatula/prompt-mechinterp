"""Tests for report.py — _classify_story, compute_cooking_table, compute_context_bleed."""

import numpy as np
import pytest

from prompt_mechinterp.analysis.report import (
    _classify_story,
    compute_context_bleed,
    compute_cooking_table,
)


def _make_sample(region_map, per_layer_weights, position="terminal"):
    """Build a sample dict with per-token attention data."""
    per_layer = [
        {"layer": idx, "weights": w} for idx, w in per_layer_weights
    ]
    return {
        "region_map": region_map,
        "per_token_attention": {
            position: {"per_layer": per_layer}
        },
    }


# ===========================================================================
# _classify_story
# ===========================================================================

class TestClassifyStory:
    """Each branch of _classify_story represents a distinct processing pattern.

    The classification captures where in the forward pass a region peaks
    and how much attention decays after the peak. Classifications must
    scale with model depth — layer 28 of 32 should get a late-phase label,
    not the same label it would get in a 64-layer model.
    """

    # --- 64-layer model (default) ---

    def test_cooked_first_strong_fade(self):
        # Very early peak with heavy fade → model reads it immediately then forgets
        assert _classify_story(1, 15, 64) == "Cooked first, strong fade"

    def test_cooked_first(self):
        # Early peak with moderate fade
        assert _classify_story(1, 5, 64) == "Cooked first"

    def test_early_read(self):
        # Early peak but retention stays high → info persists
        assert _classify_story(2, 2, 64) == "Early read"

    def test_cooked_by_specific_layer(self):
        assert "L5" in _classify_story(5, 2, 64)

    def test_absorption_phase(self):
        assert _classify_story(10, 2, 64) == "Absorption phase"

    def test_deep_compression(self):
        assert _classify_story(20, 2, 64) == "Deep compression"

    def test_mid_phase_peak(self):
        assert _classify_story(40, 2, 64) == "Mid-phase peak"

    def test_output_prep_phase(self):
        assert _classify_story(50, 2, 64) == "Output prep phase"

    def test_latest_bloomer(self):
        # Late peak with good retention
        assert _classify_story(60, 1.5, 64) == "Latest bloomer"

    def test_late_bloomer(self):
        # Late peak with significant fade
        assert _classify_story(60, 3, 64) == "Late bloomer"

    def test_boundary_at_peak_2(self):
        """peak_layer=2 is still in the early range."""
        assert _classify_story(2, 15, 64) == "Cooked first, strong fade"

    def test_boundary_at_peak_3(self):
        """peak_layer=3 transitions to 'Cooked by L' range."""
        assert "L3" in _classify_story(3, 2, 64)

    def test_boundary_at_peak_6(self):
        """peak_layer=6 is the upper end of 'Cooked by L' range."""
        assert "L6" in _classify_story(6, 2, 64)

    def test_boundary_at_peak_7(self):
        """peak_layer=7 transitions to absorption."""
        assert _classify_story(7, 2, 64) == "Absorption phase"

    # --- Scaling to different model sizes ---
    # These are the tests that caught a real bug: the original implementation
    # used hardcoded layer thresholds that only worked for 64-layer models.

    def test_32_layer_late_peak_is_not_deep_compression(self):
        """Layer 28 of 32 (88% through) must NOT be 'Deep compression'."""
        story = _classify_story(28, 2.0, 32)
        assert story != "Deep compression", (
            f"Layer 28/32 (88% through model) classified as '{story}'. "
            "Should be a late-phase label."
        )

    def test_32_layer_last_layer_is_late_phase(self):
        """Last layer of a 32-layer model should be a bloomer/output category."""
        story = _classify_story(30, 1.5, 32)
        assert "bloomer" in story.lower() or "output" in story.lower(), (
            f"Layer 30/32 classified as '{story}', expected a late-phase label"
        )

    def test_32_layer_mid_peak(self):
        """Layer 16 of 32 (50% through) should be mid-phase, not 'Deep compression'."""
        story = _classify_story(16, 2.0, 32)
        assert "Mid" in story or "compression" in story.lower(), (
            f"Layer 16/32 classified as '{story}'"
        )

    def test_128_layer_proportional_classification(self):
        """Layer 112 of 128 (87.5%) should be output prep, not mid-phase."""
        story = _classify_story(112, 2.0, 128)
        assert "Output" in story or "bloomer" in story.lower(), (
            f"Layer 112/128 classified as '{story}', expected late-phase label"
        )


# ===========================================================================
# compute_cooking_table
# ===========================================================================

class TestComputeCookingTable:
    def test_computes_stats_for_each_region(self):
        region_map = {
            "rules": {"tok_start": 0, "tok_end": 2},
            "examples": {"tok_start": 2, "tok_end": 4},
        }
        # 4 layers, rules tokens have high attention early, examples late
        weights = [
            (0, [0.8, 0.8, 0.1, 0.1]),
            (1, [0.6, 0.6, 0.2, 0.2]),
            (2, [0.2, 0.2, 0.7, 0.7]),
            (3, [0.1, 0.1, 0.9, 0.9]),
        ]
        sample1 = _make_sample(region_map, weights)
        sample2 = _make_sample(region_map, weights)  # Duplicate for n_samples

        result = compute_cooking_table(
            [sample1, sample2], ["rules", "examples"], num_layers=4
        )

        assert "rules" in result
        assert "examples" in result
        # Rules peak early (layer 0 at 0.8), examples peak late (layer 3 at 0.9)
        assert result["rules"]["peak_layer"] == 0
        assert result["examples"]["peak_layer"] == 3
        assert result["rules"]["n_samples"] == 2

        # Rules: peak=0.8, terminal=0.1 → heavy fade → "Cooked first" family
        assert result["rules"]["peak_value"] == pytest.approx(0.8)
        assert result["rules"]["terminal_value"] == pytest.approx(0.1)
        assert "Cooked" in result["rules"]["story"]

        # Examples: peak at last layer → terminal equals peak → ratio=1
        assert result["examples"]["terminal_value"] == pytest.approx(0.9)
        assert result["examples"]["retention_ratio"] == pytest.approx(1.0)

    def test_all_zero_trajectory_excluded(self):
        region_map = {
            "active": {"tok_start": 0, "tok_end": 2},
            "silent": {"tok_start": 2, "tok_end": 4},
        }
        # "silent" region tokens always get zero attention
        weights = [
            (0, [0.5, 0.5, 0.0, 0.0]),
            (1, [0.3, 0.3, 0.0, 0.0]),
        ]
        sample = _make_sample(region_map, weights)

        result = compute_cooking_table(
            [sample], ["active", "silent"], num_layers=2
        )
        assert "active" in result
        assert "silent" not in result

    def test_returns_trajectory_list(self):
        region_map = {"rules": {"tok_start": 0, "tok_end": 2}}
        weights = [(0, [0.5, 0.5]), (1, [0.3, 0.3])]
        sample = _make_sample(region_map, weights)

        result = compute_cooking_table([sample], ["rules"], num_layers=2)
        assert "trajectory" in result["rules"]
        assert len(result["rules"]["trajectory"]) == 2

    def test_story_scales_with_num_layers(self):
        """A region peaking at the last layer should get a late-phase story
        regardless of how many layers the model has."""
        region_map = {"rules": {"tok_start": 0, "tok_end": 2}}
        # Peak at last layer (layer 3 of 4)
        weights = [
            (0, [0.1, 0.1]),
            (1, [0.2, 0.2]),
            (2, [0.3, 0.3]),
            (3, [0.9, 0.9]),
        ]
        sample = _make_sample(region_map, weights)

        result = compute_cooking_table([sample], ["rules"], num_layers=4)
        story = result["rules"]["story"]
        assert "compression" not in story.lower(), (
            f"Peak at last layer of 4-layer model got '{story}' — "
            "should be a late-phase label, not deep compression"
        )


# ===========================================================================
# compute_context_bleed
# ===========================================================================

class TestComputeContextBleed:
    def test_ratio_above_one(self):
        """When conv terminal > curr terminal, ratio should be > 1."""
        region_map = {
            "conversation_turns": {"tok_start": 0, "tok_end": 3},
            "current_message": {"tok_start": 3, "tok_end": 6},
        }
        # Terminal layer (layer 3): conv tokens=0.6, curr tokens=0.2
        weights = [
            (0, [0.1] * 6),
            (1, [0.1] * 6),
            (2, [0.1] * 6),
            (3, [0.6, 0.6, 0.6, 0.2, 0.2, 0.2]),
        ]
        sample = _make_sample(region_map, weights)

        result = compute_context_bleed([sample], num_layers=4)
        # conv terminal mean=0.6, curr terminal mean=0.2 → ratio=3.0
        assert result["mean_ratio"] == pytest.approx(3.0)
        assert result["n_samples"] == 1
        # Individual means should also be correct
        assert result["conv_turns_mean"] == pytest.approx(0.6)
        assert result["current_message_mean"] == pytest.approx(0.2)

    def test_curr_terminal_zero_excluded(self):
        """Samples where current_message terminal is 0 should be excluded from ratios."""
        region_map = {
            "conversation_turns": {"tok_start": 0, "tok_end": 2},
            "current_message": {"tok_start": 2, "tok_end": 4},
        }
        weights = [(0, [0.5, 0.5, 0.0, 0.0])]
        sample = _make_sample(region_map, weights)

        result = compute_context_bleed([sample], num_layers=1)
        assert result["n_samples"] == 0

    def test_pct_above_2x(self):
        region_map = {
            "conversation_turns": {"tok_start": 0, "tok_end": 2},
            "current_message": {"tok_start": 2, "tok_end": 4},
        }
        # Sample 1: ratio = 3x (above 2x)
        s1 = _make_sample(region_map, [(0, [0.6, 0.6, 0.2, 0.2])])
        # Sample 2: ratio = 1x (below 2x)
        s2 = _make_sample(region_map, [(0, [0.4, 0.4, 0.4, 0.4])])

        result = compute_context_bleed([s1, s2], num_layers=1)
        # 1 of 2 samples above 2x → 50%
        assert result["pct_above_2x"] == pytest.approx(50.0)
