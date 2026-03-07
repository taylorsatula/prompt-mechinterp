"""Tests for analysis/metrics.py — the core math layer."""

from math import isnan, nan

import numpy as np
import pytest

from prompt_mechinterp.analysis.metrics import (
    avg_final_layers,
    compute_per_token_density,
    compute_region_attention_per_layer,
    compute_region_ratio,
    cooking_curve_stats,
    phase_mean,
    safe_mean,
    safe_median,
    safe_std,
)


# ---------------------------------------------------------------------------
# Helpers to build realistic data structures
# ---------------------------------------------------------------------------

def _make_attention_data(num_layers, regions_by_layer):
    """Build an attention_data dict matching engine output format.

    Args:
        num_layers: number of layers
        regions_by_layer: dict of {layer_idx: {region_name: value}}
    """
    per_layer = []
    for i in range(num_layers):
        prm = regions_by_layer.get(i, {})
        per_layer.append({"per_region_mean": prm})
    return {"terminal": {"per_layer": per_layer}}


def _make_sample(region_map, per_layer_weights, position="terminal"):
    """Build a sample dict with per-token attention data.

    Args:
        region_map: dict of {name: {tok_start, tok_end}}
        per_layer_weights: list of (layer_idx, weights_list) tuples
    """
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
# avg_final_layers
# ===========================================================================

class TestAvgFinalLayers:
    def test_averages_last_n_layers(self):
        # 8 layers, region "rules" present in all layers
        regions_by_layer = {i: {"rules": 0.1 * (i + 1)} for i in range(8)}
        attn = _make_attention_data(8, regions_by_layer)

        # Default n_layers=4 → average layers 4,5,6,7 → 0.5, 0.6, 0.7, 0.8
        result = avg_final_layers(attn, "terminal", "rules", n_layers=4)
        assert result == pytest.approx(0.65)

    def test_missing_position_returns_nan(self):
        attn = _make_attention_data(8, {})
        result = avg_final_layers(attn, "nonexistent", "rules")
        assert isnan(result)

    def test_region_present_in_some_layers_only(self):
        # Region only in layers 6 and 7 (last 2 of 4 final layers)
        regions_by_layer = {6: {"rules": 0.4}, 7: {"rules": 0.8}}
        attn = _make_attention_data(8, regions_by_layer)

        result = avg_final_layers(attn, "terminal", "rules", n_layers=4)
        # Only layers 6,7 have data → average (0.4 + 0.8) / 2
        assert result == pytest.approx(0.6)

    def test_region_not_found_in_any_final_layer(self):
        regions_by_layer = {0: {"rules": 1.0}}  # Only in layer 0
        attn = _make_attention_data(8, regions_by_layer)

        result = avg_final_layers(attn, "terminal", "rules", n_layers=4)
        assert isnan(result)

    def test_n_layers_larger_than_total(self):
        regions_by_layer = {0: {"rules": 0.2}, 1: {"rules": 0.4}}
        attn = _make_attention_data(2, regions_by_layer)

        # Request last 4 layers of a 2-layer model → uses all layers
        result = avg_final_layers(attn, "terminal", "rules", n_layers=4)
        assert result == pytest.approx(0.3)


# ===========================================================================
# compute_region_ratio
# ===========================================================================

class TestComputeRegionRatio:
    def test_normal_ratio(self):
        # Two samples, both with known region values
        regions_by_layer = {
            6: {"conv": 0.4, "curr": 0.2},
            7: {"conv": 0.4, "curr": 0.2},
        }
        attn = _make_attention_data(8, regions_by_layer)
        samples = [{"attention": attn}]

        ratios = compute_region_ratio(samples, "conv", "curr", n_layers=2)
        assert len(ratios) == 1
        assert ratios[0] == pytest.approx(2.0)

    def test_region_b_zero_excluded(self):
        """When region_b attention is zero, sample should be excluded (no ZeroDivisionError)."""
        regions_by_layer = {7: {"conv": 0.5, "curr": 0.0}}
        attn = _make_attention_data(8, regions_by_layer)
        samples = [{"attention": attn}]

        ratios = compute_region_ratio(samples, "conv", "curr", n_layers=1)
        assert ratios == []

    def test_nan_samples_excluded(self):
        # Sample 1: valid, Sample 2: missing region_b entirely
        attn_valid = _make_attention_data(8, {7: {"conv": 0.6, "curr": 0.3}})
        attn_missing = _make_attention_data(8, {7: {"conv": 0.5}})

        samples = [{"attention": attn_valid}, {"attention": attn_missing}]
        ratios = compute_region_ratio(samples, "conv", "curr", n_layers=1)
        # Only first sample should produce a ratio
        assert len(ratios) == 1
        assert ratios[0] == pytest.approx(2.0)


# ===========================================================================
# compute_per_token_density
# ===========================================================================

class TestComputePerTokenDensity:
    def test_normal(self):
        assert compute_per_token_density(0.5, 10) == pytest.approx(0.05)

    def test_zero_tokens_returns_nan(self):
        assert isnan(compute_per_token_density(0.5, 0))

    def test_negative_tokens_returns_nan(self):
        assert isnan(compute_per_token_density(0.5, -1))

    def test_nan_attention_returns_nan(self):
        assert isnan(compute_per_token_density(nan, 10))


# ===========================================================================
# compute_region_attention_per_layer
# ===========================================================================

class TestComputeRegionAttentionPerLayer:
    def test_correct_per_layer_means(self):
        region_map = {"rules": {"tok_start": 0, "tok_end": 3}}
        # Layer 0: tokens [0.1, 0.2, 0.3, ...] → mean of [0:3] = 0.2
        # Layer 1: tokens [0.4, 0.5, 0.6, ...] → mean of [0:3] = 0.5
        per_layer = [
            (0, [0.1, 0.2, 0.3, 0.9]),
            (1, [0.4, 0.5, 0.6, 0.9]),
        ]
        sample = _make_sample(region_map, per_layer)

        result = compute_region_attention_per_layer(sample, "rules", num_layers=2)
        assert result[0] == pytest.approx(0.2)
        assert result[1] == pytest.approx(0.5)

    def test_missing_region_returns_zeros(self):
        sample = _make_sample({"other": {"tok_start": 0, "tok_end": 2}}, [])
        result = compute_region_attention_per_layer(sample, "missing", num_layers=4)
        np.testing.assert_array_equal(result, np.zeros(4))

    def test_zero_token_region_returns_zeros(self):
        region_map = {"empty": {"tok_start": 5, "tok_end": 5}}
        sample = _make_sample(region_map, [(0, [0.1] * 10)])
        result = compute_region_attention_per_layer(sample, "empty", num_layers=4)
        np.testing.assert_array_equal(result, np.zeros(4))

    def test_layer_beyond_num_layers_ignored(self):
        region_map = {"rules": {"tok_start": 0, "tok_end": 2}}
        per_layer = [
            (0, [0.5, 0.5]),
            (99, [0.9, 0.9]),  # Beyond num_layers=4
        ]
        sample = _make_sample(region_map, per_layer)

        result = compute_region_attention_per_layer(sample, "rules", num_layers=4)
        assert result[0] == pytest.approx(0.5)
        # Layer 99 should be ignored, layers 1-3 should be 0
        assert result[1] == 0.0
        assert result[2] == 0.0
        assert result[3] == 0.0

    def test_tok_end_beyond_weights_skips_layer(self):
        """If tok_end > len(weights), the layer should be skipped to avoid indexing errors."""
        region_map = {"rules": {"tok_start": 0, "tok_end": 10}}
        per_layer = [(0, [0.5] * 5)]  # Only 5 weights, but tok_end=10
        sample = _make_sample(region_map, per_layer)

        result = compute_region_attention_per_layer(sample, "rules", num_layers=4)
        np.testing.assert_array_equal(result, np.zeros(4))

    def test_explicit_n_tokens_zero_returns_zeros(self):
        """If n_tokens is explicitly 0, region should be treated as empty."""
        region_map = {"rules": {"tok_start": 0, "tok_end": 5, "n_tokens": 0}}
        per_layer = [(0, [0.5] * 5)]
        sample = _make_sample(region_map, per_layer)

        result = compute_region_attention_per_layer(sample, "rules", num_layers=4)
        np.testing.assert_array_equal(result, np.zeros(4))

    def test_inverted_boundaries_returns_zeros(self):
        """If tok_start > tok_end (malformed data), should return zeros, not NaN.

        Malformed region boundaries could come from corrupt data or a bug in
        the region annotation pipeline. The function should degrade gracefully
        instead of silently contaminating downstream results with NaN.
        """
        region_map = {"broken": {"tok_start": 5, "tok_end": 3}}
        per_layer = [(0, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])]
        sample = _make_sample(region_map, per_layer)

        result = compute_region_attention_per_layer(sample, "broken", num_layers=4)
        # Should be zeros, not NaN
        assert not np.any(np.isnan(result)), (
            f"Inverted boundaries produced NaN: {result}. "
            "Should return zeros for malformed regions."
        )
        np.testing.assert_array_equal(result, np.zeros(4))


# ===========================================================================
# cooking_curve_stats
# ===========================================================================

class TestCookingCurveStats:
    def test_peak_at_known_layer(self):
        curve = np.zeros(10)
        curve[5] = 1.0
        curve[9] = 0.2  # terminal

        stats = cooking_curve_stats(curve)
        assert stats["peak_layer"] == 5
        assert stats["peak_value"] == pytest.approx(1.0)
        assert stats["terminal_value"] == pytest.approx(0.2)
        assert stats["retention_ratio"] == pytest.approx(0.2)

    def test_all_zero_curve(self):
        curve = np.zeros(10)
        stats = cooking_curve_stats(curve)
        assert stats["peak_layer"] == -1
        assert stats["peak_value"] == 0.0
        assert stats["terminal_value"] == 0.0
        assert stats["retention_ratio"] == 0.0

    def test_single_element_curve(self):
        curve = np.array([0.5])
        stats = cooking_curve_stats(curve)
        assert stats["peak_layer"] == 0
        assert stats["peak_value"] == pytest.approx(0.5)
        assert stats["terminal_value"] == pytest.approx(0.5)
        assert stats["retention_ratio"] == pytest.approx(1.0)

    def test_empty_curve(self):
        curve = np.array([])
        stats = cooking_curve_stats(curve)
        assert stats["peak_layer"] == -1

    def test_constant_curve_retention_is_one(self):
        curve = np.full(10, 0.3)
        stats = cooking_curve_stats(curve)
        assert stats["retention_ratio"] == pytest.approx(1.0)


# ===========================================================================
# phase_mean
# ===========================================================================

class TestPhaseMean:
    def test_normal_range(self):
        curve = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        # phase_start=1, phase_end=3 → mean of [2, 3, 4]
        assert phase_mean(curve, 1, 3) == pytest.approx(3.0)

    def test_clamps_phase_end(self):
        curve = np.array([1.0, 2.0, 3.0])
        # phase_end=10 beyond length → clamps to len-1=2
        assert phase_mean(curve, 0, 10) == pytest.approx(2.0)

    def test_start_gt_end_returns_nan(self):
        curve = np.array([1.0, 2.0, 3.0])
        assert isnan(phase_mean(curve, 3, 1))

    def test_nan_values_skipped(self):
        curve = np.array([1.0, nan, 3.0])
        result = phase_mean(curve, 0, 2)
        # Should skip NaN → mean of [1.0, 3.0]
        assert result == pytest.approx(2.0)

    def test_single_layer_phase(self):
        """When start equals end, should return the value at that single layer."""
        curve = np.array([1.0, 2.0, 3.0])
        assert phase_mean(curve, 1, 1) == pytest.approx(2.0)

    def test_all_nan_returns_nan(self):
        curve = np.array([nan, nan, nan])
        assert isnan(phase_mean(curve, 0, 2))


# ===========================================================================
# safe_mean / safe_median / safe_std
# ===========================================================================

class TestSafeMean:
    def test_mixed_nan_and_valid(self):
        assert safe_mean([1.0, nan, 3.0]) == pytest.approx(2.0)

    def test_all_nan(self):
        assert isnan(safe_mean([nan, nan]))

    def test_empty_list(self):
        assert isnan(safe_mean([]))

    def test_single_value(self):
        assert safe_mean([5.0]) == pytest.approx(5.0)


class TestSafeMedian:
    def test_mixed_nan_and_valid(self):
        assert safe_median([1.0, nan, 5.0, 3.0]) == pytest.approx(3.0)

    def test_all_nan(self):
        assert isnan(safe_median([nan, nan]))

    def test_even_count(self):
        assert safe_median([1.0, 2.0, 3.0, 4.0]) == pytest.approx(2.5)


class TestSafeStd:
    def test_mixed_nan_and_valid(self):
        result = safe_std([2.0, nan, 4.0])
        # std of [2, 4] with ddof=1 = sqrt((2-3)^2 + (4-3)^2) = sqrt(2) ≈ 1.4142
        assert result == pytest.approx(2**0.5)

    def test_all_nan(self):
        assert isnan(safe_std([nan, nan]))

    def test_single_value_returns_nan(self):
        """With ddof=1, std of a single value is undefined."""
        assert isnan(safe_std([5.0]))

    def test_identical_values(self):
        assert safe_std([3.0, 3.0, 3.0]) == pytest.approx(0.0)
