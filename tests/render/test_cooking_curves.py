"""Tests for render/cooking_curves.py — trajectory computation and tick generation."""

import math

import numpy as np
import pytest

from prompt_mechinterp.render.cooking_curves import (
    _nice_ticks,
    compute_region_trajectories,
)


# ===========================================================================
# compute_region_trajectories
# ===========================================================================

class TestComputeRegionTrajectories:
    def test_correct_mean_per_layer(self):
        region_map = {
            "rules": {"tok_start": 0, "tok_end": 3},
            "examples": {"tok_start": 3, "tok_end": 5},
        }
        layer_weights = {
            0: np.array([0.3, 0.3, 0.3, 0.1, 0.1]),
            1: np.array([0.1, 0.1, 0.1, 0.5, 0.5]),
            2: np.array([0.2, 0.2, 0.2, 0.3, 0.3]),
        }

        trajs = compute_region_trajectories(
            region_map, layer_weights, ["rules", "examples"]
        )

        # rules: mean of tokens [0:3] per layer
        assert trajs["rules"][0] == pytest.approx(0.3)
        assert trajs["rules"][1] == pytest.approx(0.1)
        assert trajs["rules"][2] == pytest.approx(0.2)

        # examples: mean of tokens [3:5] per layer
        assert trajs["examples"][0] == pytest.approx(0.1)
        assert trajs["examples"][1] == pytest.approx(0.5)
        assert trajs["examples"][2] == pytest.approx(0.3)

    def test_region_not_in_map_excluded(self):
        region_map = {"rules": {"tok_start": 0, "tok_end": 3}}
        layer_weights = {0: np.array([0.5, 0.5, 0.5])}

        trajs = compute_region_trajectories(
            region_map, layer_weights, ["rules", "missing"]
        )
        assert "rules" in trajs
        assert "missing" not in trajs

    def test_zero_width_region_excluded(self):
        region_map = {"empty": {"tok_start": 3, "tok_end": 3}}
        layer_weights = {0: np.array([0.5, 0.5, 0.5, 0.5])}

        trajs = compute_region_trajectories(
            region_map, layer_weights, ["empty"]
        )
        assert "empty" not in trajs

    def test_trajectory_length_matches_layer_count(self):
        region_map = {"rules": {"tok_start": 0, "tok_end": 2}}
        layer_weights = {i: np.array([0.1, 0.2]) for i in range(8)}

        trajs = compute_region_trajectories(
            region_map, layer_weights, ["rules"]
        )
        assert len(trajs["rules"]) == 8

    def test_layers_processed_in_sorted_order(self):
        """Layer indices should be processed in order, not dict insertion order."""
        region_map = {"rules": {"tok_start": 0, "tok_end": 1}}
        # Insert layers out of order
        layer_weights = {
            2: np.array([0.3]),
            0: np.array([0.1]),
            1: np.array([0.2]),
        }

        trajs = compute_region_trajectories(
            region_map, layer_weights, ["rules"]
        )
        assert trajs["rules"][0] == pytest.approx(0.1)
        assert trajs["rules"][1] == pytest.approx(0.2)
        assert trajs["rules"][2] == pytest.approx(0.3)

    def test_tok_end_beyond_weights_gives_zero(self):
        """When tok_end exceeds the weight array length, should give 0 for that layer.

        This matches the defensive behavior in metrics.compute_region_attention_per_layer
        which skips layers where the weight array is too short. Without this guard,
        np.mean(w[0:10]) when len(w)=5 silently computes the mean of tokens 0-4
        instead of the intended 0-9, giving wrong results.
        """
        region_map = {"rules": {"tok_start": 0, "tok_end": 10}}
        layer_weights = {
            0: np.array([0.5] * 5),  # Only 5 weights, but region says 10 tokens
        }

        trajs = compute_region_trajectories(
            region_map, layer_weights, ["rules"]
        )
        # Should be 0 (skipped) not 0.5 (wrong — partial mean of 5 tokens)
        assert trajs["rules"][0] == pytest.approx(0.0), (
            f"Got {trajs['rules'][0]} — silently computed mean of 5 tokens "
            "instead of the intended 10. Should skip this layer."
        )


# ===========================================================================
# _nice_ticks
# ===========================================================================

class TestNiceTicks:
    def test_small_range(self):
        ticks = _nice_ticks(0, 0.01)
        # Should produce reasonable tick spacing
        assert len(ticks) >= 3
        assert all(0 <= t <= 0.012 for t in ticks)
        # Spacing should be uniform
        spacings = [ticks[i + 1] - ticks[i] for i in range(len(ticks) - 1)]
        assert all(abs(s - spacings[0]) < 1e-10 for s in spacings)

    def test_large_range(self):
        ticks = _nice_ticks(0, 100)
        assert len(ticks) >= 3
        # Ticks should be at "nice" multiples
        for t in ticks:
            # Each tick should be a round number
            assert t == int(t) or t == 0

    def test_equal_min_max(self):
        ticks = _nice_ticks(0.5, 0.5)
        assert ticks == [0.5]

    def test_max_less_than_min(self):
        ticks = _nice_ticks(1.0, 0.5)
        assert ticks == [1.0]

    def test_ticks_near_range(self):
        """Ticks should be near the input range, snapped to nice values."""
        ticks = _nice_ticks(0, 0.9)
        # Should start at or near 0
        assert min(ticks) <= 0.01
        # Should reach at or near the max
        assert max(ticks) >= 0.8

    def test_ticks_are_ascending(self):
        ticks = _nice_ticks(0, 50)
        for i in range(1, len(ticks)):
            assert ticks[i] > ticks[i - 1]
