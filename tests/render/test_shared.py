"""Tests for render/_shared.py — normalize_weights, parse_layer_spec, gaussian_smooth, sanitize_token."""

import numpy as np
import pytest

from prompt_mechinterp.render._shared import (
    gaussian_smooth,
    normalize_weights,
    parse_layer_spec,
    sanitize_token,
)


# ===========================================================================
# normalize_weights
# ===========================================================================

class TestNormalizeWeights:
    def test_rank_preserving(self):
        """Monotonically increasing input should produce monotonically increasing output."""
        weights = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        normed = normalize_weights(weights, clip_low=0)
        # Each consecutive value should be >= the previous
        for i in range(1, len(normed)):
            assert normed[i] >= normed[i - 1]

    def test_tie_handling(self):
        """Identical values should all get the same normalized rank."""
        weights = np.array([5.0, 5.0, 5.0])
        normed = normalize_weights(weights, clip_low=0)
        assert normed[0] == normed[1] == normed[2]

    def test_mask_zeroes_excluded_tokens(self):
        """Masked-out tokens should get 0, included tokens ranked among themselves."""
        weights = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mask = np.array([True, False, True, False, True])
        normed = normalize_weights(weights, clip_low=0, mask=mask)

        # Masked-out positions should be 0
        assert normed[1] == 0.0
        assert normed[3] == 0.0
        # Included positions should be ranked
        assert normed[0] < normed[2] < normed[4]

    def test_clip_low_zeros_bottom_percentile(self):
        """With clip_low=20, the bottom 20% of ranks should map to 0."""
        weights = np.arange(1.0, 11.0)  # 10 elements
        normed = normalize_weights(weights, clip_low=20)

        # Bottom 2 of 10 (20%) should be at or near 0
        assert normed[0] == 0.0
        assert normed[1] == 0.0
        # Upper values should be positive
        assert normed[-1] > 0.0

    def test_clip_low_zero_no_floor(self):
        weights = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        normed = normalize_weights(weights, clip_low=0)
        # Lowest rank should be 0 (rank 0 / (n-1))
        assert normed[0] == 0.0
        # Highest rank should be 1.0
        assert normed[-1] == pytest.approx(1.0)

    def test_single_element(self):
        """A single token has no relative rank, so it gets the minimum value."""
        weights = np.array([7.0])
        normed = normalize_weights(weights, clip_low=0)
        assert normed[0] == 0.0

    def test_output_range_zero_to_one(self):
        """All output values should be in [0, 1]."""
        rng = np.random.RandomState(42)
        weights = rng.exponential(1.0, size=100)
        normed = normalize_weights(weights)
        assert np.all(normed >= 0.0)
        assert np.all(normed <= 1.0)

    def test_equalization_spreads_power_law(self):
        """Rank-based normalization should spread a power-law distribution uniformly.

        This is the whole point of the function: raw attention follows a power law
        where a few tokens dominate. Rank-based normalization gives every percentile
        equal visual weight, so the output should be roughly uniform regardless of
        input distribution.
        """
        rng = np.random.RandomState(42)
        # Power-law-ish input: most values near 0, a few very large
        weights = rng.exponential(0.01, size=200)
        weights[0] = 10.0  # one huge outlier

        normed = normalize_weights(weights, clip_low=0)

        # If equalization works, the output histogram should be roughly uniform.
        # Split into 5 bins and check each has roughly 20% of values.
        hist, _ = np.histogram(normed, bins=5, range=(0.0, 1.0))
        fractions = hist / len(normed)
        for frac in fractions:
            assert 0.10 < frac < 0.30, (
                f"Bin fraction {frac:.2f} is far from uniform (expected ~0.20). "
                f"Histogram: {hist}"
            )


# ===========================================================================
# gaussian_smooth
# ===========================================================================

class TestGaussianSmooth:
    def test_sigma_zero_returns_input(self):
        values = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
        result = gaussian_smooth(values, sigma=0)
        np.testing.assert_array_equal(result, values)

    def test_sigma_positive_spreads_spike(self):
        spike = np.zeros(21)
        spike[10] = 1.0
        smoothed = gaussian_smooth(spike, sigma=2.0)

        # Peak should decrease
        assert smoothed[10] < 1.0
        # Neighbors should gain some energy
        assert smoothed[9] > 0.0
        assert smoothed[11] > 0.0

    def test_output_length_equals_input(self):
        """Output length must always equal input length, regardless of sigma."""
        values = np.arange(64, dtype=float)
        result = gaussian_smooth(values, sigma=1.0)
        assert len(result) == len(values)

    def test_output_length_with_large_sigma(self):
        """Even when kernel would be larger than input, output length must match.

        This caught a real bug: np.convolve mode='same' returns max(M, N),
        so an uncapped kernel produced output longer than input.
        """
        values = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
        result = gaussian_smooth(values, sigma=2.0)  # kernel_size would be 13
        assert len(result) == len(values), (
            f"Output length {len(result)} != input length {len(values)}"
        )

    def test_smoothing_preserves_approximate_total(self):
        """Gaussian smoothing should roughly preserve the sum (energy)."""
        values = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        result = gaussian_smooth(values, sigma=1.0)
        # Some energy lost at edges, but most preserved
        assert abs(result.sum() - values.sum()) < 0.2


# ===========================================================================
# parse_layer_spec
# ===========================================================================

class TestParseLayerSpec:
    def test_all(self):
        result = parse_layer_spec("all", 64)
        assert result == list(range(64))

    def test_final(self):
        result = parse_layer_spec("final", 64)
        assert result == [60, 61, 62, 63]

    def test_single_number(self):
        assert parse_layer_spec("48", 64) == [48]

    def test_comma_separated(self):
        assert parse_layer_spec("0,16,32", 64) == [0, 16, 32]

    def test_range(self):
        assert parse_layer_spec("60-63", 64) == [60, 61, 62, 63]

    def test_mixed_commas_and_range(self):
        result = parse_layer_spec("0,16,60-63", 64)
        assert result == [0, 16, 60, 61, 62, 63]

    def test_case_insensitive(self):
        assert parse_layer_spec("ALL", 64) == list(range(64))
        assert parse_layer_spec("Final", 64) == [60, 61, 62, 63]

    def test_whitespace_tolerance(self):
        assert parse_layer_spec("  0 , 16 , 32  ", 64) == [0, 16, 32]

    def test_final_scales_to_smaller_model(self):
        result = parse_layer_spec("final", 32)
        assert result == [28, 29, 30, 31]


# ===========================================================================
# sanitize_token
# ===========================================================================

class TestIsNewlineToken:
    """Tests for is_newline_token — detects whether a token represents a line break."""

    def test_plain_newline(self):
        from prompt_mechinterp.render._shared import is_newline_token
        assert is_newline_token("\n") is True

    def test_crlf_is_newline(self):
        """Windows-style \\r\\n should be detected as a newline token."""
        from prompt_mechinterp.render._shared import is_newline_token
        assert is_newline_token("\r\n") is True

    def test_bare_carriage_return_is_not_newline(self):
        """A bare \\r without \\n is not a newline — it's a control character."""
        from prompt_mechinterp.render._shared import is_newline_token
        assert is_newline_token("\r") is False

    def test_text_with_newline_is_not_newline_token(self):
        from prompt_mechinterp.render._shared import is_newline_token
        assert is_newline_token("hello\n") is False

    def test_empty_string_is_not_newline(self):
        from prompt_mechinterp.render._shared import is_newline_token
        assert is_newline_token("") is False


class TestSanitizeToken:
    def test_tab_becomes_double_space(self):
        assert sanitize_token("\t") == "  "

    def test_carriage_return_removed(self):
        assert sanitize_token("hello\rworld") == "helloworld"

    def test_control_char_escaped(self):
        assert sanitize_token("\x01") == "\\x01"

    def test_normal_text_unchanged(self):
        assert sanitize_token("hello world") == "hello world"

    def test_newline_preserved(self):
        """Newlines are not control chars for display purposes."""
        assert sanitize_token("\n") == "\n"

    def test_mixed_content(self):
        assert sanitize_token("a\tb\x02c\rd") == "a  b\\x02cd"
