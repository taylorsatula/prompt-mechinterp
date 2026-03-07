"""Tests for formatting helpers — delta_str and fmt."""

from math import isnan, nan

from prompt_mechinterp.analysis.formatting import delta_str, fmt


class TestDeltaStr:
    def test_positive_change(self):
        result = delta_str(0.6, 0.5)
        assert "+20.0%" in result

    def test_negative_change(self):
        result = delta_str(0.4, 0.5)
        assert "-20.0%" in result

    def test_no_change(self):
        result = delta_str(0.5, 0.5)
        assert "+0.0%" in result

    def test_ref_zero_returns_na(self):
        result = delta_str(0.6, 0)
        assert "N/A" in result

    def test_val_nan_returns_na(self):
        result = delta_str(nan, 0.5)
        assert "N/A" in result

    def test_ref_nan_returns_na(self):
        result = delta_str(0.5, nan)
        assert "N/A" in result

    def test_both_nan_returns_na(self):
        result = delta_str(nan, nan)
        assert "N/A" in result

    def test_negative_ref_uses_absolute_value(self):
        # delta_str uses abs(ref) in denominator
        result = delta_str(-0.3, -0.5)
        # (-0.3 - -0.5) / abs(-0.5) * 100 = 0.2/0.5 * 100 = +40.0%
        assert "+40.0%" in result


class TestFmt:
    def test_normal_value_four_decimals(self):
        result = fmt(0.1234)
        assert "0.1234" in result

    def test_very_small_value_six_decimals(self):
        result = fmt(0.00005)
        assert "0.000050" in result

    def test_small_value_five_decimals(self):
        result = fmt(0.0005)
        assert "0.00050" in result

    def test_nan_returns_na(self):
        result = fmt(nan)
        assert "N/A" in result

    def test_zero_uses_four_decimals(self):
        # Zero should not trigger the small-value branches
        result = fmt(0.0)
        assert "0.0000" in result

    def test_result_is_right_justified(self):
        result = fmt(0.1234, width=12)
        assert len(result) == 12
        assert result.endswith("0.1234")
