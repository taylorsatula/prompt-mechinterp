"""Table output helpers for MI analysis scripts."""

from math import isnan


def fmt(val: float, width: int = 9) -> str:
    """Format a float with smart precision, handling NaN."""
    if isnan(val):
        return "N/A".rjust(width)
    if abs(val) < 0.0001 and val != 0:
        return f"{val:.6f}".rjust(width)
    if abs(val) < 0.001 and val != 0:
        return f"{val:.5f}".rjust(width)
    return f"{val:.4f}".rjust(width)


def pct(val: float, width: int = 8) -> str:
    """Format as percentage."""
    if isnan(val):
        return "N/A".rjust(width)
    return f"{val*100:.3f}%".rjust(width)


def delta_str(val: float, ref: float, width: int = 9) -> str:
    """Format delta as percentage change with +/- prefix."""
    if isnan(val) or isnan(ref) or ref == 0:
        return "N/A".rjust(width)
    pct_change = (val - ref) / abs(ref) * 100
    sign = "+" if pct_change >= 0 else ""
    return f"{sign}{pct_change:.1f}%".rjust(width)


def print_header(title: str, width: int = 90) -> None:
    """Print a bordered section header."""
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


def print_subheader(title: str) -> None:
    """Print a lighter subsection header."""
    print(f"\n  --- {title} ---")
