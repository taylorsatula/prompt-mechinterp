"""Shared metric computations for MI analysis.

Canonical implementations extracted from compare_variants.py,
analyze_experiments.py, and analyze_v19_multiseed.py.
"""

from math import isnan
from typing import Dict, List, Optional

import numpy as np

from ..constants import FINAL_LAYERS


def avg_final_layers(
    attention_data: dict,
    position: str,
    region: str,
    n_layers: int = FINAL_LAYERS,
) -> float:
    """Average attention to a region at a query position over the last N layers.

    Args:
        attention_data: The 'attention' field from a result JSON.
        position: Query position name (e.g., 'terminal').
        region: Region name to measure.
        n_layers: Number of final layers to average (default: FINAL_LAYERS).

    Returns:
        Mean attention value, or NaN if not available.
    """
    if position not in attention_data:
        return float("nan")
    layers = attention_data[position]["per_layer"]
    values = []
    for layer_data in layers[-n_layers:]:
        prm = layer_data["per_region_mean"]
        if region in prm:
            values.append(prm[region])
    return sum(values) / len(values) if values else float("nan")


def compute_region_ratio(
    samples: List[dict],
    region_a: str,
    region_b: str,
    position: str = "terminal",
    n_layers: int = FINAL_LAYERS,
) -> List[float]:
    """Compute per-sample ratio of region_a / region_b attention.

    Generalized "context bleed" metric. For the original subcortical use case,
    region_a='conversation_turns', region_b='current_message'.

    Returns list of per-sample ratios (NaN excluded where region_b is zero).
    """
    ratios = []
    for sample in samples:
        attn = sample.get("attention", sample)
        a = avg_final_layers(attn, position, region_a, n_layers)
        b = avg_final_layers(attn, position, region_b, n_layers)
        if b > 0 and not isnan(a) and not isnan(b):
            ratios.append(a / b)
    return ratios


def compute_per_token_density(
    attention: float, n_tokens: int
) -> float:
    """Attention per token — fair cross-region comparison metric."""
    if n_tokens <= 0 or isnan(attention):
        return float("nan")
    return attention / n_tokens


def compute_region_attention_per_layer(
    sample: dict,
    region_name: str,
    position: str = "terminal",
    num_layers: int = 64,
) -> np.ndarray:
    """Return per-layer mean attention for a region as (num_layers,) array.

    Uses per-token attention data (weights array) and region boundaries
    to compute mean attention per token for the region at each layer.
    Returns zeros if region or data not found.
    """
    region_map = sample.get("region_map", {})
    region = region_map.get(region_name)
    if region is None:
        return np.zeros(num_layers)

    tok_start = region["tok_start"]
    tok_end = region["tok_end"]
    n_tokens = region.get("n_tokens", tok_end - tok_start)
    if n_tokens <= 0:
        return np.zeros(num_layers)

    per_layer = (
        sample.get("per_token_attention", {})
        .get(position, {})
        .get("per_layer", [])
    )
    if not per_layer:
        return np.zeros(num_layers)

    result = np.zeros(num_layers)
    for layer_data in per_layer:
        layer_idx = layer_data["layer"]
        weights = layer_data["weights"]
        if layer_idx < num_layers and tok_end <= len(weights):
            result[layer_idx] = np.mean(weights[tok_start:tok_end])

    return result


def cooking_curve_stats(curve: np.ndarray) -> dict:
    """Compute summary statistics for a cooking curve.

    Returns dict with peak_layer, peak_value, terminal_value,
    retention_ratio (terminal/peak).
    """
    if len(curve) == 0 or not np.any(curve > 0):
        return {
            "peak_layer": -1,
            "peak_value": 0.0,
            "terminal_value": 0.0,
            "retention_ratio": 0.0,
        }

    peak_layer = int(np.argmax(curve))
    peak_value = float(curve[peak_layer])
    terminal_value = float(curve[-1])
    retention = terminal_value / peak_value if peak_value > 0 else 0.0

    return {
        "peak_layer": peak_layer,
        "peak_value": peak_value,
        "terminal_value": terminal_value,
        "retention_ratio": retention,
    }


def phase_mean(
    curve: np.ndarray, phase_start: int, phase_end: int
) -> float:
    """Mean attention during a layer range."""
    if phase_end >= len(curve):
        phase_end = len(curve) - 1
    if phase_start > phase_end:
        return float("nan")
    segment = curve[phase_start : phase_end + 1]
    clean = segment[~np.isnan(segment)]
    return float(np.mean(clean)) if len(clean) > 0 else float("nan")


def safe_mean(values: list) -> float:
    """NaN-safe mean."""
    clean = [v for v in values if not isnan(v)]
    return float(np.mean(clean)) if clean else float("nan")


def safe_median(values: list) -> float:
    """NaN-safe median."""
    clean = sorted(v for v in values if not isnan(v))
    return float(np.median(clean)) if clean else float("nan")


def safe_std(values: list) -> float:
    """NaN-safe standard deviation (ddof=1)."""
    clean = [v for v in values if not isnan(v)]
    return float(np.std(clean, ddof=1)) if len(clean) > 1 else float("nan")
