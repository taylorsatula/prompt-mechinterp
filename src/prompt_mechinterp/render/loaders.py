"""Unified data loading from MI result JSONs.

Each loader extracts the specific data shape needed by its corresponding
renderer, but they all share the same JSON traversal pattern.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from ._shared import parse_layer_spec


def _load_result_json(path: str) -> dict:
    """Load and validate a result JSON file."""
    with open(path) as f:
        return json.load(f)


def _require_per_token(data: dict) -> None:
    """Validate that per-token attention data exists."""
    if "per_token_attention" not in data:
        print("ERROR: This result JSON has no per-token attention data.")
        print("Re-run the analysis without --no-per-token to capture it.")
        sys.exit(1)
    if "token_labels" not in data:
        print("ERROR: This result JSON has no token_labels.")
        print("Re-run the analysis (latest version) to capture token labels.")
        sys.exit(1)


def _require_position(per_token: dict, position: str) -> None:
    """Validate that the requested position exists in the data."""
    if position not in per_token:
        available = list(per_token.keys())
        print(f"ERROR: Position '{position}' not found. Available: {available}")
        sys.exit(1)


def _extract_piece_boundaries(region_map: dict) -> dict:
    """Extract top-level piece boundaries from the region map."""
    piece_boundaries = {}
    for piece_name in ["system_prompt", "user_message", "response"]:
        if piece_name in region_map:
            piece_boundaries[piece_name] = region_map[piece_name]
    return piece_boundaries


def load_heatmap_data(
    result_path: str,
    position: str,
    layer_spec: str,
) -> Tuple[List[str], np.ndarray, Dict, Dict]:
    """Load and average per-token attention from result JSON.

    Returns:
        token_labels: decoded text for each token
        weights: 1D array of attention weights (averaged across selected layers)
        region_map: token-level region boundaries
        piece_boundaries: top-level piece info
    """
    data = _load_result_json(result_path)
    _require_per_token(data)

    token_labels = data["token_labels"]
    region_map = data["region_map"]
    per_token = data["per_token_attention"]
    _require_position(per_token, position)

    # Detect num_layers from data
    all_layers = [e["layer"] for e in per_token[position]["per_layer"]]
    num_layers = max(all_layers) + 1 if all_layers else 64

    target_layers = set(parse_layer_spec(layer_spec, num_layers))
    per_layer = per_token[position]["per_layer"]

    collected = []
    for entry in per_layer:
        if entry["layer"] in target_layers:
            collected.append(np.array(entry["weights"], dtype=np.float64))

    if not collected:
        print(f"ERROR: No layer data found for layers {sorted(target_layers)}")
        sys.exit(1)

    weights = np.mean(collected, axis=0)
    piece_boundaries = _extract_piece_boundaries(region_map)

    return token_labels, weights, region_map, piece_boundaries


def load_cooking_data(
    result_path: str,
    position: str,
) -> Tuple[Dict[str, Dict], Dict[int, np.ndarray], List[str]]:
    """Load per-layer attention data and region map from result JSON.

    Returns:
        region_map: region name -> {tok_start, tok_end, ...}
        layer_weights: layer_number -> 1D attention weight array
        token_labels: decoded text for each token
    """
    data = _load_result_json(result_path)
    _require_per_token(data)

    per_token = data["per_token_attention"]
    _require_position(per_token, position)

    region_map = data["region_map"]
    token_labels = data.get("token_labels", [])

    layer_weights = {}
    for entry in per_token[position]["per_layer"]:
        layer_weights[entry["layer"]] = np.array(
            entry["weights"], dtype=np.float64
        )

    return region_map, layer_weights, token_labels


def load_all_layers(
    result_path: str,
    position: str,
) -> Tuple[List[str], Dict[int, np.ndarray], Dict, Dict]:
    """Load per-token attention for ALL layers from result JSON.

    Returns:
        token_labels, layer_weights (layer->1D array), region_map, piece_boundaries
    """
    data = _load_result_json(result_path)
    _require_per_token(data)

    per_token = data["per_token_attention"]
    _require_position(per_token, position)

    token_labels = data["token_labels"]
    region_map = data["region_map"]

    layer_weights = {}
    for entry in per_token[position]["per_layer"]:
        layer_weights[entry["layer"]] = np.array(
            entry["weights"], dtype=np.float64
        )

    piece_boundaries = _extract_piece_boundaries(region_map)

    return token_labels, layer_weights, region_map, piece_boundaries


def load_variant_curves(
    base_path: Path,
    dirname: str,
) -> Dict[str, np.ndarray]:
    """Load all samples in a directory, return {region: (n_samples, n_layers) array}.

    Uses per-region attention from the 'attention' field (not per-token),
    at the 'terminal' position.
    """
    dirpath = base_path / dirname
    all_data: Dict[str, list] = {}

    for f in sorted(dirpath.glob("sample_*.json")):
        with open(f) as fh:
            d = json.load(fh)

        rm = d["region_map"]
        per_layer = d["attention"]["terminal"]["per_layer"]

        for region in rm:
            if region not in all_data:
                all_data[region] = []
            curve = []
            for layer_data in per_layer:
                val = layer_data["per_region_mean"].get(region, 0.0)
                curve.append(val)
            all_data[region].append(curve)

    result = {}
    for region, curves in all_data.items():
        result[region] = np.array(curves)

    return result
