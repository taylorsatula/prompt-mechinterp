#!/usr/bin/env python3
"""Generic N-variant comparison with auto-discovered regions.

Replaces compare_variants.py and analyze_v19_multiseed.py with a
model-agnostic, region-agnostic comparison tool.

Usage:
    python -m prompt_mechinterp.analysis.compare \\
        --base-dir ./data/results \\
        --variants results_baseline:Baseline results_composite:Composite

    python -m prompt_mechinterp.analysis.compare \\
        --base-dir ./data/results \\
        --variants results_baseline:Baseline results_v19:V19 \\
        --ratio conversation_turns:current_message \\
        --by-seed \\
        --logit-lens-tokens "<"
"""

import argparse
import json
from math import isnan
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from ..constants import FINAL_LAYERS, SKIP_REGIONS
from .formatting import delta_str, fmt, pct, print_header, print_subheader
from .metrics import (
    avg_final_layers,
    compute_region_ratio,
    safe_mean,
    safe_median,
    safe_std,
)


def _load_variant(dirpath: Path) -> List[dict]:
    """Load all sample_*.json files from a directory."""
    samples = []
    for f in sorted(dirpath.glob("sample_*.json")):
        with open(f) as fh:
            samples.append(json.load(fh))
    return samples


def _auto_discover_regions(samples: List[dict]) -> List[str]:
    """Get all regions from the first sample, excluding containers."""
    if not samples:
        return []
    region_map = samples[0].get("region_map", {})
    return sorted(r for r in region_map if r not in SKIP_REGIONS)


def _detect_num_layers(samples: List[dict]) -> int:
    """Detect number of layers from data."""
    for sample in samples:
        attn = sample.get("attention", {})
        for pos_data in attn.values():
            per_layer = pos_data.get("per_layer", [])
            if per_layer:
                return max(e["layer"] for e in per_layer) + 1
    return 64


def _detect_seed(case_id: str) -> str:
    """Detect seed from case_id suffix (e.g., sample_01_b -> seed_b)."""
    parts = case_id.rsplit("_", 1)
    if len(parts) == 2 and len(parts[1]) == 1 and parts[1].isalpha():
        return f"seed_{parts[1]}"
    return "seed_default"


def table_terminal_attention(
    variants: Dict[str, List[dict]],
    regions: List[str],
    position: str,
    num_layers: int = 64,
) -> None:
    """Per-region terminal attention table."""
    layer_start = num_layers - FINAL_LAYERS
    layer_end = num_layers - 1
    print_header(f"Per-Region Terminal Attention ({position}, L{layer_start}-{layer_end} avg)")

    first_label = next(iter(variants))
    baseline_vals: Dict[str, float] = {}

    header = f"  {'Region':<22}"
    for label in variants:
        header += f" {label:>12}"
    if len(variants) > 1:
        header += " |"
        for label in list(variants.keys())[1:]:
            header += f" {'d_' + label:>9}"
    print(header)
    print(f"  {'─' * 22}" + "".join(f" {'─' * 12}" for _ in variants) +
          (" | " + " ".join(f"{'─' * 9}" for _ in list(variants.keys())[1:]) if len(variants) > 1 else ""))

    for region in regions:
        row = f"  {region:<22}"
        for label, samples in variants.items():
            vals = [avg_final_layers(s.get("attention", s), position, region) for s in samples]
            m = safe_mean(vals)
            if label == first_label:
                baseline_vals[region] = m
            row += f" {pct(m, 12)}"

        if len(variants) > 1:
            row += " |"
            for label, samples in list(variants.items())[1:]:
                vals = [avg_final_layers(s.get("attention", s), position, region) for s in samples]
                m = safe_mean(vals)
                row += f" {delta_str(m, baseline_vals.get(region, float('nan')))}"
        print(row)


def table_region_ratios(
    variants: Dict[str, List[dict]],
    ratio_pairs: List[Tuple[str, str]],
    position: str,
    num_layers: int = 64,
) -> None:
    """Region ratio table (e.g., context bleed)."""
    layer_start = num_layers - FINAL_LAYERS
    layer_end = num_layers - 1
    for region_a, region_b in ratio_pairs:
        print_header(f"Region Ratio: {region_a} / {region_b} ({position}, L{layer_start}-{layer_end} avg)")
        print(f"  Lower = less {region_a} relative to {region_b}.")
        print()

        header = f"  {'Variant':<14} {'n':>4} {'Mean':>9} {'Median':>9} {'Std':>9} {'Min':>9} {'Max':>9}"
        print(header)
        print(f"  {'─' * 14} {'─' * 4} {'─' * 9} {'─' * 9} {'─' * 9} {'─' * 9} {'─' * 9}")

        for label, samples in variants.items():
            ratios = compute_region_ratio(samples, region_a, region_b, position)
            clean = [r for r in ratios if not isnan(r)]
            print(
                f"  {label:<14} {len(clean):>4} "
                f"{fmt(safe_mean(ratios))} {fmt(safe_median(ratios))} "
                f"{fmt(safe_std(ratios))} "
                f"{fmt(min(clean) if clean else float('nan'))} "
                f"{fmt(max(clean) if clean else float('nan'))}"
            )


def table_per_token_density(
    variants: Dict[str, List[dict]],
    regions: List[str],
    position: str,
    num_layers: int = 64,
) -> None:
    """Per-token attention density table."""
    layer_start = num_layers - FINAL_LAYERS
    layer_end = num_layers - 1
    print_header(f"Per-Token Attention Density ({position}, L{layer_start}-{layer_end} avg)")
    print(f"  Density = attention_weight / n_tokens * 1000. Higher = more attention per token.")
    print()

    header = f"  {'Region':<22}"
    for label in variants:
        header += f" {label:>12}"
    print(header)
    print(f"  {'─' * 22}" + "".join(f" {'─' * 12}" for _ in variants))

    for region in regions:
        row = f"  {region:<22}"
        for label, samples in variants.items():
            densities = []
            for s in samples:
                attn = avg_final_layers(s.get("attention", s), position, region)
                rm = s.get("region_map", {}).get(region, {})
                n_tok = rm.get("n_tokens", rm.get("tok_end", 0) - rm.get("tok_start", 0))
                if n_tok > 0 and not isnan(attn):
                    densities.append(attn / n_tok * 1000)
            row += f" {fmt(safe_mean(densities), 12)}"
        print(row)


def table_logit_lens(
    variants: Dict[str, List[dict]],
    tracked_tokens: List[str],
    position: str,
    num_layers: int,
) -> None:
    """Logit lens trajectory table for tracked tokens."""
    if not tracked_tokens:
        return

    check_layers = [0, 16, 32, 48, 56, num_layers - 4, num_layers - 1]
    check_layers = sorted(set(l for l in check_layers if 0 <= l < num_layers))

    for token_str in tracked_tokens:
        print_header(f"Logit Lens: '{token_str}' Rank Trajectory at {position}")
        print(f"  Rank of token across layers. Lower = stronger prediction.")
        print()

        header = f"  {'Variant':<14}"
        for l in check_layers:
            header += f" {'L' + str(l):>6}"
        print(header)
        print(f"  {'─' * 14}" + "".join(f" {'─' * 6}" for _ in check_layers))

        for label, samples in variants.items():
            layer_ranks: Dict[int, list] = {l: [] for l in check_layers}
            for sample in samples:
                ll = sample.get("logit_lens", {}).get(position, [])
                for entry in ll:
                    if entry["layer"] in check_layers:
                        tracked = entry.get("tracked", {})
                        tok_data = tracked.get(token_str, {})
                        rank = tok_data.get("rank", float("nan"))
                        layer_ranks[entry["layer"]].append(rank)

            row = f"  {label:<14}"
            for l in check_layers:
                vals = layer_ranks[l]
                avg = sum(vals) / len(vals) if vals else float("nan")
                if isnan(avg):
                    row += f" {'N/A':>6}"
                else:
                    row += f" {avg:6.0f}"
            print(row)


def table_by_seed(
    variants: Dict[str, List[dict]],
    regions: List[str],
    position: str,
) -> None:
    """Per-seed breakdown for variants with multiple seeds."""
    for label, samples in variants.items():
        seeds = sorted(set(_detect_seed(s["metadata"]["case_id"]) for s in samples))
        if len(seeds) <= 1:
            continue

        print_header(f"{label} — Per-Seed Terminal Attention")

        header = f"  {'Region':<22}"
        for seed in seeds:
            header += f" {seed:>12}"
        header += f" {'All':>12}"
        print(header)
        print(f"  {'─' * 22}" + "".join(f" {'─' * 12}" for _ in seeds) + f" {'─' * 12}")

        for region in regions:
            row = f"  {region:<22}"
            for seed in seeds:
                ss = [s for s in samples if _detect_seed(s["metadata"]["case_id"]) == seed]
                vals = [avg_final_layers(s.get("attention", s), position, region) for s in ss]
                row += f" {pct(safe_mean(vals), 12)}"
            all_vals = [avg_final_layers(s.get("attention", s), position, region) for s in samples]
            row += f" {pct(safe_mean(all_vals), 12)}"
            print(row)


def main():
    parser = argparse.ArgumentParser(
        description="Generic N-variant MI comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--base-dir", required=True,
        help="Base directory containing variant result directories",
    )
    parser.add_argument(
        "--variants", nargs="+", required=True,
        help="Variant specs as dirname:label pairs (e.g., results_baseline:Baseline)",
    )
    parser.add_argument(
        "--position", default="terminal",
        help="Query position to analyze (default: terminal)",
    )
    parser.add_argument(
        "--ratio", nargs="*", default=[],
        help="Region ratio pairs as region_a:region_b (e.g., conversation_turns:current_message)",
    )
    parser.add_argument(
        "--logit-lens-tokens", nargs="*", default=[],
        help="Tokens to track in logit lens trajectory",
    )
    parser.add_argument(
        "--by-seed", action="store_true",
        help="Break down metrics per seed for stability analysis",
    )
    parser.add_argument(
        "--metrics", default="all",
        help="Comma-separated metrics to print: terminal,density,ratios,logit_lens,all",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    metrics = set(args.metrics.split(","))
    show_all = "all" in metrics

    # Parse variants
    variants: Dict[str, List[dict]] = {}
    for spec in args.variants:
        if ":" in spec:
            dirname, label = spec.split(":", 1)
        else:
            dirname = spec
            label = spec.replace("results_", "")
        dirpath = base_dir / dirname
        if not dirpath.exists():
            print(f"  SKIP {label}: {dirpath} not found")
            continue
        samples = _load_variant(dirpath)
        variants[label] = samples
        print(f"  Loaded {label}: {len(samples)} samples")

    if not variants:
        print("No data found!")
        return

    # Auto-discover regions from first variant
    first_samples = next(iter(variants.values()))
    regions = _auto_discover_regions(first_samples)
    num_layers = _detect_num_layers(first_samples)
    print(f"  Auto-discovered {len(regions)} regions, {num_layers} layers")

    # Parse ratio pairs
    ratio_pairs = []
    for r in args.ratio:
        parts = r.split(":")
        if len(parts) == 2:
            ratio_pairs.append((parts[0], parts[1]))

    # Print tables
    if show_all or "terminal" in metrics:
        table_terminal_attention(variants, regions, args.position, num_layers)

    if show_all or "density" in metrics:
        table_per_token_density(variants, regions, args.position, num_layers)

    if show_all or "ratios" in metrics:
        if ratio_pairs:
            table_region_ratios(variants, ratio_pairs, args.position, num_layers)

    if show_all or "logit_lens" in metrics:
        table_logit_lens(variants, args.logit_lens_tokens, args.position, num_layers)

    if args.by_seed:
        table_by_seed(variants, regions, args.position)


if __name__ == "__main__":
    main()
