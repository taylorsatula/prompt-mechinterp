#!/usr/bin/env python3
"""Generate markdown experiment reports with delta-from-baseline analysis.

Usage:
    python -m prompt_mechinterp.analysis.report \\
        --base-dir ./data/results \\
        --experiments baseline:Baseline:results_baseline composite:Composite:results_composite \\
        --output-dir ./reports
"""

import argparse
import json
from math import isnan
from pathlib import Path
from typing import Dict, List

import numpy as np

from ..constants import FINAL_LAYERS, SKIP_REGIONS
from .metrics import (
    compute_region_attention_per_layer,
    cooking_curve_stats,
    safe_mean,
)


def _load_samples(dirpath: Path) -> List[dict]:
    samples = []
    for f in sorted(dirpath.glob("sample_*.json")):
        with open(f) as fh:
            samples.append(json.load(fh))
    return samples


def _auto_regions(samples: List[dict]) -> List[str]:
    if not samples:
        return []
    rm = samples[0].get("region_map", {})
    return sorted(r for r in rm if r not in SKIP_REGIONS)


def _detect_num_layers(samples: List[dict]) -> int:
    for s in samples:
        pt = s.get("per_token_attention", {})
        for pos_data in pt.values():
            per_layer = pos_data.get("per_layer", [])
            if per_layer:
                return max(e["layer"] for e in per_layer) + 1
    return 64


def _classify_story(peak_layer: int, ratio: float) -> str:
    if peak_layer <= 2:
        if ratio > 10:
            return "Cooked first, strong fade"
        elif ratio > 3:
            return "Cooked first"
        else:
            return "Early read"
    elif peak_layer <= 6:
        return f"Cooked by L{peak_layer}"
    elif peak_layer <= 11:
        return "Absorption phase"
    elif peak_layer <= 31:
        return "Deep compression"
    elif peak_layer <= 47:
        return "Mid-phase peak"
    elif peak_layer <= 55:
        return "Output prep phase"
    else:
        if ratio < 2:
            return "Latest bloomer"
        return "Late bloomer"


def compute_cooking_table(
    samples: List[dict],
    regions: List[str],
    position: str = "terminal",
    num_layers: int = 64,
) -> Dict[str, dict]:
    """Compute cooking curve stats for all regions across samples."""
    region_stats = {}

    for region_name in regions:
        all_trajectories = []
        for sample in samples:
            traj = compute_region_attention_per_layer(
                sample, region_name, position, num_layers
            )
            if np.any(traj > 0):
                all_trajectories.append(traj)

        if not all_trajectories:
            continue

        avg_traj = np.mean(all_trajectories, axis=0)
        stats = cooking_curve_stats(avg_traj)
        ratio = (
            stats["peak_value"] / stats["terminal_value"]
            if stats["terminal_value"] > 0
            else float("inf")
        )
        story = _classify_story(stats["peak_layer"], ratio)

        region_stats[region_name] = {
            **stats,
            "peak_terminal_ratio": ratio,
            "story": story,
            "n_samples": len(all_trajectories),
            "trajectory": avg_traj.tolist(),
        }

    return region_stats


def compute_context_bleed(
    samples: List[dict],
    conv_region: str = "conversation_turns",
    curr_region: str = "current_message",
    position: str = "terminal",
    num_layers: int = 64,
) -> dict:
    """Compute context bleed ratio."""
    ratios = []
    conv_attns = []
    curr_attns = []

    for sample in samples:
        conv = compute_region_attention_per_layer(
            sample, conv_region, position, num_layers
        )
        curr = compute_region_attention_per_layer(
            sample, curr_region, position, num_layers
        )

        conv_terminal = conv[-1]
        curr_terminal = curr[-1]

        if curr_terminal > 0:
            ratios.append(conv_terminal / curr_terminal)
        conv_attns.append(conv_terminal)
        curr_attns.append(curr_terminal)

    return {
        "mean_ratio": float(np.mean(ratios)) if ratios else 0,
        "median_ratio": float(np.median(ratios)) if ratios else 0,
        "pct_above_2x": float(np.mean([r > 2 for r in ratios]) * 100) if ratios else 0,
        "conv_turns_mean": float(np.mean(conv_attns)),
        "current_message_mean": float(np.mean(curr_attns)),
        "n_samples": len(ratios),
    }


def write_experiment_report(
    exp_key: str,
    exp_label: str,
    samples: List[dict],
    regions: List[str],
    baseline_stats: Dict[str, dict],
    baseline_bleed: dict,
    num_layers: int,
    output_dir: Path,
) -> Path:
    """Write a full experiment report to markdown."""
    cooking = compute_cooking_table(samples, regions, num_layers=num_layers)
    bleed = compute_context_bleed(samples, num_layers=num_layers)

    report_path = output_dir / f"{exp_key}.md"
    lines = []
    lines.append(f"# {exp_label}")
    lines.append("")
    lines.append(f"**Samples analyzed**: {len(samples)}")
    lines.append("")

    # Context bleed
    lines.append("## Context Bleed")
    lines.append("")
    lines.append("| Metric | Value | Baseline | Delta |")
    lines.append("|--------|-------|----------|-------|")

    if baseline_bleed:
        bl_mean = baseline_bleed.get("mean_ratio", 0)
        bleed_delta = bleed["mean_ratio"] - bl_mean
        bleed_pct = (bleed_delta / bl_mean * 100) if bl_mean > 0 else 0
        lines.append(
            f"| Mean conv/curr ratio | {bleed['mean_ratio']:.2f}x | "
            f"{bl_mean:.2f}x | {bleed_delta:+.2f}x ({bleed_pct:+.1f}%) |"
        )
        lines.append(
            f"| Median | {bleed['median_ratio']:.2f}x | "
            f"{baseline_bleed.get('median_ratio', 0):.2f}x | "
            f"{bleed['median_ratio'] - baseline_bleed.get('median_ratio', 0):+.2f}x |"
        )
        lines.append(
            f"| Samples >2x | {bleed['pct_above_2x']:.0f}% | "
            f"{baseline_bleed.get('pct_above_2x', 0):.0f}% | "
            f"{bleed['pct_above_2x'] - baseline_bleed.get('pct_above_2x', 0):+.0f}pp |"
        )
    else:
        lines.append(
            f"| Mean conv/curr ratio | {bleed['mean_ratio']:.2f}x | — | — |"
        )
    lines.append("")

    # Cooking curves
    lines.append("## Region Cooking Curves")
    lines.append("")
    lines.append("| Region | Peak Layer | Peak Attn | Terminal Attn | Peak/Terminal | Story |")
    lines.append("|--------|-----------|-----------|-------------|---------------|-------|")
    for region_name in regions:
        if region_name not in cooking:
            continue
        s = cooking[region_name]
        lines.append(
            f"| {region_name} | L{s['peak_layer']:02d} | "
            f"{s['peak_value']:.6f} | {s['terminal_value']:.6f} | "
            f"{s['peak_terminal_ratio']:.1f}x | {s['story']} |"
        )
    lines.append("")

    # Delta from baseline
    if exp_key != "baseline" and baseline_stats:
        lines.append("## Delta from Baseline")
        lines.append("")
        lines.append("| Region | Peak Layer \u0394 | Terminal Attn \u0394 | Interpretation |")
        lines.append("|--------|-----------|--------------| ---------------|")

        for region_name in regions:
            if region_name not in baseline_stats or region_name not in cooking:
                if region_name in baseline_stats and region_name not in cooking:
                    lines.append(f"| {region_name} | \u2014 | \u2014 | **REMOVED** |")
                continue

            b = baseline_stats[region_name]
            e = cooking[region_name]
            layer_delta = e["peak_layer"] - b["peak_layer"]
            term_delta_pct = (
                (e["terminal_value"] - b["terminal_value"]) / b["terminal_value"] * 100
                if b["terminal_value"] > 0
                else 0
            )

            layer_str = f"+{layer_delta}" if layer_delta > 0 else str(layer_delta)
            lines.append(
                f"| {region_name} | {layer_str} | {term_delta_pct:+.1f}% | "
                f"{'Stable' if abs(term_delta_pct) < 10 else ('Gained' if term_delta_pct > 0 else 'Lost')} |"
            )
        lines.append("")

    # Raw trajectories
    lines.append("## Raw Trajectories (JSON)")
    lines.append("")
    lines.append("<details>")
    lines.append("<summary>Click to expand</summary>")
    lines.append("")
    lines.append("```json")
    trajectory_data = {}
    for region_name, stats in cooking.items():
        trajectory_data[region_name] = {
            "peak_layer": stats["peak_layer"],
            "peak_value": stats["peak_value"],
            "terminal_value": stats["terminal_value"],
            "trajectory": [round(v, 8) for v in stats["trajectory"]],
        }
    lines.append(json.dumps(trajectory_data, indent=2))
    lines.append("```")
    lines.append("")
    lines.append("</details>")

    report_path.write_text("\n".join(lines))
    return report_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate markdown experiment reports",
    )
    parser.add_argument(
        "--base-dir", required=True,
        help="Base directory containing experiment result directories",
    )
    parser.add_argument(
        "--experiments", nargs="+", required=True,
        help="Experiment specs as key:label:dirname triples",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Output directory for reports (default: base-dir/reports)",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    output_dir = Path(args.output_dir) if args.output_dir else base_dir / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)

    experiments = []
    for spec in args.experiments:
        parts = spec.split(":")
        if len(parts) == 3:
            experiments.append({"key": parts[0], "label": parts[1], "dirname": parts[2]})
        elif len(parts) == 2:
            experiments.append({"key": parts[0], "label": parts[1], "dirname": parts[0]})
        else:
            experiments.append({"key": parts[0], "label": parts[0], "dirname": parts[0]})

    # Load all experiments
    all_cooking = {}
    all_bleed = {}
    all_samples = {}

    for exp in experiments:
        exp_path = base_dir / exp["dirname"]
        samples = _load_samples(exp_path)
        if not samples:
            print(f"SKIP {exp['key']}: no samples found in {exp_path}")
            continue

        all_samples[exp["key"]] = samples
        regions = _auto_regions(samples)
        num_layers = _detect_num_layers(samples)

        cooking = compute_cooking_table(samples, regions, num_layers=num_layers)
        bleed = compute_context_bleed(samples, num_layers=num_layers)

        all_cooking[exp["key"]] = cooking
        all_bleed[exp["key"]] = bleed
        print(f"  Analyzed {exp['key']}: {len(samples)} samples, {len(regions)} regions")

    # Write reports
    baseline_key = experiments[0]["key"] if experiments else None
    baseline_stats = all_cooking.get(baseline_key, {})
    baseline_bleed = all_bleed.get(baseline_key, {})

    for exp in experiments:
        if exp["key"] not in all_samples:
            continue
        samples = all_samples[exp["key"]]
        regions = _auto_regions(samples)
        num_layers = _detect_num_layers(samples)

        report_path = write_experiment_report(
            exp["key"], exp["label"], samples, regions,
            baseline_stats, baseline_bleed, num_layers, output_dir,
        )
        print(f"  Written: {report_path}")

    print(f"\nAll reports in: {output_dir}")


if __name__ == "__main__":
    main()
