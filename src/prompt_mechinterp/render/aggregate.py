#!/usr/bin/env python3
"""Aggregate cooking curves across all samples in a variant directory.

Averages per-region attention across all samples at each layer, with
optional confidence bands (std). Can overlay multiple variants for comparison.

Usage:
    python -m prompt_mechinterp.render.aggregate --base-dir ./data/results --dirs variant_a
    python -m prompt_mechinterp.render.aggregate --base-dir ./data/results --dirs baseline composite --compare
"""

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np

from PIL import Image, ImageDraw

from ..constants import display_phases, SKIP_REGIONS
from ._shared import BG_COLOR, REGION_COLORS, get_font
from .loaders import load_variant_curves


# Default regions to display (when user doesn't specify)
DEFAULT_REGIONS = [
    "entity_rules", "passage_rules", "expansion_rules", "complexity_rules",
    "directive", "output_format",
    "conversation_turns", "current_message", "stored_passages",
]

# Variant line styles for comparison mode
VARIANT_STYLES = [
    {"dash": None, "alpha": 255},
    {"dash": [8, 4], "alpha": 200},
    {"dash": [3, 3], "alpha": 180},
]


def render_single_variant(
    curves: Dict[str, np.ndarray],
    dirname: str,
    regions: List[str],
    normalize: str,
    output: Path,
    width: int = 1400,
    height: int = 700,
    show_bands: bool = True,
):
    """Render aggregate cooking curve for a single variant with optional std bands."""
    n_layers = next(iter(curves.values())).shape[1] if curves else 64
    n_samples = next(iter(curves.values())).shape[0] if curves else 0

    left, right, top, bottom = 80, 250, 80, 60
    plot_w = width - left - right
    plot_h = height - top - bottom

    img = Image.new("RGB", (width, height), BG_COLOR)
    draw = ImageDraw.Draw(img)

    font = get_font(12)
    font_sm = get_font(10)
    font_title = get_font(16)

    draw.text(
        (10, 8),
        f"Aggregate Cooking Curves \u2014 {dirname}  [n={n_samples}, {normalize}]",
        fill=(220, 220, 220),
        font=font_title,
    )
    draw.text(
        (10, 30),
        "Mean attention per token by region across all layers (shaded = \u00b11 std)",
        fill=(160, 160, 160),
        font=font_sm,
    )

    # Phase labels
    for label, start, end in display_phases(n_layers):
        end_c = min(end, n_layers - 1)
        x_start = left + int(start / (n_layers - 1) * plot_w)
        x_end = left + int(end_c / (n_layers - 1) * plot_w)
        x_mid = (x_start + x_end) // 2
        draw.text(
            (x_mid - len(label) * 3, top - 18),
            label,
            fill=(120, 120, 130),
            font=font_sm,
        )
        draw.line(
            [(x_start, top - 5), (x_start, top + plot_h)],
            fill=(60, 60, 70),
            width=1,
        )

    # Compute means and stds
    region_stats = {}
    for region in regions:
        if region not in curves:
            continue
        arr = curves[region]
        mean = np.mean(arr, axis=0)
        std = (
            np.std(arr, axis=0, ddof=1)
            if arr.shape[0] > 1
            else np.zeros(n_layers)
        )

        if normalize == "per-region":
            peak = np.max(mean)
            if peak > 0:
                std = std / peak
                mean = mean / peak

        region_stats[region] = (mean, std)

    # Y-axis scaling
    y_max = 0
    for region, (mean, std) in region_stats.items():
        y_max = max(y_max, np.max(mean + std))
    y_max *= 1.1

    # Grid lines
    for frac in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        y = (
            top + plot_h - int(frac / y_max * plot_h)
            if y_max > 0
            else top + plot_h
        )
        if 0 <= y - top <= plot_h:
            draw.line([(left, y), (left + plot_w, y)], fill=(50, 50, 60), width=1)
            draw.text(
                (left - 35, y - 6),
                f"{frac:.1f}",
                fill=(120, 120, 130),
                font=font_sm,
            )

    # X-axis labels
    for l in range(0, n_layers, 8):
        x = left + int(l / (n_layers - 1) * plot_w)
        draw.text(
            (x - 8, top + plot_h + 8), f"L{l}", fill=(120, 120, 130), font=font_sm
        )
    draw.text(
        (left + plot_w - 16, top + plot_h + 8),
        f"L{n_layers - 1}",
        fill=(120, 120, 130),
        font=font_sm,
    )

    # Plot bands
    if show_bands:
        for region, (mean, std) in region_stats.items():
            color = REGION_COLORS.get(region, (180, 180, 180))
            band_color = (*color, 40)
            band_img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            band_draw = ImageDraw.Draw(band_img)

            upper_points = []
            lower_points = []
            for l in range(n_layers):
                x = left + int(l / (n_layers - 1) * plot_w)
                y_up = mean[l] + std[l]
                y_lo = mean[l] - std[l]
                yu = (
                    top + plot_h - int(y_up / y_max * plot_h)
                    if y_max > 0
                    else top + plot_h
                )
                yl = (
                    top + plot_h - int(y_lo / y_max * plot_h)
                    if y_max > 0
                    else top + plot_h
                )
                yu = max(top, min(top + plot_h, yu))
                yl = max(top, min(top + plot_h, yl))
                upper_points.append((x, yu))
                lower_points.append((x, yl))

            polygon = upper_points + lower_points[::-1]
            if len(polygon) >= 3:
                band_draw.polygon(polygon, fill=band_color)

            img = Image.alpha_composite(img.convert("RGBA"), band_img).convert("RGB")
            draw = ImageDraw.Draw(img)

    # Plot mean lines
    legend_y = top + 10
    for region, (mean, std) in region_stats.items():
        color = REGION_COLORS.get(region, (180, 180, 180))
        peak_layer = int(np.argmax(mean))

        points = []
        for l in range(n_layers):
            x = left + int(l / (n_layers - 1) * plot_w)
            y = (
                top + plot_h - int(mean[l] / y_max * plot_h)
                if y_max > 0
                else top + plot_h
            )
            y = max(top, min(top + plot_h, y))
            points.append((x, y))

        for i in range(len(points) - 1):
            draw.line([points[i], points[i + 1]], fill=color, width=2)

        px, py = points[peak_layer]
        draw.ellipse([(px - 3, py - 3), (px + 3, py + 3)], fill=color)

        draw.rectangle(
            [(width - 230, legend_y), (width - 218, legend_y + 10)], fill=color
        )
        draw.text(
            (width - 214, legend_y - 1),
            f"{region} (L{peak_layer})",
            fill=color,
            font=font_sm,
        )
        legend_y += 16

    draw.text(
        (left + plot_w // 2 - 20, height - 20),
        "Layer",
        fill=(160, 160, 160),
        font=font,
    )

    img.save(str(output))
    print(f"Saved: {output} ({width}x{height})")


def render_comparison(
    all_curves: Dict[str, Dict[str, np.ndarray]],
    regions: List[str],
    normalize: str,
    output: Path,
    width: int = 1400,
    height: int = 700,
):
    """Render overlaid cooking curves from multiple variants."""
    # Detect num_layers from data
    n_layers = 64
    for curves in all_curves.values():
        for arr in curves.values():
            n_layers = arr.shape[1]
            break
        break

    left, right, top, bottom = 80, 300, 80, 60
    plot_w = width - left - right
    plot_h = height - top - bottom

    img = Image.new("RGB", (width, height), BG_COLOR)
    draw = ImageDraw.Draw(img)

    font = get_font(12)
    font_sm = get_font(10)
    font_title = get_font(16)

    variant_names = list(all_curves.keys())
    draw.text(
        (10, 8),
        f"Cooking Curve Comparison \u2014 {' vs '.join(variant_names)}  [{normalize}]",
        fill=(220, 220, 220),
        font=font_title,
    )
    draw.text(
        (10, 30),
        f"Mean attention per token (solid={variant_names[0] if variant_names else '?'}, "
        f"dashed={variant_names[1] if len(variant_names) > 1 else '?'}, "
        f"dotted={variant_names[2] if len(variant_names) > 2 else '?'})",
        fill=(160, 160, 160),
        font=font_sm,
    )

    for label, start, end in display_phases(n_layers):
        end_c = min(end, n_layers - 1)
        x_start = left + int(start / (n_layers - 1) * plot_w)
        x_end = left + int(end_c / (n_layers - 1) * plot_w)
        x_mid = (x_start + x_end) // 2
        draw.text(
            (x_mid - len(label) * 3, top - 18),
            label,
            fill=(120, 120, 130),
            font=font_sm,
        )
        draw.line(
            [(x_start, top - 5), (x_start, top + plot_h)],
            fill=(60, 60, 70),
            width=1,
        )

    # Compute all means
    all_stats: Dict[str, Dict[str, np.ndarray]] = {}
    y_max = 0
    for vname, curves in all_curves.items():
        all_stats[vname] = {}
        for region in regions:
            if region not in curves:
                continue
            arr = curves[region]
            mean = np.mean(arr, axis=0)
            if normalize == "per-region":
                peak = np.max(mean)
                if peak > 0:
                    mean = mean / peak
            all_stats[vname][region] = mean
            y_max = max(y_max, np.max(mean))
    y_max *= 1.1

    # Grid
    for frac in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        y = (
            top + plot_h - int(frac / y_max * plot_h)
            if y_max > 0
            else top + plot_h
        )
        if 0 <= y - top <= plot_h:
            draw.line([(left, y), (left + plot_w, y)], fill=(50, 50, 60), width=1)
            draw.text(
                (left - 35, y - 6),
                f"{frac:.1f}",
                fill=(120, 120, 130),
                font=font_sm,
            )

    for l in range(0, n_layers, 8):
        x = left + int(l / (n_layers - 1) * plot_w)
        draw.text(
            (x - 8, top + plot_h + 8), f"L{l}", fill=(120, 120, 130), font=font_sm
        )
    draw.text(
        (left + plot_w - 16, top + plot_h + 8),
        f"L{n_layers - 1}",
        fill=(120, 120, 130),
        font=font_sm,
    )

    # Plot lines
    legend_y = top + 10
    for region in regions:
        color = REGION_COLORS.get(region, (180, 180, 180))
        drawn_any = False

        for vi, (vname, stats) in enumerate(all_stats.items()):
            if region not in stats:
                continue
            mean = stats[region]
            style = VARIANT_STYLES[vi % len(VARIANT_STYLES)]

            points = []
            for l in range(n_layers):
                x = left + int(l / (n_layers - 1) * plot_w)
                y = (
                    top + plot_h - int(mean[l] / y_max * plot_h)
                    if y_max > 0
                    else top + plot_h
                )
                y = max(top, min(top + plot_h, y))
                points.append((x, y))

            line_width = 2
            if style["dash"] is None:
                for i in range(len(points) - 1):
                    draw.line(
                        [points[i], points[i + 1]], fill=color, width=line_width
                    )
            else:
                seg_on, seg_off = style["dash"]
                counter = 0
                drawing = True
                for i in range(len(points) - 1):
                    if drawing:
                        draw.line(
                            [points[i], points[i + 1]], fill=color, width=line_width
                        )
                    counter += 1
                    if drawing and counter >= seg_on:
                        drawing = False
                        counter = 0
                    elif not drawing and counter >= seg_off:
                        drawing = True
                        counter = 0

            drawn_any = True

        if drawn_any:
            draw.rectangle(
                [(width - 280, legend_y), (width - 268, legend_y + 10)], fill=color
            )
            draw.text(
                (width - 264, legend_y - 1), region, fill=color, font=font_sm
            )
            legend_y += 16

    # Variant legend
    legend_y += 10
    for vi, vname in enumerate(variant_names):
        style = VARIANT_STYLES[vi % len(VARIANT_STYLES)]
        y = legend_y + vi * 16
        lx = width - 280
        if style["dash"] is None:
            draw.line(
                [(lx, y + 5), (lx + 20, y + 5)], fill=(200, 200, 200), width=2
            )
        else:
            for seg in range(0, 20, sum(style["dash"])):
                draw.line(
                    [(lx + seg, y + 5), (lx + seg + style["dash"][0], y + 5)],
                    fill=(200, 200, 200),
                    width=2,
                )
        draw.text((lx + 26, y - 1), vname, fill=(200, 200, 200), font=font_sm)

    draw.text(
        (left + plot_w // 2 - 20, height - 20),
        "Layer",
        fill=(160, 160, 160),
        font=font,
    )

    img.save(str(output))
    print(f"Saved: {output} ({width}x{height})")


def main():
    parser = argparse.ArgumentParser(description="Aggregate cooking curves")
    parser.add_argument(
        "--base-dir", type=str, required=True, help="Base directory containing variant result dirs"
    )
    parser.add_argument("--dirs", nargs="+", required=True, help="Result directory names")
    parser.add_argument(
        "--regions", type=str, default=None, help="Comma-separated region names"
    )
    parser.add_argument(
        "--normalize",
        choices=["raw", "per-region"],
        default="per-region",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Overlay variants instead of single+bands",
    )
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--width", type=int, default=1400)
    parser.add_argument("--height", type=int, default=700)
    args = parser.parse_args()

    base_path = Path(args.base_dir)
    regions = args.regions.split(",") if args.regions else DEFAULT_REGIONS

    if args.compare or len(args.dirs) > 1:
        all_curves = {}
        for dirname in args.dirs:
            label = dirname.replace("results_", "")
            print(f"Loading {dirname}...")
            all_curves[label] = load_variant_curves(base_path, dirname)
            n = next(iter(all_curves[label].values())).shape[0]
            print(f"  {n} samples loaded")

        output = (
            Path(args.output)
            if args.output
            else base_path
            / f"aggregate_comparison_{'_vs_'.join(d.replace('results_', '') for d in args.dirs)}_{args.normalize}.png"
        )
        render_comparison(
            all_curves, regions, args.normalize, output, args.width, args.height
        )
    else:
        dirname = args.dirs[0]
        print(f"Loading {dirname}...")
        curves = load_variant_curves(base_path, dirname)
        n = next(iter(curves.values())).shape[0]
        print(f"  {n} samples loaded")

        output = (
            Path(args.output)
            if args.output
            else base_path
            / f"aggregate_{dirname.replace('results_', '')}_{args.normalize}.png"
        )
        render_single_variant(
            curves, dirname, regions, args.normalize, output, args.width, args.height
        )


if __name__ == "__main__":
    main()
