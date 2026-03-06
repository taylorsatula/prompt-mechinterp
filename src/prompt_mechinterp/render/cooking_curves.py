#!/usr/bin/env python3
"""Render per-region attention trajectory ("cooking curves") across all layers.

Shows how each named region's attention evolves through the full forward pass.

Usage:
    python -m prompt_mechinterp.render.cooking_curves --result sample_01.json
    python -m prompt_mechinterp.render.cooking_curves --result sample_01.json --normalize per-region
"""

import argparse
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

from ..constants import DISPLAY_PHASES, SKIP_REGIONS, DEFAULT_DISPLAY_REGIONS
from ._shared import (
    BG_COLOR,
    GRID_COLOR,
    AXIS_COLOR,
    TEXT_COLOR,
    TEXT_DIM,
    REGION_PALETTE,
    get_font,
)
from .loaders import load_cooking_data


# Chart layout
CHART_LEFT = 100
CHART_TOP = 70
CHART_RIGHT = 40
CHART_BOTTOM = 60
LEGEND_WIDTH = 220
LINE_WIDTH = 2


def compute_region_trajectories(
    region_map: Dict[str, Dict],
    layer_weights: Dict[int, np.ndarray],
    regions: List[str],
) -> Dict[str, np.ndarray]:
    """Compute mean attention per token for each region at each layer.

    Returns:
        trajectories: region_name -> array of shape (n_layers,)
    """
    n_layers = len(layer_weights)
    layers_sorted = sorted(layer_weights.keys())

    trajectories = {}
    for region in regions:
        if region not in region_map:
            continue
        info = region_map[region]
        ts, te = info["tok_start"], info["tok_end"]
        if te <= ts:
            continue

        curve = np.zeros(n_layers)
        for i, layer in enumerate(layers_sorted):
            w = layer_weights[layer]
            curve[i] = np.mean(w[ts:te])
        trajectories[region] = curve

    return trajectories


def _nice_ticks(vmin: float, vmax: float, n_ticks: int = 5) -> List[float]:
    """Generate human-readable tick values for an axis range."""
    if vmax <= vmin:
        return [vmin]
    raw_step = (vmax - vmin) / n_ticks
    magnitude = 10 ** math.floor(math.log10(raw_step))
    residual = raw_step / magnitude
    if residual <= 1.5:
        nice_step = magnitude
    elif residual <= 3.5:
        nice_step = 2 * magnitude
    elif residual <= 7.5:
        nice_step = 5 * magnitude
    else:
        nice_step = 10 * magnitude

    start = math.floor(vmin / nice_step) * nice_step
    ticks = []
    val = start
    while val <= vmax + nice_step * 0.01:
        if val >= vmin - nice_step * 0.01:
            ticks.append(val)
        val += nice_step
    return ticks


def render_cooking_curves(
    trajectories: Dict[str, np.ndarray],
    position: str,
    result_path: str,
    width: int = 1400,
    height: int = 700,
    normalize_mode: str = "raw",
    highlight: Optional[List[str]] = None,
) -> Image.Image:
    """Render the cooking curve chart as a PIL Image."""
    img = Image.new("RGB", (width, height), BG_COLOR)
    draw = ImageDraw.Draw(img)

    font_title = get_font(14)
    font_label = get_font(11)
    font_tick = get_font(10)
    font_legend = get_font(10)

    chart_x0 = CHART_LEFT
    chart_y0 = CHART_TOP
    chart_x1 = width - CHART_RIGHT - LEGEND_WIDTH
    chart_y1 = height - CHART_BOTTOM
    chart_w = chart_x1 - chart_x0
    chart_h = chart_y1 - chart_y0

    case_name = Path(result_path).stem
    title = f"Attention Cooking Curves \u2014 {case_name} ({position})"
    if normalize_mode == "per-region":
        title += "  [per-region normalized]"
    draw.text((chart_x0, 15), title, fill=TEXT_COLOR, font=font_title)

    subtitle = "Mean attention per token by region across all layers"
    draw.text((chart_x0, 35), subtitle, fill=TEXT_DIM, font=font_tick)

    region_names = list(trajectories.keys())
    n_layers = len(next(iter(trajectories.values())))

    plot_data = {}
    for name, curve in trajectories.items():
        if normalize_mode == "per-region":
            peak = curve.max()
            plot_data[name] = curve / peak if peak > 0 else curve
        else:
            plot_data[name] = curve

    all_vals = np.concatenate(list(plot_data.values()))
    y_min = 0.0
    y_max = float(np.max(all_vals)) * 1.1

    x_ticks = list(range(0, n_layers, 8))
    if (n_layers - 1) not in x_ticks:
        x_ticks.append(n_layers - 1)
    y_ticks = _nice_ticks(y_min, y_max, 6)

    def to_px(layer: int, val: float) -> Tuple[int, int]:
        x = chart_x0 + int(layer / max(1, n_layers - 1) * chart_w)
        y = chart_y1 - int((val - y_min) / max(1e-15, y_max - y_min) * chart_h)
        return x, y

    for layer in x_ticks:
        px, _ = to_px(layer, 0)
        draw.line([(px, chart_y0), (px, chart_y1)], fill=GRID_COLOR, width=1)
        draw.text((px - 4, chart_y1 + 6), f"L{layer}", fill=TEXT_DIM, font=font_tick)

    for val in y_ticks:
        _, py = to_px(0, val)
        if chart_y0 <= py <= chart_y1:
            draw.line([(chart_x0, py), (chart_x1, py)], fill=GRID_COLOR, width=1)
            if normalize_mode == "per-region":
                label = f"{val:.1f}"
            else:
                label = f"{val:.1e}" if val != 0 else "0"
            draw.text((chart_x0 - 70, py - 6), label, fill=TEXT_DIM, font=font_tick)

    draw.rectangle([chart_x0, chart_y0, chart_x1, chart_y1], outline=AXIS_COLOR)

    draw.text(
        (chart_x0 + chart_w // 2 - 20, chart_y1 + 25),
        "Layer",
        fill=TEXT_COLOR,
        font=font_label,
    )

    # Phase annotations
    for label, l_start, l_end in DISPLAY_PHASES:
        if l_start >= n_layers:
            break
        l_end_clamped = min(l_end, n_layers - 1)
        px_start, _ = to_px(l_start, 0)
        px_end, _ = to_px(l_end_clamped, 0)
        mid = (px_start + px_end) // 2
        bbox = font_tick.getbbox(label)
        text_w = bbox[2] - bbox[0]
        draw.text(
            (mid - text_w // 2, chart_y0 - 15),
            label,
            fill=(100, 100, 110),
            font=font_tick,
        )
        if l_start > 0:
            draw.line(
                [(px_start, chart_y0), (px_start, chart_y0 + 6)],
                fill=(80, 80, 90),
                width=1,
            )

    # Draw curves
    dimmed = highlight is not None

    for idx, name in enumerate(region_names):
        curve = plot_data[name]
        color = REGION_PALETTE[idx % len(REGION_PALETTE)]

        if dimmed and name not in highlight:
            color = tuple(c // 4 for c in color)
            lw = 1
        else:
            lw = LINE_WIDTH

        points = [to_px(layer, curve[layer]) for layer in range(n_layers)]
        for i in range(len(points) - 1):
            draw.line([points[i], points[i + 1]], fill=color, width=lw)

        peak_layer = int(np.argmax(curve))
        px, py = to_px(peak_layer, curve[peak_layer])
        if not dimmed or name in (highlight or []):
            draw.ellipse([px - 3, py - 3, px + 3, py + 3], fill=color)

    # Legend
    legend_x = chart_x1 + 20
    legend_y = chart_y0 + 5
    draw.text((legend_x, legend_y - 2), "Regions", fill=TEXT_COLOR, font=font_label)
    legend_y += 20

    sorted_regions = sorted(
        region_names, key=lambda n: int(np.argmax(plot_data[n]))
    )

    for name in sorted_regions:
        idx = region_names.index(name)
        color = REGION_PALETTE[idx % len(REGION_PALETTE)]
        if dimmed and name not in (highlight or []):
            color = tuple(c // 4 for c in color)

        peak_layer = int(np.argmax(trajectories[name]))
        short_name = name[:20]
        draw.rectangle(
            [legend_x, legend_y + 2, legend_x + 10, legend_y + 12], fill=color
        )
        draw.text(
            (legend_x + 14, legend_y),
            f"{short_name} (L{peak_layer})",
            fill=color,
            font=font_legend,
        )
        legend_y += 16

    draw.text(
        (chart_x0, height - 18),
        "Peak layer shown in legend. Dot marks each region's peak.",
        fill=TEXT_DIM,
        font=font_tick,
    )

    return img


def main():
    parser = argparse.ArgumentParser(
        description="Render per-region attention cooking curves across all layers",
    )
    parser.add_argument("--result", required=True, help="Path to result JSON")
    parser.add_argument("--output", default=None, help="Output PNG path")
    parser.add_argument(
        "--position", default="terminal", help="Query position (default: terminal)"
    )
    parser.add_argument("--width", type=int, default=1400, help="Image width")
    parser.add_argument("--height", type=int, default=700, help="Image height")
    parser.add_argument(
        "--normalize",
        choices=["raw", "per-region"],
        default="raw",
        help="Normalization: raw or per-region (each line 0-1)",
    )
    parser.add_argument(
        "--regions", default=None, help="Comma-separated region names to plot"
    )
    parser.add_argument(
        "--highlight", default=None, help="Comma-separated region names to highlight"
    )
    args = parser.parse_args()

    if args.output is None:
        result_path = Path(args.result)
        case_name = result_path.stem
        args.output = str(
            result_path.parent / f"cooking_{case_name}_{args.position}.png"
        )

    print(f"Loading: {args.result}")
    region_map, layer_weights, token_labels = load_cooking_data(
        args.result, args.position
    )
    print(
        f"  {len(token_labels)} tokens, {len(layer_weights)} layers, "
        f"{len(region_map)} regions"
    )

    if args.regions:
        regions = [r.strip() for r in args.regions.split(",")]
    else:
        regions = [r for r in region_map if r not in SKIP_REGIONS]

        def sort_key(name):
            try:
                return DEFAULT_DISPLAY_REGIONS.index(name)
            except ValueError:
                return 999

        regions.sort(key=sort_key)

    trajectories = compute_region_trajectories(region_map, layer_weights, regions)
    print(f"  Plotting {len(trajectories)} regions")

    for name, curve in sorted(
        trajectories.items(), key=lambda x: int(np.argmax(x[1]))
    ):
        peak_l = int(np.argmax(curve))
        peak_v = float(curve.max())
        terminal_v = float(np.mean(curve[-4:]))
        ratio = peak_v / terminal_v if terminal_v > 0 else float("inf")
        print(
            f"    {name:25s}  peak L{peak_l:02d} ({peak_v:.6f})  "
            f"terminal ({terminal_v:.6f})  ratio {ratio:.1f}x"
        )

    highlight = None
    if args.highlight:
        highlight = [h.strip() for h in args.highlight.split(",")]

    img = render_cooking_curves(
        trajectories,
        position=args.position,
        result_path=args.result,
        width=args.width,
        height=args.height,
        normalize_mode=args.normalize,
        highlight=highlight,
    )

    img.save(args.output)
    print(f"Saved: {args.output} ({img.width}x{img.height})")


if __name__ == "__main__":
    main()
