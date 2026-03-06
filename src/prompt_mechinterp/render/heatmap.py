#!/usr/bin/env python3
"""Render per-token attention heatmap from MI analysis results.

Produces a PNG showing the full token sequence with each token's background
colored by attention intensity. Uses rank-based normalization (histogram
equalization) to handle power-law skew in attention distributions.

Usage:
    python -m prompt_mechinterp.render.heatmap --result sample_01.json --mask-chatml
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw

from ._shared import (
    BG_COLOR,
    CHATML_COLOR,
    LEFT_MARGIN,
    RIGHT_MARGIN,
    SECTION_GAP,
    TOKEN_PAD_X,
    TOP_MARGIN,
    COLORMAPS,
    colormap_lookup,
    draw_gradient_rect,
    gaussian_smooth,
    get_colormap,
    get_font,
    layout_tokens,
    normalize_weights,
)
from .loaders import load_heatmap_data


LEGEND_TOP_PAD = 30
LEGEND_ITEM_HEIGHT = 20


def _draw_legend(
    draw: ImageDraw.ImageDraw,
    lut: np.ndarray,
    y_start: int,
    width: int,
    position: str,
    layer_spec: str,
    smoothing: float,
    raw_min: float,
    raw_max: float,
    show_regions: bool,
    n_tokens: int,
    colormap_name: str,
) -> int:
    """Draw the interpretive key/legend at the bottom of the image."""
    font = get_font(11)
    font_small = get_font(10)
    x = LEFT_MARGIN
    y = y_start

    draw.line([(x, y), (width - RIGHT_MARGIN, y)], fill="#444444", width=1)
    y += 10

    draw.text((x, y), "KEY", fill="#aaaaaa", font=get_font(13))
    y += 22

    # Colorbar
    cb_width = 300
    cb_height = 16
    draw.text((x, y + 1), "Attention:", fill="#999999", font=font)
    cb_x = x + 80
    for i in range(cb_width):
        t = i / (cb_width - 1)
        idx = min(255, int(t * 255))
        r, g, b = int(lut[idx][0]), int(lut[idx][1]), int(lut[idx][2])
        draw.line([(cb_x + i, y), (cb_x + i, y + cb_height)], fill=(r, g, b))
    draw.rectangle(
        [cb_x - 1, y - 1, cb_x + cb_width, y + cb_height], outline="#555555"
    )
    draw.text((cb_x - 1, y + cb_height + 2), "Low", fill="#777777", font=font_small)
    draw.text(
        (cb_x + cb_width - 18, y + cb_height + 2),
        "High",
        fill="#777777",
        font=font_small,
    )
    draw.text(
        (cb_x + cb_width + 12, y + 1),
        f"(raw range: {raw_min:.2e} \u2013 {raw_max:.2e}, rank-normalized)",
        fill="#666666",
        font=font_small,
    )
    y += cb_height + 18

    draw.rectangle([x, y + 4, x + 20, y + 6], fill="#ff4444")
    draw.text(
        (x + 28, y),
        "Section boundary  \u2014  divides system_prompt / user_message / response",
        fill="#999999",
        font=font,
    )
    y += LEGEND_ITEM_HEIGHT

    if show_regions:
        draw.rectangle([x, y + 2, x + 1, y + 14], fill="#00ff88")
        draw.text(
            (x + 28, y),
            "Region boundary  \u2014  annotated sub-regions within each section",
            fill="#999999",
            font=font,
        )
        y += LEGEND_ITEM_HEIGHT

    draw.text((x + 2, y), "\\n", fill="#cccccc", font=font)
    draw.text(
        (x + 28, y),
        "Newline token  \u2014  fills remaining line width; color shows its attention weight",
        fill="#999999",
        font=font,
    )
    y += LEGEND_ITEM_HEIGHT

    y += 6
    draw.text((x, y), "Parameters", fill="#aaaaaa", font=get_font(12))
    y += 18
    params = [
        f"Query position: {position}",
        f"Layers averaged: {layer_spec}",
        f"Smoothing sigma: {smoothing}" if smoothing > 0 else "Smoothing: none",
        f"Colormap: {colormap_name}",
        f"Total tokens: {n_tokens}",
    ]
    for param in params:
        draw.text((x + 8, y), param, fill="#888888", font=font_small)
        y += 16

    y += 10
    return y - y_start


def render_heatmap(
    token_labels: List[str],
    weights: np.ndarray,
    region_map: Dict,
    piece_boundaries: Dict,
    width: int,
    smoothing: float,
    colormap_name: str,
    show_regions: bool,
    position: str,
    layer_spec: str,
    result_path: str,
    clip_low: float = 5.0,
    mask_chatml: bool = False,
) -> Image.Image:
    """Render the full attention heatmap as a PIL Image."""
    lut = get_colormap(colormap_name)
    font = get_font(12)

    raw_min, raw_max = float(weights.min()), float(weights.max())

    # Build ChatML mask
    chatml_mask = None
    chatml_indices = set()
    if mask_chatml:
        chatml_mask = np.array([
            not (lab.startswith("<|im_") or lab.endswith("|>") and "im" in lab)
            for lab in token_labels
        ])
        chatml_indices = {i for i, m in enumerate(chatml_mask) if not m}

    smoothed = gaussian_smooth(weights, smoothing)
    normed = normalize_weights(smoothed, clip_low=clip_low, mask=chatml_mask)
    colors = colormap_lookup(normed, lut)

    if mask_chatml:
        for i in chatml_indices:
            if i < len(colors):
                colors[i] = CHATML_COLOR

    content_width = width - LEFT_MARGIN - RIGHT_MARGIN

    token_rects, body_height = layout_tokens(
        token_labels, colors, piece_boundaries, region_map,
        show_regions, font, content_width,
    )

    idx_to_rect = {r["token_idx"]: r for r in token_rects}
    legend_estimate = 220

    total_height = TOP_MARGIN + body_height + LEGEND_TOP_PAD + legend_estimate
    img = Image.new("RGB", (width, int(total_height)), BG_COLOR)
    draw = ImageDraw.Draw(img)

    # Title
    font_title = get_font(14)
    case_name = Path(result_path).stem
    title = f"Attention Heatmap \u2014 {case_name}"
    draw.text((LEFT_MARGIN, 12), title, fill="#dddddd", font=font_title)

    subtitle = f"position={position}  layers={layer_spec}"
    if smoothing > 0:
        subtitle += f"  smoothing={smoothing}"
    draw.text((LEFT_MARGIN, 30), subtitle, fill="#888888", font=get_font(10))

    # Pre-compute piece/region starts
    piece_starts = {}
    for piece_name, info in piece_boundaries.items():
        piece_starts[info["tok_start"]] = piece_name

    region_starts: Dict[int, str] = {}
    if show_regions:
        for region_name, info in region_map.items():
            if region_name in (
                "system_prompt", "user_message", "response", "chat_template"
            ):
                continue
            region_starts[info["tok_start"]] = region_name

    # Draw tokens
    body_y_offset = TOP_MARGIN
    section_lines_drawn: set = set()

    for rect in token_rects:
        rx = LEFT_MARGIN + rect["x"]
        ry = body_y_offset + rect["y"]
        rw = rect["w"]
        rh = rect["h"]
        idx = rect["token_idx"]

        if idx in piece_starts and idx > 0 and idx not in section_lines_drawn:
            section_lines_drawn.add(idx)
            sep_y = ry - SECTION_GAP // 2
            draw.line(
                [(LEFT_MARGIN, sep_y), (width - RIGHT_MARGIN, sep_y)],
                fill="#ff4444",
                width=2,
            )
            label = piece_starts[idx].replace("_", " ")
            draw.text(
                (LEFT_MARGIN + 4, sep_y - 14),
                f"\u25b6 {label}",
                fill="#ff6666",
                font=get_font(10),
            )

        if show_regions and idx in region_starts:
            draw.line([(rx, ry), (rx, ry + rh)], fill="#00ff88", width=1)
            if rect["x"] < 2:
                draw.text(
                    (rx + 3, ry - 12),
                    region_starts[idx][:25],
                    fill="#00cc66",
                    font=get_font(9),
                )

        own_color = rect["color"]

        prev_rect = idx_to_rect.get(idx - 1)
        if prev_rect is not None and prev_rect["y"] == rect["y"]:
            pc = prev_rect["color"]
            left_color = (
                (pc[0] + own_color[0]) // 2,
                (pc[1] + own_color[1]) // 2,
                (pc[2] + own_color[2]) // 2,
            )
        else:
            left_color = own_color

        next_rect = idx_to_rect.get(idx + 1)
        if next_rect is not None and next_rect["y"] == rect["y"]:
            nc = next_rect["color"]
            right_color = (
                (nc[0] + own_color[0]) // 2,
                (nc[1] + own_color[1]) // 2,
                (nc[2] + own_color[2]) // 2,
            )
        else:
            right_color = own_color

        draw_gradient_rect(
            draw, int(rx), int(ry), int(rw), int(rh), left_color, right_color
        )

        draw.text(
            (rx + TOKEN_PAD_X, ry + 2),
            rect["text"],
            fill=rect["fg"],
            font=font,
        )

    # Legend
    legend_y = body_y_offset + body_height + LEGEND_TOP_PAD
    legend_height = _draw_legend(
        draw, lut, legend_y, width,
        position, layer_spec, smoothing,
        raw_min, raw_max, show_regions,
        len(token_labels), colormap_name,
    )

    actual_height = legend_y + legend_height + 20
    if actual_height < total_height:
        img = img.crop((0, 0, width, int(actual_height)))

    return img


def main():
    parser = argparse.ArgumentParser(
        description="Render per-token attention heatmap from MI analysis results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--result", required=True,
        help="Path to result JSON (must include per-token data)",
    )
    parser.add_argument("--output", default=None, help="Output PNG file path")
    parser.add_argument(
        "--position", default="terminal", help="Query position (default: terminal)"
    )
    parser.add_argument(
        "--layers", default="final",
        help="Layer range to average: 'final' (default, last N layers), 'all', '48', '60-63'",
    )
    parser.add_argument(
        "--smoothing", type=float, default=0.0,
        help="Gaussian smoothing sigma (default: 0 = none)",
    )
    parser.add_argument(
        "--width", type=int, default=1800, help="Image width (default: 1800)"
    )
    parser.add_argument(
        "--colormap", default="inferno",
        help=f"Colormap: {', '.join(COLORMAPS)} (default: inferno)",
    )
    parser.add_argument(
        "--no-regions", action="store_true", help="Hide region boundary lines"
    )
    parser.add_argument(
        "--clip-low", type=float, default=5.0,
        help="Floor trim — bottom X%% of ranked tokens become black (default: 5)",
    )
    parser.add_argument(
        "--mask-chatml", action="store_true",
        help="Exclude ChatML tokens from ranking, render as neutral gray",
    )
    args = parser.parse_args()

    if args.output is None:
        result_path = Path(args.result)
        case_name = result_path.stem
        args.output = str(
            result_path.parent / f"heatmap_{case_name}_{args.position}.png"
        )

    print(f"Loading: {args.result}")
    token_labels, weights, region_map, piece_boundaries = load_heatmap_data(
        args.result, args.position, args.layers,
    )
    print(f"  {len(token_labels)} tokens, {len(region_map)} regions")
    print(f"  Attention range: [{weights.min():.6f}, {weights.max():.6f}]")

    img = render_heatmap(
        token_labels, weights, region_map, piece_boundaries,
        width=args.width,
        smoothing=args.smoothing,
        colormap_name=args.colormap,
        show_regions=not args.no_regions,
        position=args.position,
        layer_spec=args.layers,
        result_path=args.result,
        clip_low=args.clip_low,
        mask_chatml=args.mask_chatml,
    )

    img.save(args.output)
    print(f"Saved: {args.output} ({img.width}x{img.height})")


if __name__ == "__main__":
    main()
