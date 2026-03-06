#!/usr/bin/env python3
"""Render animated GIF showing per-token attention sweeping through all layers.

Each frame is a spatial heatmap for a single layer. Watching the animation
reveals forward pass dynamics: rules lighting up early, going dark in the
middle, then current_message and examples blazing late.

Usage:
    python -m prompt_mechinterp.render.layer_gif --result sample_01.json --mask-chatml
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image, ImageDraw

from ..constants import display_phases
from ._shared import (
    BG_COLOR,
    CHATML_COLOR,
    LEFT_MARGIN,
    LINE_HEIGHT,
    LINE_SPACING,
    RIGHT_MARGIN,
    SECTION_GAP,
    TOKEN_PAD_X,
    COLORMAPS,
    colormap_lookup,
    draw_gradient_rect,
    gaussian_smooth,
    get_colormap,
    get_font,
    layout_tokens,
    normalize_weights,
)
from .loaders import load_all_layers


def render_single_layer_frame(
    token_labels: List[str],
    weights: np.ndarray,
    region_map: Dict,
    piece_boundaries: Dict,
    layer: int,
    width: int,
    smoothing: float,
    colormap_name: str,
    mask_chatml: bool,
    clip_low: float,
    position: str,
    result_path: str,
    target_height: int,
    num_layers: int = 64,
) -> Image.Image:
    """Render a single-layer heatmap frame for the GIF."""
    lut = get_colormap(colormap_name)
    font = get_font(12)

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
        False, font, content_width,
    )

    idx_to_rect = {r["token_idx"]: r for r in token_rects}

    img = Image.new("RGB", (width, target_height), BG_COLOR)
    draw = ImageDraw.Draw(img)

    # Layer label
    font_layer = get_font(18)
    draw.text(
        (LEFT_MARGIN, 8),
        f"Layer {layer}/{num_layers - 1}",
        fill="#dddddd",
        font=font_layer,
    )

    # Phase label
    font_phase = get_font(11)
    phase = "Unknown"
    phase_color = "#888888"
    phases = display_phases(num_layers)
    phase_colors = ["#78cdc4", "#888888", "#ff6b6b", "#ffe66d"]
    for i, (label, l_start, l_end) in enumerate(phases):
        if l_start <= layer <= l_end:
            phase = label
            phase_color = phase_colors[min(i, len(phase_colors) - 1)]
            break
    draw.text((LEFT_MARGIN + 160, 12), phase, fill=phase_color, font=font_phase)

    # Case name
    case_name = Path(result_path).stem
    font_case = get_font(10)
    draw.text(
        (width - RIGHT_MARGIN - 200, 12),
        f"{case_name}  pos={position}",
        fill="#666666",
        font=font_case,
    )

    # Draw tokens
    body_y_offset = 36

    piece_starts = {}
    for piece_name, info in piece_boundaries.items():
        piece_starts[info["tok_start"]] = piece_name

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

    return img


def main():
    parser = argparse.ArgumentParser(
        description="Render animated GIF of attention sweeping through all layers",
    )
    parser.add_argument("--result", required=True, help="Path to result JSON")
    parser.add_argument("--output", default=None, help="Output GIF path")
    parser.add_argument(
        "--position", default="terminal", help="Query position (default: terminal)"
    )
    parser.add_argument(
        "--smoothing", type=float, default=2.0, help="Gaussian smoothing sigma"
    )
    parser.add_argument("--width", type=int, default=2000, help="Image width")
    parser.add_argument(
        "--colormap", default="inferno",
        help=f"Colormap (default: inferno)",
    )
    parser.add_argument(
        "--mask-chatml", action="store_true", help="Mask ChatML tokens"
    )
    parser.add_argument(
        "--clip-low", type=float, default=5.0, help="Floor trim percentage"
    )
    parser.add_argument("--fps", type=int, default=6, help="Frames per second")
    parser.add_argument("--stride", type=int, default=1, help="Layer stride")
    args = parser.parse_args()

    if args.output is None:
        result_path = Path(args.result)
        case_name = result_path.stem
        args.output = str(
            result_path.parent / f"layersweep_{case_name}_{args.position}.gif"
        )

    print(f"Loading: {args.result}")
    token_labels, layer_weights, region_map, piece_boundaries = load_all_layers(
        args.result, args.position,
    )

    layers_sorted = sorted(layer_weights.keys())
    num_layers = len(layers_sorted)
    layers_to_render = layers_sorted[:: args.stride]
    print(f"  {len(token_labels)} tokens, {num_layers} layers")
    print(f"  Rendering {len(layers_to_render)} frames (stride={args.stride})")

    # Pre-compute target height from first frame layout
    lut = get_colormap(args.colormap)
    font = get_font(12)
    content_width = args.width - LEFT_MARGIN - RIGHT_MARGIN

    first_weights = layer_weights[layers_to_render[0]]
    smoothed = gaussian_smooth(first_weights, args.smoothing)
    normed = normalize_weights(smoothed, clip_low=args.clip_low)
    colors = colormap_lookup(normed, lut)
    _, body_height = layout_tokens(
        token_labels, colors, piece_boundaries, region_map,
        False, font, content_width,
    )
    target_height = 36 + body_height + 20

    print(f"  Frame size: {args.width}x{target_height}")

    frames = []
    for i, layer in enumerate(layers_to_render):
        weights = layer_weights[layer]
        frame = render_single_layer_frame(
            token_labels, weights, region_map, piece_boundaries,
            layer=layer,
            width=args.width,
            smoothing=args.smoothing,
            colormap_name=args.colormap,
            mask_chatml=args.mask_chatml,
            clip_low=args.clip_low,
            position=args.position,
            result_path=args.result,
            target_height=target_height,
            num_layers=num_layers,
        )
        frames.append(frame)
        print(f"  Frame {i + 1}/{len(layers_to_render)}: Layer {layer}", end="\r")

    print(f"\n  Assembling GIF ({len(frames)} frames at {args.fps} fps)...")

    duration_ms = int(1000 / args.fps)
    frames[0].save(
        args.output,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )

    size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(
        f"Saved: {args.output} ({size_mb:.1f} MB, {len(frames)} frames, "
        f"{args.fps} fps)"
    )


if __name__ == "__main__":
    main()
