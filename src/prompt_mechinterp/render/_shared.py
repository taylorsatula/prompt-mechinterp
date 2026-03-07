"""Shared rendering utilities — fonts, colormaps, normalization, layout.

Extracted from render_heatmap.py (primary donor) with duplicates reconciled
from render_cooking_curves.py and render_aggregate_cooking.py.
"""

from typing import Dict, List, Tuple

import numpy as np
from PIL import ImageDraw, ImageFont


# ============================================================================
# UI COLORS
# ============================================================================

BG_COLOR = (24, 24, 32)
GRID_COLOR = (50, 50, 60)
AXIS_COLOR = (120, 120, 130)
TEXT_COLOR = (200, 200, 200)
TEXT_DIM = (130, 130, 140)
CHATML_COLOR = (60, 60, 60)

# Layout constants
LINE_HEIGHT = 20
LINE_SPACING = 2
SECTION_GAP = 12
LEFT_MARGIN = 16
RIGHT_MARGIN = 16
TOP_MARGIN = 50
TOKEN_PAD_X = 1


# ============================================================================
# REGION COLOR PALETTES
# ============================================================================

# Positional palette for up to 16 regions — for cooking curves where
# regions are assigned colors by index
REGION_PALETTE = [
    (255, 107, 107),  # coral red
    (78, 205, 196),   # teal
    (255, 230, 109),  # gold
    (162, 155, 254),  # lavender
    (0, 216, 146),    # emerald
    (255, 159, 67),   # orange
    (116, 185, 255),  # sky blue
    (255, 118, 200),  # pink
    (186, 220, 88),   # lime
    (232, 67, 147),   # magenta
    (99, 230, 226),   # cyan
    (253, 203, 110),  # sand
    (108, 92, 231),   # purple
    (223, 249, 251),  # ice
    (250, 177, 160),  # peach
    (129, 236, 236),  # aqua
]

# Named palette for aggregate/comparison charts where regions need
# consistent colors across multiple plots
REGION_COLORS = {
    "directive":           (255, 80, 80),
    "entity_rules":        (0, 220, 200),
    "passage_rules":       (200, 160, 0),
    "expansion_rules":     (140, 120, 220),
    "complexity_rules":    (0, 200, 100),
    "output_format":       (255, 160, 0),
    "conversation_turns":  (100, 160, 255),
    "current_message":     (255, 100, 200),
    "stored_passages":     (180, 200, 0),
    "task_reminders":      (150, 150, 150),
    "expansion_examples":  (200, 100, 255),
}


# ============================================================================
# FONT
# ============================================================================

def get_font(size: int) -> ImageFont.FreeTypeFont:
    """Load a monospace font, falling back to default if unavailable."""
    mono_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
        "/usr/share/fonts/TTF/DejaVuSansMono.ttf",
        "/System/Library/Fonts/Menlo.ttc",
        "/usr/share/fonts/truetype/ubuntu/UbuntuMono-R.ttf",
        "C:/Windows/Fonts/consola.ttf",
    ]
    for path in mono_paths:
        try:
            return ImageFont.truetype(path, size)
        except (IOError, OSError):
            continue
    return ImageFont.load_default()


# ============================================================================
# COLORMAP LOOKUP TABLES
# ============================================================================

def _build_inferno_lut() -> np.ndarray:
    """Inferno colormap: black -> purple -> orange -> yellow."""
    points = [
        (0, 0, 0, 4), (16, 10, 7, 46), (32, 40, 11, 84),
        (48, 72, 12, 104), (64, 101, 21, 110), (80, 128, 37, 104),
        (96, 153, 56, 89), (112, 177, 75, 75), (128, 199, 100, 55),
        (144, 216, 126, 35), (160, 231, 153, 17), (176, 241, 181, 9),
        (192, 246, 210, 23), (208, 248, 235, 57), (224, 248, 251, 105),
        (240, 251, 253, 167), (255, 252, 255, 164),
    ]
    return _interpolate_lut(points)


def _build_viridis_lut() -> np.ndarray:
    """Viridis colormap: dark purple -> teal -> yellow."""
    points = [
        (0, 68, 1, 84), (16, 72, 23, 105), (32, 68, 45, 120),
        (48, 62, 66, 131), (64, 54, 87, 136), (80, 46, 107, 142),
        (96, 38, 126, 142), (112, 31, 146, 140), (128, 31, 163, 134),
        (144, 53, 179, 122), (160, 86, 194, 103), (176, 127, 205, 81),
        (192, 170, 214, 54), (208, 212, 225, 27), (224, 241, 229, 29),
        (240, 253, 231, 37), (255, 253, 231, 37),
    ]
    return _interpolate_lut(points)


def _build_hot_lut() -> np.ndarray:
    """Hot colormap: black -> red -> yellow -> white."""
    lut = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        t = i / 255.0
        if t < 0.375:
            r = t / 0.375
            g, b = 0.0, 0.0
        elif t < 0.75:
            r = 1.0
            g = (t - 0.375) / 0.375
            b = 0.0
        else:
            r, g = 1.0, 1.0
            b = (t - 0.75) / 0.25
        lut[i] = [int(r * 255), int(g * 255), int(b * 255)]
    return lut


def _build_coolwarm_lut() -> np.ndarray:
    """Coolwarm colormap: blue -> white -> red."""
    points = [
        (0, 59, 76, 192), (32, 98, 130, 222), (64, 141, 176, 238),
        (96, 184, 210, 243), (128, 230, 230, 230), (160, 241, 195, 169),
        (192, 230, 145, 105), (224, 209, 88, 58), (255, 180, 4, 38),
    ]
    return _interpolate_lut(points)


def _interpolate_lut(points: list) -> np.ndarray:
    """Interpolate control points into a 256-entry RGB LUT."""
    lut = np.zeros((256, 3), dtype=np.uint8)
    for i in range(len(points) - 1):
        i0, r0, g0, b0 = points[i]
        i1, r1, g1, b1 = points[i + 1]
        n = i1 - i0
        for j in range(n):
            t = j / n
            lut[i0 + j] = [
                int(r0 + (r1 - r0) * t),
                int(g0 + (g1 - g0) * t),
                int(b0 + (b1 - b0) * t),
            ]
    lut[255] = [points[-1][1], points[-1][2], points[-1][3]]
    return lut


COLORMAPS = {
    "inferno": _build_inferno_lut,
    "viridis": _build_viridis_lut,
    "hot": _build_hot_lut,
    "coolwarm": _build_coolwarm_lut,
}


def get_colormap(name: str) -> np.ndarray:
    """Get a 256x3 uint8 colormap LUT by name."""
    builder = COLORMAPS.get(name)
    if builder is None:
        raise ValueError(f"Unknown colormap '{name}'. Available: {', '.join(COLORMAPS)}")
    return builder()


# ============================================================================
# SIGNAL PROCESSING
# ============================================================================

def gaussian_smooth(values: np.ndarray, sigma: float) -> np.ndarray:
    """1D Gaussian smoothing via convolution. Output length always equals input."""
    if sigma <= 0:
        return values
    kernel_size = int(sigma * 6) | 1  # ensure odd
    if kernel_size < 3:
        kernel_size = 3
    # Cap kernel to input length (keep odd) so output length = input length
    if kernel_size > len(values):
        kernel_size = len(values)
        if kernel_size % 2 == 0:
            kernel_size = max(1, kernel_size - 1)
    x = np.arange(kernel_size) - kernel_size // 2
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    return np.convolve(values, kernel, mode="same")


def normalize_weights(
    weights: np.ndarray,
    clip_low: float = 5.0,
    mask: np.ndarray = None,
) -> np.ndarray:
    """Normalize to [0, 1] via rank-based histogram equalization.

    Each token gets a color proportional to its rank among all tokens,
    not its raw attention value. This eliminates hotspot blowouts from
    the power-law attention distribution by giving every percentile
    equal visual weight.

    Args:
        clip_low: Bottom X% of ranks map to 0 (black).
        mask: Boolean array, True = include in ranking. Masked-out tokens
              are excluded from rank calculation and get 0.
    """
    w = weights.copy()

    included_idx = np.where(mask)[0] if mask is not None else np.arange(len(w))
    included_vals = w[included_idx]

    order = np.argsort(included_vals)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(order), dtype=np.float64)

    # Handle ties: average rank for identical values
    sorted_vals = included_vals[order]
    i = 0
    while i < len(sorted_vals):
        j = i + 1
        while j < len(sorted_vals) and sorted_vals[j] == sorted_vals[i]:
            j += 1
        if j > i + 1:
            avg_rank = np.mean(ranks[order[i:j]])
            ranks[order[i:j]] = avg_rank
        i = j

    n = len(ranks)
    normed_included = ranks / max(1, n - 1)

    if clip_low > 0:
        floor = clip_low / 100.0
        normed_included = np.clip(
            (normed_included - floor) / (1.0 - floor), 0.0, 1.0
        )

    normed = np.zeros_like(w)
    normed[included_idx] = normed_included
    return normed


def colormap_lookup(
    normed: np.ndarray, lut: np.ndarray
) -> List[Tuple[int, int, int]]:
    """Map normalized [0,1] values to RGB tuples via LUT."""
    indices = np.clip((normed * 255).astype(int), 0, 255)
    rgb = lut[indices]
    return [(int(r), int(g), int(b)) for r, g, b in rgb]


def parse_layer_spec(spec: str, num_layers: int = 64) -> List[int]:
    """Parse layer specification: 'final', 'all', '48', '0,16,32,48,60-63'."""
    spec = spec.strip().lower()
    if spec == "all":
        return list(range(num_layers))
    if spec == "final":
        from ..constants import FINAL_LAYERS
        start = max(0, num_layers - FINAL_LAYERS)
        return list(range(start, num_layers))

    layers = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            s, e = part.split("-", 1)
            layers.update(range(int(s), int(e) + 1))
        else:
            layers.add(int(part))
    return sorted(layers)


# ============================================================================
# TOKEN DISPLAY
# ============================================================================

def text_color_for_bg(r: int, g: int, b: int) -> str:
    """Choose white or black text for readability against background color."""
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return "#ffffff" if luminance < 140 else "#000000"


def sanitize_token(label: str) -> str:
    """Clean token text for display: replace control chars but keep content readable."""
    cleaned = []
    for ch in label:
        if ch == "\t":
            cleaned.append("  ")
        elif ch == "\r":
            cleaned.append("")
        elif ord(ch) < 32 and ch != "\n":
            cleaned.append(f"\\x{ord(ch):02x}")
        else:
            cleaned.append(ch)
    return "".join(cleaned)


def is_newline_token(label: str) -> bool:
    """Check if a token represents a line break."""
    stripped = label.strip("\r")
    return stripped == "\n"


# ============================================================================
# LAYOUT ENGINE
# ============================================================================

def layout_tokens(
    token_labels: List[str],
    colors: List[Tuple[int, int, int]],
    piece_boundaries: Dict,
    region_map: Dict,
    show_regions: bool,
    font: ImageFont.FreeTypeFont,
    content_width: int,
) -> Tuple[List[dict], int]:
    """Compute x/y positions for every token using natural text flow.

    Tokens are placed left-to-right. When a token would overflow the line,
    it wraps to the next line. Newline tokens force a line break. Piece
    boundaries insert a labeled separator with extra vertical space.

    Returns:
        token_rects: list of dicts with keys x, y, w, h, color, fg, text, token_idx
        total_height: total content height in pixels
    """
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

    rects = []
    cursor_x = 0.0
    cursor_y = 0.0

    for i, raw_label in enumerate(token_labels):
        if i in piece_starts and i > 0:
            if cursor_x > 0:
                cursor_y += LINE_HEIGHT + LINE_SPACING
            cursor_y += SECTION_GAP
            cursor_x = 0.0

        label = sanitize_token(raw_label)

        if is_newline_token(raw_label):
            remaining = content_width - cursor_x
            w = max(remaining, LINE_HEIGHT)

            fg = text_color_for_bg(*colors[i])
            rects.append({
                "x": cursor_x, "y": cursor_y,
                "w": w, "h": LINE_HEIGHT,
                "color": colors[i], "fg": fg,
                "text": "\\n", "token_idx": i,
            })

            cursor_y += LINE_HEIGHT + LINE_SPACING
            cursor_x = 0.0
            continue

        if label:
            bbox = font.getbbox(label)
            text_w = bbox[2] - bbox[0]
        else:
            text_w = 4

        w = text_w + TOKEN_PAD_X * 2

        if cursor_x > 0 and cursor_x + w > content_width:
            cursor_y += LINE_HEIGHT + LINE_SPACING
            cursor_x = 0.0

        fg = text_color_for_bg(*colors[i])
        rects.append({
            "x": cursor_x, "y": cursor_y,
            "w": w, "h": LINE_HEIGHT,
            "color": colors[i], "fg": fg,
            "text": label, "token_idx": i,
        })

        cursor_x += w

    total_height = cursor_y + LINE_HEIGHT
    return rects, int(total_height)


def draw_gradient_rect(
    draw: ImageDraw.ImageDraw,
    x: int, y: int, w: int, h: int,
    color_left: Tuple[int, int, int],
    color_right: Tuple[int, int, int],
):
    """Draw a rectangle with a horizontal gradient between two colors."""
    w_int = max(1, int(w))
    for col in range(w_int):
        t = col / max(1, w_int - 1)
        r = int(color_left[0] + (color_right[0] - color_left[0]) * t)
        g = int(color_left[1] + (color_right[1] - color_left[1]) * t)
        b = int(color_left[2] + (color_right[2] - color_left[2]) * t)
        draw.line([(x + col, y), (x + col, y + h)], fill=(r, g, b))
