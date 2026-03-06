"""Shared constants reconciled across all analysis and rendering modules.

NUM_LAYERS and NUM_QUERY_HEADS are intentionally absent — these are model-specific
and must come from model auto-discovery (engine/model_adapter.py), not constants.
"""

# How many final layers to average for "what the model decided" metrics.
# L60-63 for a 64-layer model; for other models, use the last FINAL_LAYERS layers.
FINAL_LAYERS = 4

# Display phases — 4-phase model for chart labels and phase annotations.
# Proportions empirically derived from Qwen3-32B cooking curves.
# Scaled to any layer count via display_phases(num_layers).
_DISPLAY_PHASE_FRACTIONS = [
    ("Rules absorbed", 0.0, 0.125),
    ("Internal computation", 0.125, 0.5),
    ("Focus narrows", 0.5, 0.75),
    ("Output formatting", 0.75, 1.0),
]


def display_phases(num_layers: int) -> list:
    """Return display phases scaled to the given layer count."""
    last = num_layers - 1
    return [
        (label, round(start * last), round(end * last))
        for label, start, end in _DISPLAY_PHASE_FRACTIONS
    ]


# Analysis phases — 5-phase model for detailed per-phase metrics.
# Proportions scaled to any layer count via analysis_phases(num_layers).
_ANALYSIS_PHASE_FRACTIONS = {
    "P1_broad_read": (0.0, 0.094),
    "P2_absorption": (0.109, 0.172),
    "P3_compression": (0.188, 0.484),
    "P4_reengagement": (0.5, 0.734),
    "P5_output_prep": (0.75, 1.0),
}


def analysis_phases(num_layers: int) -> dict:
    """Return analysis phases scaled to the given layer count."""
    last = num_layers - 1
    return {
        name: (round(start * last), round(end * last))
        for name, (start, end) in _ANALYSIS_PHASE_FRACTIONS.items()
    }

# Container regions that should never be plotted as individual curves.
# These are structural groupings, not meaningful attention targets.
SKIP_REGIONS = {
    "chat_template",
    "system_prompt",
    "user_message",
    "response",
    "thinking_section",
    "entities_section",
    "passages_section",
    "expansion_section",
    "complexity_section",
}

# Default region display order for cooking curves (when auto-discovered
# regions need a canonical ordering).
DEFAULT_DISPLAY_REGIONS = [
    "directive",
    "entity_rules",
    "passage_rules",
    "expansion_rules",
    "expansion_examples",
    "complexity_rules",
    "output_format",
    "conversation_turns",
    "task_reminders",
    "current_message",
    "stored_passages",
]
