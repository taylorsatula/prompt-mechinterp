"""Shared constants reconciled across all analysis and rendering modules.

NUM_LAYERS and NUM_QUERY_HEADS are intentionally absent — these are model-specific
and must come from model auto-discovery (engine/model_adapter.py), not constants.
"""

# How many final layers to average for "what the model decided" metrics.
# L60-63 for a 64-layer model; for other models, use the last FINAL_LAYERS layers.
FINAL_LAYERS = 4

# Display phases — 4-phase model for chart labels and phase annotations.
# Empirically derived from Qwen3-32B cooking curves; boundaries are approximate
# and may shift for models with different layer counts.
DISPLAY_PHASES = [
    ("Rules absorbed", 0, 8),
    ("Internal computation", 8, 32),
    ("Focus narrows", 32, 48),
    ("Output formatting", 48, 63),
]

# Analysis phases — 5-phase model for detailed per-phase metrics.
ANALYSIS_PHASES = {
    "P1_broad_read": (0, 6),
    "P2_absorption": (7, 11),
    "P3_compression": (12, 31),
    "P4_reengagement": (32, 47),
    "P5_output_prep": (48, 63),
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
