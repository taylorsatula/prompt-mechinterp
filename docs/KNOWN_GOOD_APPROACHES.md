# Known Good Approaches

Patterns validated across multiple experiments.

## GPU Configuration

**Single GPU `device_map={"": 0}`** eliminates all accelerate hook
interference. The memory savings from multi-GPU don't justify the hook
ordering and OOM-leak bugs. If the model doesn't fit on one GPU, use a
smaller model or quantization — don't use device_map="auto".

## Tokenization

**Piecewise tokenization with full-sequence validation** is the only
reliable way to establish piece boundaries in the token sequence. Tokenize
each content piece independently, locate it in the `apply_chat_template()`
output via subsequence matching, and validate token-by-token.

**Cumulative decode mapping** (decode tokens 0..N, measure character count,
advance) is the only robust char-to-token resolution for large regions.
Offset mapping fails on special tokens. Subsequence matching fails on
regions >100 tokens due to BPE boundary effects.

## Visualization

**Rank-based histogram equalization** for heatmaps eliminates power-law
hotspot blowouts where 2-3 tokens have 100x the attention of the mean.
Every percentile gets equal visual weight, making the full attention
landscape visible.

**Per-region normalization** (`--normalize per-region`) for cooking curves
when comparing trajectory SHAPES across regions of vastly different
magnitude. Each region's curve is normalized to its own [0, 1] range.
Use `raw` mode for seeing actual attention budget allocation.

## Analysis Metrics

**4-layer terminal average** (final 4 layers) as the "what the model
decided" signal. More stable than single-layer measurements, captures
the output preparation phase. For a 64-layer model, this is L60-63.

**Per-region per-token density** (`attention / n_tokens`) is the fair
comparison metric across regions. Raw attention sums are dominated by
region length — a region with 500 tokens will naturally capture more
total attention than a region with 20 tokens, even if the 20-token
region is per-token more important.

## Attention Capture

**Hook self_attn, not the decoder layer.** This avoids accelerate's
AlignDevicesHook ordering issues and ensures immediate capture.

**`attn_implementation="eager"`** is mandatory. Flash attention doesn't
materialize the attention matrix, so there's nothing for hooks to capture.
Eager attention always materializes the full matrix regardless of
`output_attentions` flag.

**Immediate None-replacement** of the attention output in hooks saves
~2.2GB per layer (for large models). Without this, HuggingFace accumulates
64 copies of the attention matrix.

## Experiment Design

**Region ratios** (e.g., conv_turns / current_message) are more informative
than absolute attention values. Absolute values shift with prompt length
and content, but ratios reveal relative priority.

**Multi-seed testing** catches seed-dependent instabilities. A variant
that looks good on one seed may regress on another. Test with 3-4 seeds
before declaring a variant successful.

**Cooking curves** (per-region attention trajectories across layers) reveal
dynamics that terminal-layer metrics miss entirely. Rules may peak at
L0-8 with 14x more attention than at terminal — this absorption pattern
is invisible in terminal-only analysis.
