# Pipeline Explained

How the MI pipeline works mechanically, from prompt assembly to visualization.

## Pipeline Overview

```
prep/inputs.py (local)
    prompt.txt + regions.json + conversations.json
    → test_cases.json (char-level region annotations)

engine/run_analysis.py (GPU box, self-contained)
    test_cases.json + HuggingFace model
    → per-case JSON (attention data + logit lens + per-token weights)

render/*.py + analysis/*.py (local)
    per-case JSONs → PNGs, GIFs, comparison tables, markdown reports
```

## Region Annotation

Regions are named character spans in the prompt text. The user defines them
via a JSON config file with marker-based, regex-based, or explicit-offset
boundaries.

**Character-level to token-level resolution** is the critical bridge. Three
strategies were evaluated empirically:

### Offset Mapping (FAILED)

HuggingFace's `return_offsets_mapping=True` returns `(0, 0)` for all special
tokens, making them invisible to character-to-token mapping. Since chat
template tokens (e.g., `<|im_start|>`) are special tokens, this approach
cannot establish piece boundaries.

### Subsequence Matching (FAILED for large regions)

Tokenize the region text independently, then search for that token subsequence
in the full sequence. This works for short regions but fails for regions >100
tokens because BPE tokenization of a substring differs from BPE tokenization
of the same text in context (boundary effects). In testing, 10/11 system
prompt regions failed this approach.

### Cumulative Decode Mapping (ROBUST - used)

Decode each token to get its character length, accumulate character offsets,
then use binary search to map character positions to token indices. This is
O(n) in sequence length and robust to BPE boundary effects because it works
from the actual token sequence rather than trying to match independently
tokenized substrings.

## Piecewise Tokenization

The chat template is assembled by tokenizing each content piece (system prompt,
user message, response) independently, then locating each piece in the full
`apply_chat_template()` output via subsequence matching. This establishes
piece boundaries without needing to know the template format.

**Why not just tokenize the full text?** Because special tokens (role markers,
delimiters) aren't in the vocabulary's text representation. Piecewise
tokenization lets us identify exactly where content starts and stops within
the template structure.

**Validation**: The assembled sequence is compared token-by-token against
`apply_chat_template()` output. Any mismatch means the region map would be
invalid and analysis is aborted.

## Attention Hooks

### Why hook self_attn, not the decoder layer

With accelerate's device mapping (`device_map="auto"`), accelerate registers
`AlignDevicesHook` on each decoder layer. These hooks fire AFTER user-registered
forward hooks. This means:

1. The ~2.2GB attention matrix gets physically moved between GPUs by
   `AlignDevicesHook.post_forward` BEFORE our hook can discard it
2. This causes OOM on multi-GPU setups

By hooking `self_attn` directly (which has no accelerate hooks), our
None-replacement fires immediately after attention computation, before the
matrix ever reaches `send_to_device`.

### Immediate None-replacement

The attention hook replaces `output[1]` (the attention weight matrix) with
`None` before returning. Since HuggingFace accumulates
`all_self_attns += (layer_output[1],)`, this stores a tuple of Nones instead
of a tuple of 2.2GB tensors. This saves ~140GB across 64 layers.

### Residual hooks

Residual stream capture hooks are registered on the decoder layer (not
self_attn). These read `output[0]` (hidden states) without modification.
The hook extracts vectors at query positions and stores them as float32
on CPU for logit lens projection.

## Logit Lens

At each layer, the residual stream vector at each query position is:

1. Extracted from the layer hook output
2. Projected through the model's final normalization layer (RMSNorm/LayerNorm)
3. Multiplied by the unembedding matrix (lm_head weights)
4. Softmaxed to get a probability distribution over the vocabulary

This reveals what the model would predict if decoding stopped at that layer.
Tracking specific tokens (e.g., `<` for XML format compliance) across layers
shows when the model "decides" on format vs. content.

## Per-Token Attention

For each layer at each query position, the attention weights across all heads
are averaged to produce a single attention value per token. This is the raw
data consumed by all visualization tools:

- **Heatmap**: Spatial attention map at selected layers
- **Cooking curves**: Mean attention per region at each layer
- **Layer GIF**: Animated sweep through all layers
- **Aggregate**: Cross-sample statistics with confidence bands

## Model Auto-Discovery

The engine auto-detects model architecture by:

1. Reading `model.config` for layer count, head counts, hidden size, vocab size
2. Walking the module tree to find the layers container (`model.model.layers`,
   `model.transformer.h`, etc.)
3. Finding attention submodules within each layer (`self_attn`, `attention`, `attn`)
4. Locating the LM head (`lm_head`, `output`, `embed_out`)
5. Locating the final norm (`model.model.norm`, `model.transformer.ln_f`, etc.)

This works for Llama, Qwen, Mistral, Gemma, GPT-NeoX, and most decoder-only
transformers without model-specific code.
