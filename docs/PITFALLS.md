# Pitfalls

Failure modes discovered empirically, with root causes and solutions.

## OOM Between Cases

**Symptom**: First case analyzes fine, second case OOMs despite identical size.

**Root cause**: accelerate's `AlignDevicesHook` leaks state across forward passes
when using `device_map="auto"`. Internal buffers accumulate and aren't freed
between cases.

**Solution**: Use single GPU with `device_map={"": 0}`. This eliminates all
accelerate hooks. Peak memory is model_params * 2 bytes + ~3-5GB overhead.
For Qwen3-32B (fp16), peak is 65-69GB on an 80GB H100.

## Hook Ordering on Layer Modules

**Symptom**: Attention capture returns corrupted or None data when using
`device_map="auto"`.

**Root cause**: With multi-GPU device mapping, accelerate registers
`AlignDevicesHook` on each decoder layer. User-registered forward hooks fire
AFTER accelerate's hooks, which means `send_to_device(output, input_device)`
executes before your hook can capture or replace the attention matrix.

**Solution**: Hook `self_attn` directly instead of the decoder layer. The
self_attn submodule has no accelerate hooks, so user hooks fire immediately.

## BPE Boundary Effects in Region Resolution

**Symptom**: Region token subsequence matching fails for regions >100 tokens.
Only 1/11 system prompt regions matched.

**Root cause**: BPE tokenization of a substring produces different tokens than
BPE tokenization of the same text within a longer context. Token boundaries
shift when surrounding context changes, so independently-tokenized region text
doesn't appear as a contiguous subsequence in the full-context tokenization.

**Solution**: Use cumulative character offset mapping via `tokenizer.decode()`.
Decode each token, accumulate character lengths, use binary search for
char-to-token resolution. This works from the actual token sequence.

## (0,0) Offset Mapping

**Symptom**: `return_offsets_mapping=True` returns `(0, 0)` for special tokens,
making them invisible to character-to-token mapping.

**Root cause**: HuggingFace's offset mapping doesn't track special tokens
(they have no character-level source text).

**Solution**: Piecewise tokenization avoids this entirely by locating content
pieces via subsequence matching rather than relying on offset mapping.

## Recency Gradient Catastrophe

**Symptom**: Placing directive content at the end of the prompt to leverage
recency made the problem WORSE.

**Root cause**: The recency gradient is powerful enough to overwrite the format
instruction representation at L48. In testing, a terminal `<primary_focus>`
directive increased context bleed by 19% and destroyed format compliance
(`<` token rank 34K at L63 — never recovers from the L48 crash).

**Solution**: Don't fight the recency gradient. Place instructions where the
model naturally reads them (early in the prompt). Use structural markers
(rare-token anchors) for salience rather than positional tricks.

## ChatML Token Rank Domination

**Symptom**: Heatmaps show all attention concentrated on a few tokens,
with content regions washed out.

**Root cause**: ChatML structural tokens (`<|im_start|>`, `<|im_end|>`)
naturally receive very high attention and dominate the rank normalization,
compressing the useful range for content tokens.

**Solution**: Use `--mask-chatml` to exclude ChatML tokens from rank
calculation and render them as neutral gray. This lets the colormap
represent the full dynamic range of content-token attention.

## Memory Estimation

**Rule of thumb**: fp16 model weights + attention capture overhead.

- Model weights: `model_params * 2 bytes` (fp16)
- Attention capture: ~3-5GB overhead on peak
- Per-layer attention matrix: `batch * heads * seq_len * seq_len * 4 bytes`
  (float32 before our hook replaces it with None)

For Qwen3-32B: 64GB weights + 5GB overhead = 69GB peak on 80GB H100.
For Llama-3-8B: 16GB weights + 3GB overhead = 19GB peak.

Always provision GPU with at least model_params * 2 + 5GB headroom.

## output_attentions Flag Irrelevance

**Symptom**: Setting `output_attentions=True/False` has no effect on whether
attention weights are captured.

**Root cause**: Many models' eager attention implementations return the full
attention matrix regardless of this flag. The decoder layer discards attention
unconditionally (`hidden_states, _ = self.self_attn(...)`). The flag only
controls whether the DECODER LAYER keeps the return — but hooks on self_attn
capture it before the layer discards it.

**Solution**: Don't rely on `output_attentions`. Always use hooks for
capture, and always use `attn_implementation="eager"` to ensure the attention
matrix is materialized (flash attention doesn't produce it).
