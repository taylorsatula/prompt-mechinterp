# MI Analysis Operational Reference

Operational guide for running the prompt-mechinterp pipeline end-to-end. Written for a collaborator fluent in transformer internals, attention mechanics, and logit lens methodology.

## Pipeline Orchestration

### Stage 1: Region Annotation (local)

Define regions via `regions.json` — named character spans in the prompt text. Three detection strategies:

- **Marker-based**: `"start_marker": "## Rules", "end_marker": "## Examples"` — literal text boundaries
- **Regex-based**: `"start_pattern": "Task \\d+:", "end_pattern": "Task \\d+:|$"` — for repeating structures
- **Character range**: `"start_char": 0, "end_char": 500` — explicit offsets

Assemble test cases:
```bash
python -m prompt_mechinterp.prep.inputs \
    --prompt system_prompt.txt \
    --regions regions.json \
    --conversations conversations.json \
    --output test_cases.json
```

The output `test_cases.json` contains the full prompt text, character-level region annotations, query position definitions, and tracked token list — everything the engine needs.

### Stage 2: GPU-Side Analysis

`run_analysis.py` is self-contained — no package imports, only torch/transformers/stdlib. Deploy via scp:

```bash
scp src/prompt_mechinterp/engine/run_analysis.py gpu:/workspace/
scp test_cases.json gpu:/workspace/
ssh gpu

# On GPU box:
bash /workspace/vastai_setup.sh  # installs deps, downloads model
python /workspace/run_analysis.py \
    --input /workspace/test_cases.json \
    --output /workspace/results/ \
    --model-path /workspace/models/YourModel \
    --tracked-tokens "<" "keyword" \
    --query-positions terminal
```

Key flags:
- `--model-path`: Local path to HF model weights (required)
- `--tracked-tokens`: Tokens to track in logit lens (default: none)
- `--query-positions`: Which positions to analyze (default: from test_cases.json)
- `--cases`: Specific case indices to run (default: all)
- `--no-per-token`: Skip per-token attention capture (faster, but no heatmaps)
- `--top-k`: Number of top logit lens predictions to save per layer (default: 10)

The engine auto-discovers model architecture — layer count, head config, attention module paths, LM head, final norm — from `model.config` and module tree walking.

### Stage 3: Result Retrieval

```bash
scp -r gpu:/workspace/results/ ./data/results/
```

Each result JSON contains: per-token attention weights (all layers), logit lens projections (tracked tokens), region map (token-level), and metadata.

### Stage 4: Visualization

```bash
# Heatmap: "where is the model looking at position X?"
python -m prompt_mechinterp.render.heatmap \
    --result data/results/case_0.json \
    --position terminal \
    --layers L60-63 \
    --mask-chatml \
    --clip-low 0.05 \
    --colormap inferno

# Cooking curves: "how does attention to each region evolve through the forward pass?"
python -m prompt_mechinterp.render.cooking_curves \
    --result data/results/case_0.json \
    --position terminal \
    --normalize per-region   # or 'raw' for absolute values

# Layer GIF: "watch attention flow through all layers"
python -m prompt_mechinterp.render.layer_gif \
    --result data/results/case_0.json \
    --position terminal \
    --mask-chatml \
    --fps 4 --stride 1

# Aggregate: "is this pattern stable across samples?"
python -m prompt_mechinterp.render.aggregate \
    --base-dir data/results/ \
    --variants baseline:Baseline modified:Modified
```

### Stage 5: Comparative Analysis

```bash
# N-variant comparison with delta tables
python -m prompt_mechinterp.analysis.compare \
    --base-dir data/results/ \
    --variants baseline:Baseline anchor:Anchor trim3:Trim3 \
    --ratio conv_turns:current_message \
    --logit-lens-tokens "<" \
    --by-seed \
    --metrics all

# Markdown experiment reports
python -m prompt_mechinterp.analysis.report \
    --base-dir data/results/ \
    --experiments baseline:Baseline:results_baseline anchor:Anchor:results_anchor \
    --output-dir reports/
```

## Vast.ai Workflow

### Provisioning

VRAM requirement: `model_params * 2 bytes + 5GB headroom` (fp16).

- 8B model: A100 40GB ($0.50-0.80/hr)
- 32B model: H100 80GB ($2.50-3.50/hr)

Search for instances with CUDA 12.x base image and sufficient VRAM. Prefer single-GPU instances — multi-GPU with `device_map="auto"` causes OOM from accelerate hook state leaks.

### Bootstrap

```bash
# Set env vars before running
export HF_TOKEN="your_token"
export MODEL_ID="Qwen/Qwen3-32B"  # or any HF model

scp infra/vastai_setup.sh gpu:/workspace/
scp src/prompt_mechinterp/engine/run_analysis.py gpu:/workspace/
scp test_cases.json gpu:/workspace/
ssh gpu 'bash /workspace/vastai_setup.sh'
```

The setup script installs torch/transformers/accelerate, authenticates with HuggingFace (if HF_TOKEN set), and downloads model weights.

### Execution Monitoring

Watch for:
- Model loading: "Loading model from..." — takes 2-5 min for 32B models
- Per-case progress: "Processing case N/M" — each case takes 30-90s depending on sequence length
- Memory: peak should be near `model_params * 2 + 5GB`. If it grows between cases, there's a leak.
- Attention capture confirmation: "Registered hooks on N layers"

### Result Size

Each result JSON is 5-50MB depending on sequence length and number of layers. Budget ~500MB per 10-case experiment.

## Visualization Guide

### Which renderer for which question?

| Question | Renderer | Key flags |
|----------|----------|-----------|
| Where does the model look at position X? | `heatmap` | `--layers`, `--mask-chatml` |
| How does region attention evolve across layers? | `cooking_curves` | `--normalize` |
| What's the full layer-by-layer attention dynamics? | `layer_gif` | `--fps`, `--stride` |
| Is this pattern stable across samples? | `aggregate` | `--variants` |

### Flag guidance

- `--mask-chatml`: Exclude ChatML structural tokens from rank normalization. Use for content-focused analysis. Without it, `<|im_start|>` and `<|im_end|>` dominate the colormap.
- `--clip-low N`: Set attention values below the Nth percentile to zero. Use 0.05-0.10 to cut noise floor.
- `--normalize per-region`: Each region's cooking curve normalized to its own [0,1] range. Use for comparing trajectory SHAPES across regions of vastly different magnitude.
- `--normalize raw`: Absolute attention values. Use for seeing actual attention budget allocation. Current_message will dominate.
- `--layers L60-63`: Analyze specific layers. "L60-63" = final 4 layers = "what the model decided." "L0-8" = early attention = rules absorption phase.
- `--smoothing N`: Gaussian smoothing sigma for heatmaps. Higher = smoother. Default 0 (no smoothing).

## Interpretive Framework

### Layer phases (for a 64-layer model)

| Phase | Layers | What happens |
|-------|--------|--------------|
| Broad read | 0-6 | Model reads everything. Rules get 14x more attention than at terminal layers. |
| Absorption | 7-11 | Rule content gets absorbed into residual stream. Attention drops sharply. |
| Compression | 12-31 | Quiet middle layers. Information is being integrated. |
| Re-engagement | 32-47 | Current message blazes. Context-dependent processing. |
| Output prep | 48-63 | Model commits to output. Examples and format tokens dominate. |

### What attention patterns reveal

- **Region ratio** (e.g., conv_turns / current_message): Measures relative priority. A ratio >2x at terminal position indicates the model gives more attention to conversation history than the current message — potential context bleed.
- **Per-token density** (attention / n_tokens): Fair comparison across regions of different size. A 20-token region with 0.05 density is more influential per-token than a 500-token region with 0.02 density.
- **Cooking curve peak timing**: Rules peaking at L0-8 then fading = normal absorption. Rules still high at L48+ = persistent influence (or the model is confused).
- **Retention ratio** (terminal_value / peak_value): How much of a region's peak attention survives to the output layers. Low retention = absorbed early. High retention = persistent influence.

### Logit lens interpretation

- Track specific tokens across layers to see when the model "decides" on them
- A token at rank 1-10 means the model would produce it if decoding stopped at that layer
- Rank crashes (e.g., from rank 4 to rank 64K at layer 48) indicate competing representations overwrote the prediction
- Recovery from a crash (back to rank <100 by terminal layers) means the model resolved the conflict

### Red flags in analysis

- Context bleed ratio >2x: conversation history dominates over current message
- Format token rank >1000 at terminal layer: format compliance likely broken
- Region with <0.1% per-token density at its expected influence point: the model ignores it
- Cooking curve that never peaks: the region has no influence at any layer
- Multi-seed instability: if a metric varies >50% across seeds, the pattern is fragile

## Data Conventions

### Directory structure

```
data/
    results_baseline/
        sample_0.json
        sample_1.json
        ...
    results_variant_a/
        sample_0.json
        ...
```

### Result JSON schema

```json
{
  "case_id": "case_0",
  "model": "Qwen/Qwen3-32B",
  "num_layers": 64,
  "num_tokens": 2048,
  "region_map": {"rules": [100, 250], "examples": [250, 400], ...},
  "piece_boundaries": {"system": [5, 500], "user": [502, 800], "assistant": [802, 810]},
  "positions": {
    "terminal": {
      "token_idx": 2047,
      "per_token_attention": {"0": [...], "1": [...], ...},
      "logit_lens": {"0": {"top_k": [...], "tracked": {...}}, ...}
    }
  }
}
```

- `region_map`: Maps region name to `[start_token, end_token)` half-open intervals
- `piece_boundaries`: Maps chat piece name to token range
- `per_token_attention`: Layer index (string) -> array of float32 attention weights, one per token. Head-averaged.
- `logit_lens.tracked`: Token string -> `{"rank": int, "prob": float}` at each layer
