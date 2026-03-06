# prompt-mechinterp

Mechanistic interpretability toolkit for analyzing how any LLM processes any prompt. Captures per-token attention weights and logit lens projections across all layers, then renders the results as heatmaps, cooking curves, animated GIFs, and aggregate statistics.

## What it does

- **Attention capture**: Hooks every attention layer to extract head-averaged attention weights at configurable query positions
- **Logit lens**: Projects the residual stream through the final norm + LM head at each layer to track token rank trajectories
- **Region-based analysis**: Maps named regions (defined via JSON config) onto the token sequence, enabling per-region attention metrics
- **Visualization**: Four complementary renderers for different analytical questions
- **Variant comparison**: Automated N-variant comparison with delta tables, multi-seed stability analysis, and markdown reports

## Pipeline

```
prep/inputs.py (local)
    system_prompt.txt + regions.json + conversations.json
    --> test_cases.json (char-level region annotations)

engine/run_analysis.py (GPU box, self-contained)
    test_cases.json + any HuggingFace model
    --> per-case JSON (attention data + logit lens + per-token weights)

render/*.py + analysis/*.py (local)
    per-case JSONs --> PNGs, GIFs, comparison tables, markdown reports
```

## Quick start

```bash
# Install (local — rendering and analysis only)
pip install -e .

# Install with GPU dependencies (for running the engine)
pip install -e ".[gpu]"
```

### 1. Define regions

Create a `regions.json` describing named spans in your prompt:

```json
{
  "system_prompt": {
    "regions": [
      {"name": "rules", "start_marker": "## Rules", "end_marker": "## Examples"},
      {"name": "examples", "start_marker": "## Examples", "end_marker": null}
    ]
  },
  "user_message": {
    "regions": [
      {"name": "context", "start_marker": "Previous:", "end_marker": "Current:"},
      {"name": "current", "start_marker": "Current:", "end_marker": null}
    ]
  }
}
```

### 2. Prepare inputs

```bash
python -m prompt_mechinterp.prep.inputs \
    --prompt system_prompt.txt \
    --regions regions.json \
    --conversations conversations.json \
    --output test_cases.json
```

### 3. Run analysis on GPU

```bash
# scp the self-contained engine + test cases to your GPU box
scp src/prompt_mechinterp/engine/run_analysis.py gpu:/workspace/
scp test_cases.json gpu:/workspace/

# On the GPU box
python run_analysis.py \
    --input test_cases.json \
    --output results/ \
    --model-path /workspace/models/YourModel \
    --tracked-tokens "<" "keyword"
```

### 4. Render results

```bash
# Per-token attention heatmap
python -m prompt_mechinterp.render.heatmap --result results/case_0.json --mask-chatml

# Per-region attention trajectories across layers
python -m prompt_mechinterp.render.cooking_curves --result results/case_0.json --normalize per-region

# Animated layer sweep
python -m prompt_mechinterp.render.layer_gif --result results/case_0.json --mask-chatml

# Multi-sample aggregate with confidence bands
python -m prompt_mechinterp.render.aggregate --base-dir results/ --variants baseline:Baseline
```

### 5. Compare variants

```bash
python -m prompt_mechinterp.analysis.compare \
    --base-dir results/ \
    --variants baseline:Baseline modified:Modified \
    --ratio context:current_message

python -m prompt_mechinterp.analysis.report \
    --base-dir results/ \
    --experiments baseline:Baseline:results_baseline modified:Modified:results_modified \
    --output-dir reports/
```

## Package structure

```
src/prompt_mechinterp/
    constants.py          # Shared phase definitions, skip regions, display defaults
    engine/
        run_analysis.py   # Self-contained MI engine (scp to GPU boxes)
        model_adapter.py  # Auto-discovers architecture from any HF model
    prep/
        regions.py        # Region annotation from JSON config
        inputs.py         # Assemble test_cases.json
    render/
        _shared.py        # Fonts, colormaps, layout engine, normalization
        loaders.py        # Unified result JSON loading
        heatmap.py        # Per-token spatial attention heatmap (PNG)
        cooking_curves.py # Per-region attention trajectories (PNG)
        layer_gif.py      # Animated per-token heatmap sweep (GIF)
        aggregate.py      # Multi-sample aggregate curves (PNG)
    analysis/
        metrics.py        # Terminal avg, region ratios, density, cooking stats
        formatting.py     # Table output helpers
        compare.py        # N-variant comparison with delta tables
        report.py         # Markdown experiment reports
docs/
    PIPELINE_EXPLAINED.md # How the MI pipeline works mechanically
    PITFALLS.md           # Failure modes and solutions
    KNOWN_GOOD_APPROACHES.md # Empirically validated patterns
infra/
    vastai_setup.sh       # GPU box bootstrap script
```

## Model support

The engine auto-discovers model architecture from any HuggingFace decoder-only transformer:
- Reads layer count, head counts, hidden size, vocab size from `model.config`
- Walks the module tree to find attention submodules, LM head, and final norm
- Tested with: Llama, Qwen, Mistral, Gemma, GPT-NeoX families

Requirements: `attn_implementation="eager"` (flash attention doesn't materialize the attention matrix).

## GPU requirements

Rule of thumb: `model_params * 2 bytes + 5GB headroom` (fp16 weights + attention capture overhead).

| Model | VRAM needed | Recommended GPU |
|-------|-------------|-----------------|
| 8B params | ~21GB | A100 40GB |
| 32B params | ~69GB | H100 80GB |
| 70B params | ~145GB | Won't fit single GPU — use quantization |

See `docs/PITFALLS.md` for memory estimation details and OOM prevention.

## Documentation

- **[Pipeline Explained](docs/PIPELINE_EXPLAINED.md)** — How region annotation, attention hooks, logit lens, and per-token capture work
- **[Pitfalls](docs/PITFALLS.md)** — Failure modes discovered empirically, with root causes and solutions
- **[Known Good Approaches](docs/KNOWN_GOOD_APPROACHES.md)** — Patterns validated across multiple experiments
- **[SKILL.md](SKILL.md)** — Operational reference for running the full MI pipeline
