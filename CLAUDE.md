# prompt-mechinterp — Mechanistic Interpretability Toolkit

Generic toolkit for analyzing how any LLM processes any prompt via attention capture, logit lens, and region-based analysis.

## Package Structure

```
src/prompt_mechinterp/
    __init__.py              # Package version (0.1.0)
    constants.py             # FINAL_LAYERS, DISPLAY_PHASES, ANALYSIS_PHASES, SKIP_REGIONS

    engine/
        run_analysis.py      # Self-contained MI engine (scp to GPU boxes, no package imports)
        model_adapter.py     # Auto-discovers architecture from any HF model config

    prep/
        regions.py           # Region annotation from JSON config (marker/regex/char-range)
        inputs.py            # CLI: assemble test_cases.json from prompt + regions + conversations

    render/
        _shared.py           # Fonts, colormaps, layout engine, normalization, palettes
        loaders.py           # Unified data loading from result JSONs
        heatmap.py           # Per-token spatial attention heatmap (PNG)
        cooking_curves.py    # Per-region attention trajectories across layers (PNG)
        layer_gif.py         # Animated per-token heatmap layer sweep (GIF)
        aggregate.py         # Multi-sample aggregate curves with confidence bands (PNG)

    analysis/
        metrics.py           # Terminal avg, region ratios, density, cooking stats
        formatting.py        # Table output: fmt, pct, delta_str, print_header
        compare.py           # N-variant comparison with auto-discovered regions
        report.py            # Markdown experiment reports with delta-from-baseline

docs/
    PIPELINE_EXPLAINED.md    # How attention hooks, logit lens, region annotation work
    PITFALLS.md              # Failure modes and solutions (OOM, hook ordering, BPE)
    KNOWN_GOOD_APPROACHES.md # Empirically validated patterns

infra/
    vastai_setup.sh          # GPU box bootstrap (configurable MODEL_ID)

subcortical_tuning_test.py   # PARKED — MIRA-dependent, separate integration plan
```

## Data Flow

```
prep/inputs.py (local)
    system_prompt.txt + regions.json + conversations.json
    --> test_cases.json (char-level region annotations)

engine/run_analysis.py (GPU box, self-contained)
    test_cases.json + any HuggingFace model
    --> per-case JSON (attention + logit lens + per-token weights)

render/*.py + analysis/*.py (local)
    per-case JSONs --> PNGs, GIFs, comparison tables, markdown reports
```

## Key Design Decisions

### Self-contained engine (dual-existence pattern)
`engine/run_analysis.py` has model discovery logic inlined so it can be scp'd to a GPU box as a single file with no package imports. `engine/model_adapter.py` has the clean importable version of the same logic. The `engine/__init__.py` documents this design.

### Model auto-discovery
The engine reads `model.config` for layer count, head counts, hidden size, vocab size. It walks the module tree to find attention submodules, LM head, and final norm. No hardcoded model assumptions — works with Llama, Qwen, Mistral, Gemma, GPT-NeoX families.

### Single GPU (`device_map={"": 0}`)
Multi-GPU via `device_map="auto"` causes OOM between cases due to accelerate's `AlignDevicesHook` leaking state. Single GPU eliminates all accelerate hooks.

### Hooks on `self_attn` (not decoder layer)
With accelerate's device mapping, user hooks fire AFTER `AlignDevicesHook.post_forward`. Hooking `self_attn` directly (no accelerate hooks) ensures immediate capture.

### `attn_implementation="eager"` (mandatory)
Flash attention doesn't materialize the attention matrix. Eager attention always does, regardless of `output_attentions` flag.

### Piece boundary detection via full-sequence decode + bisect
`build_chat_tokens` decodes the full token sequence with `tokenizer.decode()`, finds content strings via `str.find()`, then maps char positions to token indices via binary search with progressive prefix decoding. This handles SentencePiece leading-space markers, BPE boundary effects at template junctions, and models that merge system into user (Gemma).

### System role fallback
Models that don't support system role (Gemma family) are handled automatically: the engine catches the template error, merges system content into the first user message with a `\n` separator, and re-applies the template. Region boundaries remain correct because the content strings are still findable in the decoded text.

### Char-to-token via cumulative decode (sub-regions)
Within located pieces, `resolve_char_regions_to_tokens` uses cumulative character offset mapping via per-token `tokenizer.decode()` for sub-region resolution. This is robust for intra-piece mapping where tokens are already sliced from the piece boundary.

### Dynamic phase scaling
Display phases and analysis phases scale proportionally to any layer count via `display_phases(num_layers)` and `analysis_phases(num_layers)`. No hardcoded layer numbers in renderers.

### Region annotation via JSON config
Users define regions in a JSON file with marker-based, regex-based, or character-range boundaries. No hardcoded markers or delimiters.

## CLI Quick Reference

```bash
# Prep
python -m prompt_mechinterp.prep.inputs --prompt X --regions X --conversations X --output X

# Engine (on GPU box)
python run_analysis.py --input X --output X --model-path X [--tracked-tokens "<"] [--no-per-token]

# Render
python -m prompt_mechinterp.render.heatmap --result X [--mask-chatml] [--clip-low 0.05]
python -m prompt_mechinterp.render.cooking_curves --result X [--normalize per-region]
python -m prompt_mechinterp.render.layer_gif --result X [--mask-chatml] [--fps 4]
python -m prompt_mechinterp.render.aggregate --base-dir X --variants name:Label

# Analysis
python -m prompt_mechinterp.analysis.compare --base-dir X --variants name:Label [--ratio a:b] [--by-seed]
python -m prompt_mechinterp.analysis.report --base-dir X --experiments key:label:dir --output-dir X
```

## Output Locations

All experimental data goes in `data/` (gitignored). Convention:
- `data/inputs/test_cases.json`
- `data/results_variant_name/sample_N.json`
- `data/reports/experiment_name.md`

## Extending

- **New renderer**: Add to `render/`, import shared utilities from `render/_shared.py` and data loading from `render/loaders.py`. Keep `main()` + `if __name__` block.
- **New metric**: Add to `analysis/metrics.py`. Use in `compare.py` or `report.py`.
- **New model architecture**: If auto-discovery fails, add detection paths to `engine/model_adapter.py` AND inline the same logic in `engine/run_analysis.py` (maintain self-containment).
- **New region detection strategy**: Add to `prep/regions.py`.
