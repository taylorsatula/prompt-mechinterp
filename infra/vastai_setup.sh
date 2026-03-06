#!/usr/bin/env bash
# Vast.ai environment setup for MI analysis.
#
# Target: A100/H100 80GB+ with CUDA 12.x base image
# Memory rule of thumb: model_params * 2 bytes + 5GB headroom
#
# Usage:
#   scp src/prompt_mechinterp/engine/run_analysis.py vastai:/workspace/
#   scp test_cases.json vastai:/workspace/
#   ssh vastai 'bash /workspace/vastai_setup.sh'

set -euo pipefail

echo "=== MI Pipeline Setup ==="

# Core dependencies
pip install --quiet \
    torch \
    transformers \
    accelerate \
    safetensors \
    huggingface_hub

# HuggingFace auth (required if model is gated)
# Set HF_TOKEN env var before running, or run: huggingface-cli login
if [ -n "${HF_TOKEN:-}" ]; then
    huggingface-cli login --token "$HF_TOKEN" 2>/dev/null || true
    echo "HF auth configured"
fi

# Pre-download model weights
# Change the model ID and local_dir for your target model.
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-32B}"
MODEL_DIR="${MODEL_DIR:-/workspace/models/$(basename $MODEL_ID)}"

echo "=== Downloading $MODEL_ID ==="
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    '${MODEL_ID}',
    ignore_patterns=['*.gguf', '*.bin'],  # prefer safetensors
    local_dir='${MODEL_DIR}'
)
print('Model download complete')
"

echo "=== Setup complete ==="
echo "Run analysis with:"
echo "  python3 /workspace/run_analysis.py --input /workspace/test_cases.json --output /workspace/results/ --model-path ${MODEL_DIR}"
