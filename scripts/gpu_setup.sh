#!/bin/bash
# CLASP GPU Training Setup Script
# Target: A100 80GB VM node
# Run this once after VM provisioning

set -euo pipefail

echo "=== CLASP Training Environment Setup ==="

# System packages
apt-get update && apt-get install -y git wget curl htop nvtop 2>/dev/null || true

# Python environment
pip install --upgrade pip

# Core training dependencies
pip install \
    torch>=2.1.0 \
    transformers>=4.45.0 \
    peft>=0.13.0 \
    bitsandbytes>=0.44.0 \
    datasets>=2.19.0 \
    accelerate>=0.34.0 \
    trl>=0.12.0 \
    wandb \
    scipy \
    sentencepiece

# Flash Attention 2 (major speedup on A100)
pip install flash-attn --no-build-isolation 2>/dev/null || echo "Flash Attention install failed (non-critical)"

# Verify GPU
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
    print(f'BF16 support: {torch.cuda.is_bf16_supported()}')
"

echo ""
echo "=== Setup complete ==="
echo "Next: upload sft_dataset.openai.jsonl and run train_qlora.py"
