#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

mkdir -p checkpoints
cd checkpoints

if [[ -d MotionGPT-base/.git ]]; then
  echo "checkpoints/MotionGPT-base already exists — skipping clone."
  exit 0
fi

mkdir -p mld_humanml3d_checkpoint
git lfs install
echo "Cloning OpenMotionLab/MotionGPT-base (large, LFS)..."
git clone https://huggingface.co/OpenMotionLab/MotionGPT-base

cd "$REPO_ROOT"
echo "MotionGPT-base download done."
