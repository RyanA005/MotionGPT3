#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

mkdir -p checkpoints
cd checkpoints

if [[ -f 1222_mld_humanml3d_FID041.ckpt ]]; then
  echo "checkpoints/1222_mld_humanml3d_FID041.ckpt already exists — skipping."
  exit 0
fi

echo "Downloading MLD HumanML3D VAE checkpoint into checkpoints/"
gdown "https://drive.google.com/uc?id=1hplrnQwUK_cZFHirZIOuVP0RSyZEC1YM"

cd "$REPO_ROOT"
echo "MLD pretrained model download done."
