#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

mkdir -p checkpoints
cd checkpoints

if [[ -f motiongpt3.ckpt ]]; then
  echo "checkpoints/motiongpt3.ckpt already exists — skipping."
  exit 0
fi

echo "The pretrained model motiongpt3.ckpt will be stored in checkpoints/"
gdown "https://drive.google.com/uc?id=1Wvx5PGJjVKPRvjcl8firChw1UVjUj36l"

cd "$REPO_ROOT"
echo "MotionGPT3 checkpoint download done."
