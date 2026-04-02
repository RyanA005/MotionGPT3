#!/usr/bin/env bash
# Clone Hugging Face openai-community/gpt2 into deps/gpt2 (LFS weights required).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

mkdir -p deps
cd deps

git lfs install

if [[ -d gpt2/.git ]]; then
  echo "deps/gpt2 already exists — skipping clone."
  exit 0
fi

echo "Cloning openai-community/gpt2 into deps/gpt2 (this may take a while)..."
git clone https://huggingface.co/openai-community/gpt2

echo "GPT-2 clone done."
