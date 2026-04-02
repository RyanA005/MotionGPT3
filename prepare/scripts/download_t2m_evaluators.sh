#!/usr/bin/env bash
# T2M evaluation assets (glove, mean/std, text-motion matching checkpoints).
# The Google Drive archive sometimes unpacks with an extra nested t2m/; we normalize paths
# so deps/t2m/t2m/... matches what the code expects (see METRIC.TM2T.t2m_path in configs/assets.yaml).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

mkdir -p deps
cd deps

fix_nested_t2m() {
  local nested="t2m/t2m/t2m"
  if [[ ! -d "$nested" ]]; then
    return 0
  fi
  echo "Normalizing nested t2m archive layout (moving t2m/t2m/t2m/* up to t2m/t2m/)..."
  shopt -s nullglob
  local item dest name
  for item in "$nested"/*; do
    name=$(basename "$item")
    dest="t2m/t2m/${name}"
    if [[ ! -e "$dest" ]]; then
      mv "$item" "t2m/t2m/"
    fi
  done
  shopt -u nullglob
  rmdir "$nested" 2>/dev/null || true
}

if [[ -f t2m/t2m/text_mot_match/model/finest.tar ]]; then
  echo "T2M evaluators already present (finest.tar found) — skipping download."
  exit 0
fi

if [[ -f t2m/t2m/t2m/text_mot_match/model/finest.tar ]]; then
  echo "Fixing nested t2m paths from a previous extract..."
  fix_nested_t2m
  cd "$REPO_ROOT"
  echo "T2M path fix done."
  exit 0
fi

echo "The t2m evaluators will be stored in deps/"
echo "Downloading"
gdown "https://drive.google.com/uc?id=1AYsmEG8I3fAAoraT4vau0GnesWBWyeT8"
echo "Extracting"
tar xfzv t2m.tar.gz

fix_nested_t2m

echo "Cleaning"
rm -f t2m.tar.gz

cd "$REPO_ROOT"
echo "T2M evaluators download done."
