#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

mkdir -p deps
cd deps

if [[ -d smpl/smpl_models ]]; then
  echo "deps/smpl already present — skipping SMPL download."
  exit 0
fi

echo "The SMPL model will be stored in deps/"
echo "Downloading"
gdown "https://drive.google.com/uc?id=1qrFkPZyRwRGd0Q3EY76K8oJaIgs_WK9i"
echo "Extracting"
tar xfzv smpl.tar.gz
echo "Cleaning"
rm smpl.tar.gz

echo "SMPL download done."
