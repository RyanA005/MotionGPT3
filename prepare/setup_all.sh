#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
S="$REPO_ROOT/prepare/scripts"

command -v python >/dev/null || { echo "python not found"; exit 1; }

python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt

bash "$S/download_smpl_model.sh"
bash "$S/download_t2m_evaluators.sh"
bash "$S/prepare_gpt2.sh"
bash "$S/download_pretrained_motiongpt3_model.sh"

if [[ ! -f deps/mot-gpt2/config.json ]]; then
  python -m scripts.gen_mot_gpt
fi

echo "done."
