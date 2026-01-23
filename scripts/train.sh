#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

ACTIVATE_HELPER="$ROOT_DIR/scripts/_activate_conda.sh"
if [[ ! -f "$ACTIVATE_HELPER" ]]; then
    echo "[train] missing conda helper: $ACTIVATE_HELPER" >&2
    exit 1
fi
source "$ACTIVATE_HELPER" ddcolor

CUDA_VISIBLE_DEVICES=0,1,2,3 \
python3 -m torch.distributed.launch --nproc_per_node=4 --master_port=3721 basicsr/train.py \
    -opt options/train/train_ddcolor.yml --auto_resume --launcher pytorch
