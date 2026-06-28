#!/usr/bin/env bash
###############################################################################
# Product A — OOD head on a Qwen3 backbone for EditLens AI-edit detection.
#
# Applies the OOD framing of arXiv 2510.08602 (machine = in-distribution, human
# = outlier, DeepSVDD hypersphere) to the EditLens continuous AI-edit-detection
# setup, on a Qwen3 backbone. Writes EVAL.md + results.json to
# .openresearch/artifacts/.
#
# Requires HF_TOKEN with access to the gated pangram/editlens_iclr dataset.
###############################################################################
set -euo pipefail
export REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

if ! command -v python >/dev/null 2>&1; then PY="$(command -v python3)"; python() { "$PY" "$@"; }; fi

export HF_HOME="${HF_HOME:-$REPO_ROOT/.hf_cache}"
export TOKENIZERS_PARALLELISM=false
# Propagate whichever HF token name is set to the one the hub libs read.
export HF_TOKEN="${HF_TOKEN:-${HUGGINGFACE_TOKEN:-}}"
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"

mkdir -p "$REPO_ROOT/.openresearch/artifacts"

echo "== Installing dependencies =="
pip install -q --no-input \
    "torch" "transformers>=4.51" "datasets>=2.19" "peft>=0.11" \
    "bitsandbytes>=0.43" "accelerate" "scikit-learn" "scipy" \
    "sentence-transformers" "emoji" "pandas" "huggingface_hub" 2>&1 | tail -5 || true

echo "== Smoke-test gated dataset access =="
python - <<'PY'
import os
from datasets import load_dataset
tok = os.environ.get("HF_TOKEN") or None
ds = load_dataset("pangram/editlens_iclr", split="val", token=tok)
print("OK: editlens_iclr val rows =", len(ds), "| cols:", ds.column_names[:6])
PY

MODEL="${MODEL:-Qwen/Qwen3-0.6B-Base}"
echo "== Training OOD head on $MODEL =="
python editlens/train_ood.py \
    --model_name "$MODEL" \
    --out_dim "${OUT_DIM:-256}" \
    --max_length "${MAXLEN:-512}" \
    --batch_size "${BATCH:-8}" \
    --epochs "${EPOCHS:-1}" \
    --lr "${LR:-1e-4}" \
    --max_train "${MAX_TRAIN:-4000}" \
    --max_val "${MAX_VAL:-1500}" 2>&1 | tee "$REPO_ROOT/.openresearch/artifacts/train.log"

echo "== Done. =="
