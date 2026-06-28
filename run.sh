#!/usr/bin/env bash
###############################################################################
# SCALED Product A — OOD AI-edit detector on the 4B EditLens-Qwen3 backbone.
#
# Scales the winning OOD-head detector (Product A, AUROC 0.941 at 0.6B) to the
# 4B EditLens-Qwen3 repro backbone on the FULL pangram/editlens_iclr training
# set. Fine-tunes the DeepSVDD OOD head (human = in-distribution) with QLoRA;
# the score is the oriented distance from the human center — a continuous
# "how-AI-edited" meter. Publishes to HF as ood-editguard-qwen3-4b.
#
# Requires HF_TOKEN with access to the gated pangram/editlens_iclr dataset.
###############################################################################
set -euo pipefail
export REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

if ! command -v python >/dev/null 2>&1; then PY="$(command -v python3)"; python() { "$PY" "$@"; }; fi

export HF_HOME="${HF_HOME:-$REPO_ROOT/.hf_cache}"
export TOKENIZERS_PARALLELISM=false
export HF_TOKEN="${HF_TOKEN:-${HUGGINGFACE_TOKEN:-}}"
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
mkdir -p "$REPO_ROOT/.openresearch/artifacts"

echo "== Installing dependencies =="
# torch>=2.6 enables the bitsandbytes 4-bit path (set_submodule); QLoRA needs it
# for the 4B model. bf16+LoRA is the automatic fallback if 4-bit can't attach.
pip install -q --no-input \
    "torch>=2.6" "transformers>=4.51" "datasets>=2.19" "peft>=0.11" \
    "bitsandbytes>=0.43" "accelerate" "scikit-learn" "scipy" \
    "sentence-transformers" "emoji" "pandas" "huggingface_hub" 2>&1 | tail -5 || true

echo "== Smoke-test gated dataset access =="
python - <<'PY'
import os
from datasets import load_dataset
ds = load_dataset("pangram/editlens_iclr", split="val", token=os.environ.get("HF_TOKEN") or None)
print("OK: editlens_iclr val rows =", len(ds))
PY

# Base = the 4B EditLens-Qwen3 repro (search HF: "editlens qwen3 repro").
# Its encoder already understands edit-extent; we fine-tune the OOD head onto it.
MODEL="${MODEL:-${BASE_MODEL:-reneeice/editlens-qwen3-4b-repro}}"
echo "== Scaling OOD head on $MODEL (full dataset) =="
python editlens/train_ood.py \
    --model_name "$MODEL" \
    --repo_suffix "${REPO_SUFFIX:-ood-editguard-qwen3-4b}" \
    --out_dim "${OUT_DIM:-256}" \
    --max_length "${MAXLEN:-1024}" \
    --batch_size "${BATCH:-8}" \
    --epochs "${EPOCHS:-2}" \
    --lr "${LR:-1e-4}" \
    --qlora \
    --max_train "${MAX_TRAIN:-0}" \
    --max_val "${MAX_VAL:-0}" 2>&1 | tee "$REPO_ROOT/.openresearch/artifacts/train.log"

echo "== Done. =="
