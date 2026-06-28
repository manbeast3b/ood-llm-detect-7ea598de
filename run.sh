#!/usr/bin/env bash
###############################################################################
# FULL-DATASET Product A — OOD AI-edit detector on the 0.6B EditLens-Qwen3 backbone.
#
# Trains the winning OOD-head detector (Product A) on the 0.6B EditLens-Qwen3
# repro backbone over the FULL pangram/editlens_iclr training set (vs the earlier
# 4,000-sample proof). 0.6B already hit AUROC 0.941 on a subsample; the full set
# gives the production model far faster than 4B (see the scaling/requirements
# report). Fine-tunes the DeepSVDD OOD head (human = in-distribution); the score
# is the oriented distance from the human center — a continuous "how-AI-edited"
# meter. Publishes to HF as ood-editguard-qwen3-0.6b-full.
#
# Requires HF_TOKEN with access to the gated pangram/editlens_iclr dataset.
###############################################################################
set -euo pipefail
export REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

if ! command -v python >/dev/null 2>&1; then PY="$(command -v python3)"; python() { "$PY" "$@"; }; fi

export HF_HOME="${HF_HOME:-$REPO_ROOT/.hf_cache}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_TOKEN="${HF_TOKEN:-${HUGGINGFACE_TOKEN:-}}"
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
mkdir -p "$REPO_ROOT/.openresearch/artifacts"

echo "== Installing dependencies =="
# Do NOT force-upgrade torch — it breaks the box's pinned torchvision/transformers
# (torchvision::nms / BloomPreTrainedModel import errors). The 4B runs fine in
# bf16 + LoRA on an 80GB GPU, the same proven path as the 0.6B run (no 4-bit).
pip install -q --no-input \
    "transformers>=4.51" "datasets>=2.19" "peft>=0.11" \
    "accelerate" "scikit-learn" "scipy" \
    "sentence-transformers" "emoji" "pandas" "huggingface_hub" 2>&1 | tail -5 || true

echo "== Smoke-test gated dataset access =="
python - <<'PY'
import os
from datasets import load_dataset
ds = load_dataset("pangram/editlens_iclr", split="val", token=os.environ.get("HF_TOKEN") or None)
print("OK: editlens_iclr val rows =", len(ds))
PY

# Base = the 0.6B EditLens-Qwen3 repro (search HF: "editlens qwen3 repro").
# Its encoder already understands edit-extent; we fine-tune the OOD head onto it.
# 0.6B is tiny: bigger batch, no gradient checkpointing → much faster, full model.
MODEL="${MODEL:-${BASE_MODEL:-reneeice/editlens-qwen3-0.6b-repro}}"
# Gradient checkpointing only when explicitly requested (not needed at 0.6B).
GCK_FLAG=""; [ "${GRAD_CKPT:-0}" = "1" ] && GCK_FLAG="--grad_ckpt"
echo "== Scaling OOD head on $MODEL (full dataset) =="
python editlens/train_ood.py \
    --model_name "$MODEL" \
    --repo_suffix "${REPO_SUFFIX:-ood-editguard-qwen3-0.6b-full}" \
    --out_dim "${OUT_DIM:-256}" \
    --max_length "${MAXLEN:-768}" \
    --batch_size "${BATCH:-16}" \
    --grad_accum "${GRAD_ACCUM:-1}" \
    $GCK_FLAG \
    --epochs "${EPOCHS:-2}" \
    --lr "${LR:-1e-4}" \
    --max_train "${MAX_TRAIN:-0}" \
    --max_val "${MAX_VAL:-0}" 2>&1 | tee "$REPO_ROOT/.openresearch/artifacts/train.log"

echo "== Done. =="
