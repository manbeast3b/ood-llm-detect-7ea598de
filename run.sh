#!/usr/bin/env bash
###############################################################################
# Product B — OOD on FROZEN EditLens embeddings.
#
# Freeze the reneeice/editlens-qwen3 model, embed the data, fit a tiny DeepSVDD
# detector (center + whitening) on top. Tests whether the OOD framing of
# arXiv 2510.08602 recovers strong human/AI separation from EditLens features
# with no backbone training, and ships a few-KB OOD adapter.
#
# Writes EVAL.md + results.json (+ ood_adapter.npz) to .openresearch/artifacts/.
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
pip install -q --no-input \
    "torch" "transformers>=4.51" "datasets>=2.19" "accelerate" \
    "scikit-learn" "scipy" "emoji" "pandas" 2>&1 | tail -5 || true

echo "== Smoke-test gated dataset access =="
python - <<'PY'
import os
from datasets import load_dataset
ds = load_dataset("pangram/editlens_iclr", split="val", token=os.environ.get("HF_TOKEN") or None)
print("OK: editlens_iclr val rows =", len(ds))
PY

MODEL="${MODEL:-reneeice/editlens-qwen3-0.6b-repro}"
echo "== Fitting OOD adapter on frozen $MODEL =="
python editlens/train_ood_frozen.py \
    --model_name "$MODEL" \
    --max_length "${MAXLEN:-512}" \
    --batch_size "${BATCH:-16}" \
    --max_train "${MAX_TRAIN:-4000}" \
    --max_val "${MAX_VAL:-1500}" 2>&1 | tee "$REPO_ROOT/.openresearch/artifacts/train.log"

echo "== Done. =="
