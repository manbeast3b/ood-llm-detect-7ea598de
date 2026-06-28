#!/usr/bin/env bash
###############################################################################
# Product C — OOD guard alongside EditLens (selective prediction).
#
# Keep EditLens's edit-score, add a DeepSVDD OOD guard (arXiv 2510.08602) that
# abstains on the most out-of-distribution inputs. Evaluated as an accuracy-vs-
# coverage curve: does abstaining on the most-OOD fraction raise accuracy?
#
# Writes EVAL.md + results.json (+ ood_guard.npz) to .openresearch/artifacts/.
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
    "scikit-learn" "scipy" "emoji" "pandas" "huggingface_hub" 2>&1 | tail -5 || true

echo "== Smoke-test gated dataset access =="
python - <<'PY'
import os
from datasets import load_dataset
ds = load_dataset("pangram/editlens_iclr", split="val", token=os.environ.get("HF_TOKEN") or None)
print("OK: editlens_iclr val rows =", len(ds))
PY

MODEL="${MODEL:-reneeice/editlens-qwen3-0.6b-repro}"
echo "== Building OOD guard on frozen $MODEL =="
python editlens/ood_guard.py \
    --model_name "$MODEL" \
    --max_length "${MAXLEN:-512}" \
    --batch_size "${BATCH:-16}" \
    --max_train "${MAX_TRAIN:-3000}" \
    --max_val "${MAX_VAL:-2000}" 2>&1 | tee "$REPO_ROOT/.openresearch/artifacts/train.log"

echo "== Done. =="
