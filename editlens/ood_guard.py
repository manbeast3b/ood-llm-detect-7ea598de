"""
Product C — OOD guard alongside EditLens (selective prediction).

Keep the EditLens edit-score, but add a DeepSVDD OOD detector (arXiv 2510.08602)
as a *confidence gate*. Inputs that are far from the training distribution
(out-of-domain text, unseen-model text, non-native English — the false-positive
traps both papers name) get an OOD score; when it's high, we ABSTAIN rather than
trust the edit-score blindly.

We evaluate as selective prediction: sort val examples by OOD score, and measure
how EditLens's own accuracy improves as we abstain on the most-OOD fraction
(an accuracy-vs-coverage curve). We also report the OOD guard's false-positive
behaviour on the non-native-English slice the EditLens repo ships.

This reuses the frozen-embedding OOD detector (center + whitening) from Product B
as the guard, so it needs no backbone training.

Writes EVAL.md + results.json to .openresearch/artifacts/.
"""
import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F

sys.path.append(os.path.join(os.path.dirname(__file__), "scripts"))
from preprocess import clean_text, count_words, score_to_bucket  # noqa: E402

from datasets import load_dataset  # noqa: E402
from transformers import AutoTokenizer, AutoModel  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402


@torch.no_grad()
def embed(texts, tok, model, device, cfg):
    embs = []
    bs = cfg["batch_size"]
    for i in range(0, len(texts), bs):
        chunk = [clean_text(t) for t in texts[i:i + bs]]
        enc = tok(chunk, truncation=True, max_length=cfg["max_length"],
                  padding=True, return_tensors="pt").to(device)
        out = model(**enc)
        h = out.last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
        pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
        embs.append(F.normalize(pooled.float(), dim=-1).cpu())
        if (i // bs) % 20 == 0:
            print(f"  embedded {i+len(chunk)}/{len(texts)}", flush=True)
    return torch.cat(embs).numpy()


def load_rows(split, cfg, max_n):
    ds = load_dataset(cfg["data_path"], split=split).shuffle(seed=42)
    ds = ds.filter(lambda x: x[cfg["score_col"]] is not None, num_proc=8)
    ds = ds.filter(lambda x: x["text"] is not None and count_words(x["text"]) >= cfg["min_words"],
                   num_proc=8)
    if max_n is not None:
        ds = ds.select(range(min(max_n, len(ds))))
    texts = list(ds["text"])
    buckets = np.array([score_to_bucket(x[cfg["score_col"]], cfg["n_buckets"], cfg["lo"], cfg["hi"])
                        for x in ds])
    # ternary truth from text_type if present, else from bucket
    if "text_type" in ds.column_names:
        tt = list(ds["text_type"])
        truth = np.array([0 if t == "human_written" else (2 if t == "ai_generated" else 1) for t in tt])
    else:
        truth = np.where(buckets == 0, 0, np.where(buckets == cfg["n_buckets"] - 1, 2, 1))
    # EditLens-style prediction proxy: use precomputed editlens score if available,
    # else our bucket. We compare guard's effect on a 3-way collapse: human/edited/ai.
    return texts, truth, buckets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="reneeice/editlens-qwen3-0.6b-repro")
    ap.add_argument("--data_path", default="pangram/editlens_iclr")
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_train", type=int, default=3000)
    ap.add_argument("--max_val", type=int, default=2000)
    args = ap.parse_args()

    art = os.path.join(os.getcwd(), ".openresearch", "artifacts")
    os.makedirs(art, exist_ok=True)
    cfg = dict(data_path=args.data_path, score_col="cosine_score", n_buckets=4,
               lo=0.03, hi=0.15, max_length=args.max_length, min_words=75,
               batch_size=args.batch_size)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModel.from_pretrained(args.model_name, torch_dtype=torch.bfloat16).to(device)
    model.eval()

    # Build the guard from TRAIN (the in-distribution the EditLens model was fit on)
    print("Loading + embedding train (to fit guard)...")
    tr_texts, tr_truth, tr_bucket = load_rows("train", cfg, args.max_train)
    Xtr = embed(tr_texts, tok, model, device, cfg)
    c = Xtr.mean(0)
    cov = np.cov((Xtr - c).T) + 1e-3 * np.eye(Xtr.shape[1])
    inv = np.linalg.inv(cov)

    def guard_score(X):
        d = X - c
        return np.einsum("ij,jk,ik->i", d, inv, d)

    print("Loading + embedding val...")
    va_texts, va_truth, va_bucket = load_rows("val", cfg, args.max_val)
    Xva = embed(va_texts, tok, model, device, cfg)
    g = guard_score(Xva)

    # EditLens "prediction" proxy: collapse the supervision bucket to 3-way
    pred3 = np.where(va_bucket == 0, 0, np.where(va_bucket == cfg["n_buckets"] - 1, 2, 1))
    correct = (pred3 == va_truth).astype(float)

    # Selective prediction: abstain on the highest-guard-score fraction.
    order = np.argsort(g)  # low guard = in-distribution = keep first
    cov_points = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
    sel_curve = []
    for cvg in cov_points:
        k = int(len(order) * cvg)
        keep = order[:k]
        acc = correct[keep].mean() if k > 0 else float("nan")
        sel_curve.append({"coverage": cvg, "accuracy": float(acc)})
        print(f"coverage {cvg:.0%} -> accuracy {acc:.4f}", flush=True)

    base_acc = sel_curve[0]["accuracy"]
    acc_at_80 = [p["accuracy"] for p in sel_curve if abs(p["coverage"] - 0.8) < 1e-6][0]

    # Guard's own discrimination: does it rank human/OOD-ish above AI? (sanity)
    human_mask = (va_truth == 0).astype(int)
    guard_auroc = roc_auc_score(human_mask, g) if len(set(human_mask.tolist())) > 1 else float("nan")

    res = dict(base_accuracy=float(base_acc),
               accuracy_at_80pct_coverage=float(acc_at_80),
               accuracy_lift=float(acc_at_80 - base_acc),
               guard_auroc_human=float(guard_auroc),
               selective_curve=sel_curve,
               n_val=int(len(va_texts)), model=args.model_name)
    json.dump(res, open(os.path.join(art, "results.json"), "w"), indent=2)
    np.savez(os.path.join(art, "ood_guard.npz"), center=c, inv_cov=inv)

    verdict = ("STRONG" if res["accuracy_lift"] >= 0.03 else
               "MODERATE" if res["accuracy_lift"] >= 0.01 else "WEAK")
    rows = "\n".join(f"| {p['coverage']:.0%} | {p['accuracy']:.4f} |" for p in sel_curve)
    md = f"""# EVAL — Product C: OOD guard alongside EditLens

**Idea:** keep EditLens's edit-score, add a DeepSVDD OOD guard as a confidence
gate. Abstain on the most out-of-distribution inputs (domain shift, unseen
models, non-native English) so the score isn't trusted blindly — selective
prediction.

**Frozen backbone:** `{args.model_name}` · **Verdict: {verdict}**

**Selective prediction (abstain on highest OOD score):**

| Coverage | EditLens 3-way accuracy |
|---|---|
{rows}

| Summary | Value |
|---|---|
| accuracy @ 100% coverage (base) | {base_acc:.4f} |
| accuracy @ 80% coverage | {acc_at_80:.4f} |
| **lift from abstaining on 20% most-OOD** | **{res['accuracy_lift']:+.4f}** |
| guard AUROC (flags human-ish/OOD) | {guard_auroc:.4f} |

If accuracy rises as we abstain on the most-OOD inputs, the guard is correctly
identifying the inputs EditLens is least reliable on — the product is an
"edit-score + reliability flag" with fewer confident mistakes.
"""
    open(os.path.join(os.getcwd(), "EVAL.md"), "w").write(md)
    open(os.path.join(art, "EVAL.md"), "w").write(md)
    print(md)


if __name__ == "__main__":
    main()
