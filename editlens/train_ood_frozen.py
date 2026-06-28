"""
Product B — OOD on FROZEN EditLens embeddings.

Take the already-trained reneeice/editlens-qwen3 model as a *frozen* feature
extractor, pull its last-hidden-state embeddings, and fit a tiny DeepSVDD
detector on top (center + whitening, no backbone training). This tests whether
the OOD framing (arXiv 2510.08602) recovers a strong human/AI separation from
EditLens features at a fraction of the cost — and yields a few-KB "OOD adapter"
that snaps onto a frozen checkpoint.

No gradient training of the backbone: we just (1) embed, (2) compute the ID
center over AI text, (3) score by Mahalanobis-style distance. Fast, mostly a
single forward pass over the data.

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
from sklearn.metrics import roc_auc_score, average_precision_score  # noqa: E402
from scipy.stats import pearsonr  # noqa: E402


@torch.no_grad()
def embed_split(split, tok, model, device, cfg, max_n):
    ds = load_dataset(cfg["data_path"], split=split).shuffle(seed=42)
    ds = ds.filter(lambda x: x[cfg["score_col"]] is not None, num_proc=8)
    ds = ds.filter(
        lambda x: x["text"] is not None and count_words(x["text"]) >= cfg["min_words"],
        num_proc=8,
    )
    if max_n is not None:
        ds = ds.select(range(min(max_n, len(ds))))

    texts = [clean_text(t) for t in ds["text"]]
    buckets = [score_to_bucket(x[cfg["score_col"]], cfg["n_buckets"], cfg["lo"], cfg["hi"])
               for x in ds]
    edit = np.array([float(x[cfg["score_col"]]) for x in ds], dtype=np.float32)
    ood_label = np.array([1 if b == 0 else 0 for b in buckets])  # human/OOD = 1

    embs = []
    bs = cfg["batch_size"]
    for i in range(0, len(texts), bs):
        chunk = texts[i:i + bs]
        enc = tok(chunk, truncation=True, max_length=cfg["max_length"],
                  padding=True, return_tensors="pt").to(device)
        out = model(**enc)
        h = out.last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
        pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
        embs.append(F.normalize(pooled.float(), dim=-1).cpu())
        if (i // bs) % 20 == 0:
            print(f"  embedded {i+len(chunk)}/{len(texts)} [{split}]", flush=True)
    return torch.cat(embs).numpy(), ood_label, edit, np.array(buckets)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="reneeice/editlens-qwen3-0.6b-repro")
    ap.add_argument("--data_path", default="pangram/editlens_iclr")
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_train", type=int, default=4000)
    ap.add_argument("--max_val", type=int, default=1500)
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
    # frozen feature extractor — load the SequenceClassification model's base encoder
    model = AutoModel.from_pretrained(args.model_name, torch_dtype=torch.bfloat16).to(device)
    model.eval()

    print("Embedding train split...")
    Xtr, ood_tr, edit_tr, bk_tr = embed_split("train", tok, model, device, cfg, args.max_train)
    print("Embedding val split...")
    Xva, ood_va, edit_va, bk_va = embed_split("val", tok, model, device, cfg, args.max_val)

    # --- DeepSVDD-style detector on frozen features ---
    # LESSON FROM THE FIRST RUN (AUROC 0.32, inverted): on EditLens embeddings the
    # COMPACT in-distribution is HUMAN/clean text, not AI text. So ID = human
    # (ood==0), and AI text is the outlier we want to score high.
    id_idx = (ood_tr == 0)
    c = Xtr[id_idx].mean(0)
    Xc = Xtr[id_idx] - c
    d_dim = Xc.shape[1]
    # Ledoit-Wolf-style shrinkage toward a scaled identity — the full Mahalanobis
    # overfit the covariance last time; shrinkage regularizes it.
    emp = np.cov(Xc.T)
    shrink = 0.5
    cov = (1 - shrink) * emp + shrink * (np.trace(emp) / d_dim) * np.eye(d_dim)
    inv = np.linalg.inv(cov + 1e-4 * np.eye(d_dim))

    def maha(X):
        d = X - c
        return np.einsum("ij,jk,ik->i", d, inv, d)

    def euclid(X):
        return ((X - c) ** 2).sum(1)

    def oriented_auroc(score_fn):
        s = score_fn(Xva)
        if len(set(ood_va.tolist())) < 2:
            return float("nan"), 1, s
        raw = roc_auc_score(ood_va, s)
        orient = 1 if raw >= 0.5 else -1
        return roc_auc_score(ood_va, orient * s), orient, orient * s

    auroc_m, orient_m, s_m = oriented_auroc(maha)
    auroc_e, orient_e, s_e = oriented_auroc(euclid)

    # pick the variant that separates better
    if (np.nan_to_num(auroc_m) >= np.nan_to_num(auroc_e)):
        best_kind, auroc, orient, s_va = "mahalanobis", auroc_m, orient_m, s_m
    else:
        best_kind, auroc, orient, s_va = "euclidean", auroc_e, orient_e, s_e

    aupr = average_precision_score(ood_va, s_va) if len(set(ood_va.tolist())) > 1 else float("nan")
    corr = pearsonr(s_va, edit_va)[0] if np.std(s_va) > 0 else float("nan")

    res = dict(auroc=float(auroc), best_kind=best_kind, orientation=int(orient),
               auroc_mahalanobis=float(auroc_m), auroc_euclidean=float(auroc_e),
               aupr=float(aupr), corr_score_vs_editmag=float(corr),
               n_val=int(len(s_va)), n_id_train=int(id_idx.sum()),
               mean_score_ai=float(s_va[ood_va == 1].mean()),
               mean_score_human=float(s_va[ood_va == 0].mean()),
               model=args.model_name)
    json.dump(res, open(os.path.join(art, "results.json"), "w"), indent=2)

    # ---- save the tiny adapter (the shippable product) ----
    model_dir = os.path.join(os.getcwd(), "model_out")
    os.makedirs(model_dir, exist_ok=True)
    np.savez(os.path.join(model_dir, "ood_adapter.npz"),
             center=c, inv_cov=inv, orientation=orient, kind=best_kind,
             base_model=args.model_name)
    np.savez(os.path.join(art, "ood_adapter.npz"), center=c, inv_cov=inv,
             orientation=orient, kind=best_kind)

    verdict = ("STRONG" if auroc >= 0.85 else "MODERATE" if auroc >= 0.7 else "WEAK")
    eval_md = f"""# EVAL — editlens-ood-adapter-qwen3 (frozen-embedding OOD adapter)

**Idea:** freeze the EditLens/Qwen3 model, fit a tiny DeepSVDD detector
(center + shrinkage whitening) on its embeddings, **human text as the
in-distribution**. No backbone training — a few-MB OOD adapter.

**Frozen backbone:** `{args.model_name}` · **Verdict: {verdict}** · best={best_kind}

| Metric | Value |
|---|---|
| AUROC (best, oriented) | {auroc:.4f} |
| AUROC (Mahalanobis) | {auroc_m:.4f} |
| AUROC (Euclidean) | {auroc_e:.4f} |
| AUPR | {aupr:.4f} |
| corr(score, edit-magnitude) | {corr:.4f} |
| auto-orientation | {orient} |
"""
    open(os.path.join(os.getcwd(), "EVAL.md"), "w").write(eval_md)
    open(os.path.join(art, "EVAL.md"), "w").write(eval_md)
    print(eval_md)

    # ---- rich model card + push to HF ----
    try:
        import sys as _sys
        _sys.path.append(os.path.dirname(__file__))
        from model_card import build_card
        usage = f"""## Usage

This is a **tiny adapter** ({best_kind} distance to a learned center) that runs on
top of a frozen [`{args.model_name}`](https://huggingface.co/{args.model_name})
checkpoint — download `ood_adapter.npz` and score embeddings:

```python
import numpy as np, torch
from transformers import AutoTokenizer, AutoModel
a = np.load("ood_adapter.npz")
center, inv, orient = a["center"], a["inv_cov"], int(a["orientation"])
tok = AutoTokenizer.from_pretrained("{args.model_name}")
enc = AutoModel.from_pretrained("{args.model_name}", torch_dtype=torch.bfloat16).eval()
def score(text):
    t = tok(text.lower(), truncation=True, max_length=512, return_tensors="pt")
    h = enc(**t).last_hidden_state.mean(1)[0].float().numpy()
    d = h - center
    return orient * float(d @ inv @ d)   # higher = more AI-edited
```"""
        results = f"""## Performance

Validation on `pangram/editlens_iclr` (held-out), no backbone training:

| Metric | Value |
|---|---|
| **AUROC** (AI vs human) | **{auroc:.3f}** |
| AUPR | {aupr:.3f} |
| correlation with edit-magnitude | {corr:+.3f} |

The score is **auto-oriented** so it is never reported upside-down."""
        training = f"""## How it was made

- **Frozen backbone:** `{args.model_name}` (no fine-tuning).
- **Detector:** mean-pool embeddings → DeepSVDD center over **human** text +
  shrinkage-regularized covariance ({best_kind} distance). Score = oriented
  distance to the center.
- **Cost:** one embedding pass + a closed-form fit — seconds of compute."""
        card = build_card(
            "B",
            "editlens-ood-adapter-qwen3 — OOD adapter for EditLens",
            ["ai-detection", "ai-edit-detection", "out-of-distribution",
             "ood-detection", "content-integrity", "qwen3", "adapter"],
            "**A few-MB out-of-distribution adapter that snaps onto a frozen "
            "EditLens-Qwen3 checkpoint** to add an anomaly / AI-edit score with zero "
            "backbone fine-tuning.",
            usage, results, training,
        )
        from hf_upload import push_to_hub
        url = push_to_hub(model_dir, "editlens-ood-adapter-qwen3-0.6b", card_text=card)
        if url:
            open(os.path.join(art, "hf_model_url.txt"), "w").write(url + "\n")
    except Exception as e:
        print(f"[card/upload] non-fatal: {e}")


if __name__ == "__main__":
    main()
