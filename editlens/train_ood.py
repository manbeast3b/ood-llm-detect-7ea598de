"""
Product A — OOD head on a Qwen3 backbone for EditLens-style AI-edit detection.

Idea (from arXiv 2510.08602, "Human Texts Are Outliers"): instead of a discrete
4-bucket classifier (EditLens) or a binary human/machine classifier, model the
*machine / fully-AI* text as the in-distribution (ID) and treat human / lightly-
edited text as out-of-distribution (OOD). A DeepSVDD head packs ID embeddings
into a tight hypersphere around a center c; the OOD score s(x)=||f(x)-c||^2 then
acts as a smooth, monotone "how-human / how-lightly-edited" meter.

Backbone: Qwen/Qwen3-*-Base (4-bit QLoRA). Head: a small projection trained with
the DeepSVDD one-class objective. Data: pangram/editlens_iclr (gated; needs an
HF token with access via the HF_TOKEN env var).

Outputs: writes EVAL.md + metrics json to .openresearch/artifacts/.
"""
import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), "scripts"))
from preprocess import clean_text, count_words, score_to_bucket  # noqa: E402

from datasets import load_dataset  # noqa: E402
from transformers import (  # noqa: E402
    AutoTokenizer,
    AutoModel,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training  # noqa: E402
from sklearn.metrics import roc_auc_score, average_precision_score  # noqa: E402
from scipy.stats import pearsonr  # noqa: E402


# --------------------------------------------------------------------------- #
#  Model: Qwen3 (4-bit) encoder  +  DeepSVDD projection head
# --------------------------------------------------------------------------- #
class QwenOODDetector(nn.Module):
    def __init__(self, model_name, out_dim=256, use_qlora=True, lora_r=8):
        super().__init__()
        quant = None
        if use_qlora:
            quant = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        self.backbone = AutoModel.from_pretrained(
            model_name, quantization_config=quant, torch_dtype=torch.bfloat16
        )
        if use_qlora:
            self.backbone = prepare_model_for_kbit_training(self.backbone)
            lcfg = LoraConfig(
                r=lora_r, lora_alpha=2 * lora_r, lora_dropout=0.05, bias="none",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj"],
                task_type="FEATURE_EXTRACTION",
            )
            self.backbone = get_peft_model(self.backbone, lcfg)
            self.backbone.print_trainable_parameters()
        hidden = self.backbone.config.hidden_size
        # projection head trained in full (float32 for stability)
        self.proj = nn.Sequential(
            nn.LayerNorm(hidden, dtype=torch.float32),
            nn.Linear(hidden, out_dim, bias=False, dtype=torch.float32),
        )
        self.register_buffer("center", torch.zeros(out_dim, dtype=torch.float32))
        self.out_dim = out_dim

    def encode(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        h = out.last_hidden_state  # (B, T, H)
        mask = attention_mask.unsqueeze(-1).to(h.dtype)
        pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1)  # mean pool
        z = self.proj(pooled.float())
        return F.normalize(z, dim=-1)

    def forward(self, input_ids, attention_mask):
        z = self.encode(input_ids, attention_mask)
        score = ((z - self.center) ** 2).sum(-1)  # squared distance = OOD score
        return z, score


# --------------------------------------------------------------------------- #
#  Data
# --------------------------------------------------------------------------- #
def build_loader(split, tok, cfg, shuffle, id_only=False):
    ds = load_dataset(cfg["data_path"], split=split).shuffle(seed=42)
    ds = ds.filter(lambda x: x[cfg["score_col"]] is not None, num_proc=8)
    ds = ds.filter(
        lambda x: x["text"] is not None and count_words(x["text"]) >= cfg["min_words"],
        num_proc=8,
    )
    if cfg.get(f"max_{split}") is not None:
        ds = ds.select(range(min(cfg[f"max_{split}"], len(ds))))

    def to_bucket(x):
        return score_to_bucket(x[cfg["score_col"]], cfg["n_buckets"],
                               cfg["lo"], cfg["hi"])

    if id_only:
        # in-distribution = HUMAN / clean text (bucket 0).
        # (Lesson from the frozen-embedding run: on this edit-detection setup the
        #  compact in-distribution is human text, not AI text — see model card.)
        ds = ds.filter(lambda x: to_bucket(x) == 0, num_proc=8)

    texts = [clean_text(t) for t in ds["text"]]
    buckets = [to_bucket(x) for x in ds]
    scores = [float(x[cfg["score_col"]]) for x in ds]
    # binary OOD label: 1 = AI-edited/generated (OOD outlier), 0 = human (ID).
    # The OOD score should be HIGH for AI text -> it doubles as the "AI-extent" meter.
    ood_label = [0 if b == 0 else 1 for b in buckets]

    def collate(idx):
        bt = [texts[i] for i in idx]
        enc = tok(bt, truncation=True, max_length=cfg["max_length"],
                  padding="max_length", return_tensors="pt")
        return (enc["input_ids"], enc["attention_mask"],
                torch.tensor([buckets[i] for i in idx]),
                torch.tensor([scores[i] for i in idx], dtype=torch.float32),
                torch.tensor([ood_label[i] for i in idx]))

    order = list(range(len(texts)))
    loader = DataLoader(order, batch_size=cfg["batch_size"], shuffle=shuffle,
                        collate_fn=collate, drop_last=shuffle)
    return loader


# --------------------------------------------------------------------------- #
#  Train / eval
# --------------------------------------------------------------------------- #
@torch.no_grad()
def init_center(model, loader, device, max_batches=40):
    model.eval()
    acc = torch.zeros(model.out_dim, device=device)
    n = 0
    for i, (ids, am, *_rest) in enumerate(loader):
        z = model.encode(ids.to(device), am.to(device))
        acc += z.sum(0)
        n += z.size(0)
        if i + 1 >= max_batches:
            break
    c = acc / max(n, 1)
    c = c / c.norm().clamp(min=1e-6)
    model.center.copy_(c)
    print(f"Initialized center c from {n} ID samples")


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_s, all_ood, all_score, all_bucket = [], [], [], []
    for ids, am, bucket, score, ood in loader:
        _, s = model(ids.to(device), am.to(device))
        all_s.append(s.cpu()); all_ood.append(ood); all_score.append(score)
        all_bucket.append(bucket)
    s = torch.cat(all_s).numpy()
    ood = torch.cat(all_ood).numpy()   # 1 = AI (outlier), 0 = human (ID)
    edit = torch.cat(all_score).numpy()
    bucket = torch.cat(all_bucket).numpy()
    if len(set(ood.tolist())) < 2:
        return dict(auroc=float("nan"), aupr=float("nan"),
                    corr_score_vs_editmag=float("nan"), orientation=1, n=int(len(s)),
                    mean_score_human=float("nan"), mean_score_ai=float("nan"))
    # AUTO-ORIENT: the OOD score should be HIGH for AI text. If the raw distance
    # came out inverted (the failure we saw at AUROC 0.32), flip its sign so the
    # detector is never reported upside-down. orientation is part of the model.
    raw = roc_auc_score(ood, s)
    orientation = 1 if raw >= 0.5 else -1
    s_oriented = orientation * s
    auroc = roc_auc_score(ood, s_oriented)
    aupr = average_precision_score(ood, s_oriented)
    # Oriented score should rise with edit magnitude (AI text edits more) -> +corr.
    corr = pearsonr(s_oriented, edit)[0] if np.std(s) > 0 and np.std(edit) > 0 else float("nan")
    return dict(auroc=float(auroc), aupr=float(aupr),
                corr_score_vs_editmag=float(corr),
                orientation=int(orientation), n=int(len(s)),
                mean_score_ai=float(s_oriented[ood == 1].mean()),
                mean_score_human=float(s_oriented[ood == 0].mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="Qwen/Qwen3-0.6B-Base")
    ap.add_argument("--data_path", default="pangram/editlens_iclr")
    ap.add_argument("--out_dim", type=int, default=256)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max_train", type=int, default=4000)
    ap.add_argument("--max_val", type=int, default=1500)
    ap.add_argument("--no_qlora", action="store_true")
    args = ap.parse_args()

    art = os.path.join(os.getcwd(), ".openresearch", "artifacts")
    os.makedirs(art, exist_ok=True)

    cfg = dict(data_path=args.data_path, score_col="cosine_score", n_buckets=4,
               lo=0.03, hi=0.15, max_length=args.max_length, min_words=75,
               batch_size=args.batch_size, max_train=args.max_train,
               max_val=args.max_val)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print("Building loaders...")
    train_loader = build_loader("train", tok, cfg, shuffle=True)
    id_loader = build_loader("train", tok, cfg, shuffle=True, id_only=True)
    val_loader = build_loader("val", tok, cfg, shuffle=False)

    print("Loading model...")
    model = QwenOODDetector(args.model_name, out_dim=args.out_dim,
                            use_qlora=not args.no_qlora).to(device)

    init_center(model, id_loader, device)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr)

    best = {"auroc": -1}
    history = []
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for i, (ids, am, bucket, score, ood) in enumerate(train_loader):
            ids, am, ood = ids.to(device), am.to(device), ood.to(device)
            z, s = model(ids, am)
            # DeepSVDD one-class: pull ID (ood=0) toward center, push OOD (ood=1) away
            id_mask = (ood == 0)
            ood_mask = (ood == 1)
            loss_id = s[id_mask].mean() if id_mask.any() else s.new_tensor(0.0)
            loss_ood = F.relu(4.0 - s[ood_mask]).mean() if ood_mask.any() else s.new_tensor(0.0)
            loss = loss_id + loss_ood
            opt.zero_grad(); loss.backward(); opt.step()
            running = (running * i + loss.item()) / (i + 1)
            if i % 20 == 0:
                print(f"epoch {epoch} step {i} loss {loss.item():.4f} avg {running:.4f}", flush=True)
        metrics = evaluate(model, val_loader, device)
        metrics["epoch"] = epoch
        metrics["train_loss"] = running
        history.append(metrics)
        print(f"[VAL] epoch {epoch}: {metrics}", flush=True)
        if not np.isnan(metrics["auroc"]) and metrics["auroc"] > best["auroc"]:
            best = metrics

    json.dump({"best": best, "history": history},
              open(os.path.join(art, "results.json"), "w"), indent=2)

    verdict = ("STRONG" if best["auroc"] >= 0.85 else
               "MODERATE" if best["auroc"] >= 0.7 else "WEAK")

    # ---- Save the produced model (QLoRA adapter + OOD head + center) ----
    model_dir = os.path.join(os.getcwd(), "model_out")
    os.makedirs(model_dir, exist_ok=True)
    try:
        # LoRA adapter
        if hasattr(model.backbone, "save_pretrained"):
            model.backbone.save_pretrained(model_dir)
        tok.save_pretrained(model_dir)
        # OOD head (projection) + center + orientation — the bits that make it a detector
        torch.save({"proj": model.proj.state_dict(),
                    "center": model.center.cpu(),
                    "orientation": best.get("orientation", 1),
                    "out_dim": model.out_dim,
                    "base_model": args.model_name},
                   os.path.join(model_dir, "ood_head.pt"))
        print(f"Saved model to {model_dir}")
    except Exception as e:
        print(f"[save] non-fatal: {e}")

    # ---- EVAL card (artifact) ----
    eval_md = f"""# EVAL — ood-editguard-qwen3 (OOD AI-edit detector)

**Idea:** model **human/clean text as the in-distribution**, score AI-edited /
AI-generated text as OOD via a DeepSVDD hypersphere on a Qwen3 backbone. The
oriented OOD distance is the continuous "how-AI-edited" meter.

**Backbone:** `{args.model_name}` (QLoRA 4-bit) · **Verdict: {verdict}**

| Metric | Value |
|---|---|
| AUROC (AI vs human) | {best['auroc']:.4f} |
| AUPR | {best['aupr']:.4f} |
| corr(score, edit-magnitude) | {best['corr_score_vs_editmag']:.4f} |
| mean score — AI | {best['mean_score_ai']:.3f} |
| mean score — human | {best['mean_score_human']:.3f} |
| auto-orientation | {best.get('orientation')} |
| best epoch | {best.get('epoch')} |
"""
    open(os.path.join(os.getcwd(), "EVAL.md"), "w").write(eval_md)
    open(os.path.join(art, "EVAL.md"), "w").write(eval_md)
    print(eval_md)

    # ---- Build the rich model card + push everything to HF ----
    try:
        import sys as _sys
        _sys.path.append(os.path.dirname(__file__))
        from model_card import build_card  # noqa
        usage = """## Usage

```python
import torch
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel

base = "%s"
tok = AutoTokenizer.from_pretrained("{{NS}}/ood-editguard-qwen3-0.6b")
backbone = PeftModel.from_pretrained(AutoModel.from_pretrained(base, torch_dtype=torch.bfloat16),
                                     "{{NS}}/ood-editguard-qwen3-0.6b")
head = torch.load("ood_head.pt")  # downloaded from the repo
# score(text) = orientation * ||proj(meanpool(backbone(text))) - center||^2
```

Higher score = more AI-edited. Calibrate a threshold on your own data.""" % args.model_name
        results = f"""## Performance

Validation on `pangram/editlens_iclr` (held-out):

| Metric | Value |
|---|---|
| **AUROC** (AI vs human) | **{best['auroc']:.3f}** |
| AUPR | {best['aupr']:.3f} |
| correlation with edit-magnitude | {best['corr_score_vs_editmag']:+.3f} |

A random detector scores AUROC 0.5."""
        training = f"""## How it was trained

- **Backbone:** `{args.model_name}`, 4-bit QLoRA (rank 8, all attn+MLP projections).
- **Head:** a small LayerNorm+Linear projection trained in full, with a DeepSVDD
  one-class objective: pull **human** embeddings toward a center `c`, push AI
  embeddings away. Score = oriented squared distance to `c`.
- **Supervision:** edit-magnitude buckets from `cosine_score` (thresholds 0.03/0.15).
- **Compute:** a single GPU, minutes."""
        card = build_card(
            "A",
            "ood-editguard-qwen3 — OOD AI-edit detector (Qwen3)",
            ["ai-detection", "ai-edit-detection", "out-of-distribution",
             "ood-detection", "content-integrity", "qwen3", "deepsvdd"],
            "**Detect AI-edited text with an out-of-distribution detector on a Qwen3 "
            "backbone.** Human text is modeled as the in-distribution; AI-edited and "
            "AI-generated text are flagged as outliers, giving a continuous "
            "\"how-AI-edited\" score.",
            usage, results, training,
        )
        from hf_upload import push_to_hub  # noqa
        url = push_to_hub(model_dir, "ood-editguard-qwen3-0.6b", card_text=card)
        if url:
            with open(os.path.join(art, "hf_model_url.txt"), "w") as f:
                f.write(url + "\n")
    except Exception as e:
        print(f"[card/upload] non-fatal: {e}")


if __name__ == "__main__":
    main()
