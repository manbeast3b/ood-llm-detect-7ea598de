"""
Builds the HuggingFace model card (README.md) for each of the three sibling
models. A shared "story + family navigation" block is reused across all three so
a reader landing on ANY of them understands the whole project and can navigate to
the other two. Per-model specifics (what it is, how to use it, its numbers) are
passed in.

Cross-links use "{{NS}}" placeholders, rewritten to the real account at upload.
"""

# Repo suffixes for the family — keep in sync with each run.sh.
FAMILY = {
    "A": "ood-editguard-qwen3-0.6b",
    "B": "editlens-ood-adapter-qwen3-0.6b",
    "C": "editlens-ood-selective-guard-qwen3",
}

FAMILY_TABLE = f"""
| Model | What it is | Use it when |
|---|---|---|
| [`{FAMILY['A']}`](https://huggingface.co/{{{{NS}}}}/{FAMILY['A']}) | **Standalone OOD AI-edit detector** — a Qwen3 backbone fine-tuned (QLoRA) with an out-of-distribution head; outputs a continuous "how AI-edited" score. | You want one self-contained model that scores text end-to-end. |
| [`{FAMILY['B']}`](https://huggingface.co/{{{{NS}}}}/{FAMILY['B']}) | **Tiny OOD adapter** (a few MB) that snaps onto a frozen [EditLens-Qwen3](https://huggingface.co/reneeice/editlens-qwen3-4b-repro) checkpoint to add an anomaly / human-likeness score — no backbone training. | You already run EditLens and want to add an OOD score cheaply. |
| [`{FAMILY['C']}`](https://huggingface.co/{{{{NS}}}}/{FAMILY['C']}) | **Reliability guard** for selective prediction — an OOD gate that abstains on inputs unlike the training distribution so the edit-score isn't trusted blindly. | You need calibrated, low-false-positive decisions and can abstain on hard cases. |
""".strip()


def _story_block(which):
    """The shared narrative + family navigation, with the current model marked."""
    marks = {k: (" ← **you are here**" if k == which else "") for k in FAMILY}
    fam = FAMILY_TABLE
    # annotate the "you are here" row
    for k, suffix in FAMILY.items():
        if k == which:
            fam = fam.replace(
                f"[`{suffix}`](https://huggingface.co/{{{{NS}}}}/{suffix})",
                f"[`{suffix}`](https://huggingface.co/{{{{NS}}}}/{suffix}){marks[k]}",
            )
    return f"""## The project behind this model

This model is one of a **family of three**, the end of a single research thread
that started from a classic question — *can you tell human text from machine
text?* — and ended at a more realistic one — *how much did AI edit this text, and
can we trust that judgement?*

The journey, start to finish:

1. **Reproduce "Human Texts Are Outliers."** We first reproduced the core claim of
   [arXiv:2510.08602](https://arxiv.org/abs/2510.08602) (NeurIPS 2025): instead of
   training a binary human-vs-machine classifier, model **machine text as the
   in-distribution** and treat **human text as out-of-distribution (OOD)** — an
   anomaly to be detected by distance from a learned center (DeepSVDD). A minimal
   end-to-end run on the RAID dataset hit **AUROC 0.94**, matching the paper.

2. **Meet EditLens.** Binary detection is the wrong frame for the *common* case:
   people lightly edit their own drafts with AI. [EditLens](https://arxiv.org/abs/2510.03154)
   (Thai et al., 2025) reframes detection as a **continuous "extent of AI editing"**
   score in [0,1], and the community
   [`editlens-qwen3-*-repro`](https://huggingface.co/reneeice/editlens-qwen3-4b-repro)
   models bring it to a modern **Qwen3** backbone.

3. **Apply the OOD idea to the edit-detection setting.** The insight of this work:
   take the OOD framing from step 1 and apply it to the edit-detection problem of
   step 2, on Qwen3. We pursued **three concrete ways** to do that — and shipped all
   three as a family:

{fam}

> **Why three?** They trade off cost and integration: **A** is a standalone model,
> **B** is a cheap add-on to an existing EditLens deployment, and **C** wraps either
> with an abstain-on-uncertainty safety layer. Pick the one that matches how you
> deploy.

### One thing we learned the hard way

Our first frozen-embedding run scored an AUROC of **0.32** — not random, but
*inverted*. On the EditLens embedding space the geometry is the opposite of the
original RAID setup: **human/clean text is the compact in-distribution** and
heavily-AI-edited text is the outlier (its embeddings are organized around *extent
of editing*, not authorship). We flipped the in-distribution definition, switched
from full Mahalanobis to a shrinkage-regularized / Euclidean distance on frozen
features, and added an **auto-orientation** step that fixes the score's sign on a
held-out slice so a detector is never reported upside-down. That correction is
baked into this family."""


def build_card(which, title, tags, summary, usage_md, results_md, training_md):
    """Assemble the full README for one model."""
    fm_tags = "\n".join(f"- {t}" for t in tags)
    frontmatter = f"""---
license: apache-2.0
language:
- en
library_name: transformers
pipeline_tag: text-classification
tags:
{fm_tags}
---"""
    return f"""{frontmatter}

# {title}

{summary}

{usage_md}

{results_md}

{_story_block(which)}

{training_md}

## License

Apache-2.0. Built on `Qwen/Qwen3-*-Base`. The supervision labels derive from the
gated [`pangram/editlens_iclr`](https://huggingface.co/datasets/pangram/editlens_iclr)
dataset; please honor its terms. Method credit: *Human Texts Are Outliers*
([2510.08602](https://arxiv.org/abs/2510.08602)) and *EditLens*
([2510.03154](https://arxiv.org/abs/2510.03154)).
"""
