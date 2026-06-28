# EditLens

This is the accompanying repository for the ICLR 2026 paper [EditLens: Quantifying the Extent of AI Editing in Text](https://arxiv.org/abs/2510.03154), which is the first paper to formalize the task of scoring text according to the extent of AI intervention in the text, as opposed to prior work that treated AI text detection as a binary (or occasionally ternary) classification task.

## Links

- **Paper:** [arXiv:2510.03154](https://arxiv.org/abs/2510.03154)
- **Models:** [pangram/models on HuggingFace](https://huggingface.co/pangram/models)
- **Dataset:** [pangram/editlens_iclr on HuggingFace](https://huggingface.co/datasets/pangram/editlens_iclr)

## Setup

```bash
pip install -r requirements.txt
```

## Training

Configs for both models are in `configs/`. The effective batch size for both models is 24.

### RoBERTa-Large (Single GPU)

```bash
python scripts/train.py -cn roberta
```

### Llama-3.2-3B QLoRA (8 GPUs)

Note that per-device batch size is 3 across 8 GPUs for an effective batch size of 24. Adjust if you are using fewer than 8 GPUs.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node 8 scripts/train.py -cn llama
```

## Inference

Run inference on any HuggingFace dataset (remote or local). The script adds two columns to the output:
- `bucket_pred`: predicted bucket (int)
- `score_pred`: continuous score in [0, 1], the expected value of the bucket distribution

```bash
python scripts/inference.py \
  --checkpoint pangram/editlens_roberta-large \
  --model_name FacebookAI/roberta-large \
  --max_length 512 \
  --dataset pangram/editlens_iclr \
  --split test \
  --text_col text \
  --output predictions.jsonl
```

You can train an EditLens model with any number of classification buckets! This script will infer the num_buckets hyperparameter automatically from the model checkpoint.

## Scoring

The `scripts/scoring/` directory contains standalone scripts for computing the two text-distance metrics described in the paper. These require the `sentence-transformers` package (included in `requirements.txt`).

### Cosine Distance

Computes cosine distance between two texts using a sentence embedding model. Returns a value where 0 = identical.

```bash
python scripts/scoring/cosine_distance.py \
  --text1 "The original text" \
  --text2 "The edited text" \
  --model_name Qwen/Qwen3-Embedding-0.6B
```

Or from Python:

```python
from scripts.scoring.cosine_distance import cosine_distance

score = cosine_distance(text1, text2, model_name="Qwen/Qwen3-Embedding-0.6B")
```

### Soft N-Gram Score

Measures how much of the edited text's phrase content is new relative to the source. Extracts word n-grams from both texts, embeds them, and computes what fraction of edited-text phrases have no semantic match in the source. Returns a value in [0, 1] where 0 = fully overlapping and 1 = no overlap.

```bash
python scripts/scoring/soft_ngrams.py \
  --source_text "The original text" \
  --edited_text "The edited text" \
  --threshold 0.8 \
  --min_length 6 \
  --max_length 12
```

Or from Python:

```python
from scripts.scoring.soft_ngrams import soft_ngram_score

score = soft_ngram_score(source_text, edited_text, threshold=0.8, min_length=6, max_length=12)
```

## Citation
If you use the code, dataset, or models mentioned in this repository, please cite our paper as follows:

```bibtex
@misc{thai2025editlensquantifyingextentai,
      title={EditLens: Quantifying the Extent of AI Editing in Text},
      author={Katherine Thai and Bradley Emi and Elyas Masrour and Mohit Iyyer},
      year={2025},
      eprint={2510.03154},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2510.03154},
}
```

## License

This work is licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
