from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
from datasets import load_dataset
import torchmetrics
import torch
import numpy as np
from scipy.special import softmax
import wandb
import hydra
import omegaconf
import os
import time

from preprocess import clean_text, count_words, score_to_bucket


def filter_fn(example, min_words):
    if example["text"] is None:
        return False
    return count_words(example["text"]) >= min_words


def tokenize_and_label(example, tokenizer, cfg):
    text = clean_text(example["text"])
    result = tokenizer(text, truncation=True, max_length=cfg.data.max_length)
    score = example[cfg.data.score_col]
    result["label"] = score_to_bucket(
        score, cfg.data.n_buckets, cfg.data.lo_threshold, cfg.data.hi_threshold
    )
    return result


def load_datasets(cfg, tokenizer):
    all_datasets = {}

    for split in ["train", "val"]:
        ds = load_dataset(cfg.data.path, split=split).shuffle(seed=cfg.data.seed)

        # Filter missing scores
        score_col = cfg.data.score_col
        ds = ds.filter(lambda x: x[score_col] is not None, num_proc=32)

        # Filter short texts
        ds = ds.filter(
            lambda x: filter_fn(x, cfg.data.min_words), num_proc=32
        )

        # Subsample if configured
        max_samples = cfg.data.get(f"max_{split}_samples", None)
        if max_samples is not None:
            ds = ds.select(range(min(max_samples, len(ds))))

        # Tokenize and assign labels
        ds = ds.map(
            lambda x: tokenize_and_label(x, tokenizer, cfg), num_proc=32
        )

        # Print label distribution
        label_counts = ds.to_pandas()["label"].value_counts().sort_index()
        print(f"Label distribution for '{split}':")
        for label, count in label_counts.items():
            print(f"  Bucket {label}: {count}")

        all_datasets[split] = ds

    return all_datasets


def make_compute_metrics(n_buckets):
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        probs = torch.from_numpy(softmax(preds, axis=1))
        labels = torch.from_numpy(labels)

        metrics = {
            "accuracy": torchmetrics.classification.MulticlassAccuracy(
                num_classes=n_buckets, average="micro"
            ),
            "f1": torchmetrics.F1Score(
                task="multiclass", num_classes=n_buckets, average=None
            ),
            "precision": torchmetrics.Precision(
                task="multiclass", num_classes=n_buckets, average=None
            ),
            "recall": torchmetrics.Recall(
                task="multiclass", num_classes=n_buckets, average=None
            ),
        }

        results = {}
        for name, metric in metrics.items():
            value = metric(probs, labels)
            if isinstance(value, torch.Tensor) and value.ndim == 1:
                for i, v in enumerate(value):
                    results[f"{name}_class_{i}"] = v.item()
            else:
                results[name] = value.item()

        return results

    return compute_metrics


class NormedLinear(torch.nn.Module):
    """Linear layer preceded by LayerNorm to keep logits well-scaled."""

    def __init__(self, hidden_size, num_labels, device=None, dtype=None):
        super().__init__()
        self.norm = torch.nn.LayerNorm(hidden_size, device=device, dtype=dtype)
        self.linear = torch.nn.Linear(hidden_size, num_labels, bias=False, device=device, dtype=dtype)

    def forward(self, x):
        return self.linear(self.norm(x))



def build_model(cfg, tokenizer):
    # Quantization config for QLoRA
    quantization_config = None
    if cfg.get("quantization", {}).get("enabled", False):
        compute_dtype = getattr(torch, cfg.quantization.bnb_4bit_compute_dtype)
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=cfg.quantization.load_in_4bit,
            bnb_4bit_quant_type=cfg.quantization.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
        )

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model.name,
        num_labels=cfg.data.n_buckets,
        quantization_config=quantization_config,
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    # For causal LMs (Llama, GPT, etc.), the classification head is a bare Linear
    # called `score`. Replace it with LayerNorm + Linear to stabilize initial loss.
    # Encoder models like RoBERTa already have a proper head with LayerNorm.
    if hasattr(model, "score") and isinstance(model.score, torch.nn.Linear):
        hidden_size = model.config.hidden_size
        device = next(model.parameters()).device
        model.score = NormedLinear(hidden_size, cfg.data.n_buckets, device=device)

    # LoRA / QLoRA
    if cfg.get("lora", {}).get("enabled", False):
        if quantization_config is not None:
            model = prepare_model_for_kbit_training(model)

        lora_config = LoraConfig(
            r=cfg.lora.r,
            lora_alpha=cfg.lora.lora_alpha,
            lora_dropout=cfg.lora.lora_dropout,
            bias=cfg.lora.bias,
            target_modules=list(cfg.lora.target_modules),
            task_type="SEQ_CLS",
            modules_to_save=["score"],
        )
        model = get_peft_model(model, lora_config)
        model.config.use_cache = False
        model.print_trainable_parameters()

    return model


@hydra.main(version_base=None, config_path="../configs")
def train(cfg):
    start = time.time()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)

    # W&B init on rank 0 only
    if local_rank <= 0:
        wandb.init(project=cfg.wandb.project, name=cfg.wandb.name)
        wandb.config.update(omegaconf.OmegaConf.to_container(cfg, resolve=True))
    else:
        os.environ["WANDB_MODE"] = "disabled"

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

    # Dataset
    all_datasets = load_datasets(cfg, tokenizer)

    # Model
    model = build_model(cfg, tokenizer)

    # Training
    training_args = TrainingArguments(
        output_dir=cfg.training.output_dir,
        learning_rate=cfg.training.learning_rate,
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.training.get("per_device_eval_batch_size", cfg.training.per_device_train_batch_size),
        gradient_accumulation_steps=cfg.training.get("gradient_accumulation_steps", 1),
        num_train_epochs=cfg.training.num_train_epochs,
        weight_decay=cfg.training.get("weight_decay", 0.0),
        eval_strategy=cfg.training.get("eval_strategy", "steps"),
        eval_steps=cfg.training.get("eval_steps", 500),
        save_strategy=cfg.training.get("save_strategy", "steps"),
        save_steps=cfg.training.get("save_steps", 500),
        logging_steps=cfg.training.get("logging_steps", 50),
        lr_scheduler_type=cfg.training.get("lr_scheduler_type", "linear"),
        warmup_ratio=cfg.training.get("warmup_ratio", 0.0),
        save_total_limit=cfg.training.get("save_total_limit", 2),
        load_best_model_at_end=cfg.training.get("load_best_model_at_end", False),
        metric_for_best_model=cfg.training.get("metric_for_best_model", "accuracy"),
        report_to="wandb",
        bf16=cfg.training.get("bf16", False),
        ddp_find_unused_parameters=cfg.training.get("ddp_find_unused_parameters", None),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=all_datasets["train"],
        eval_dataset=all_datasets["val"],
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=make_compute_metrics(cfg.data.n_buckets),
    )

    print(f"Startup overhead: {time.time() - start:.1f}s")
    trainer.train()

    if local_rank <= 0:
        wandb.finish()


if __name__ == "__main__":
    train()
