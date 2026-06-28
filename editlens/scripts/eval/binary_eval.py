"""
Binary evaluation for detectors with a single score column.

Three evaluation modes (controlled by --mode):
  human_vs_ai:   Filter out ai_edited, classify human (0) vs AI (1).
  human_vs_rest: All rows — human (0) vs rest (ai_edited + ai_generated → 1).
  ai_vs_rest:    All rows — ai_generated (1) vs rest (human + ai_edited → 0).

Finds the optimal threshold on the val set (maximizing F1), then evaluates
on the test set. Auto-detects all *_score columns if --score_col is omitted.

Usage:
    python -m scripts.eval.binary_eval --data_dir data --score_col binoculars_score
    python -m scripts.eval.binary_eval --data_dir data --mode human_vs_rest
    python -m scripts.eval.binary_eval --data_dir data --mode ai_vs_rest
"""

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from scripts.eval.threshold import find_optimal_threshold

MODES = ("human_vs_ai", "human_vs_rest", "ai_vs_rest")


def minmax_scale(scores: np.ndarray) -> np.ndarray:
    lo, hi = scores.min(), scores.max()
    if hi == lo:
        return np.zeros_like(scores)
    return (scores - lo) / (hi - lo)


def orient_scores(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, bool]:
    """Ensure higher score = more AI. Returns (oriented_scores, was_flipped)."""
    human_mean = scores[labels == 0].mean()
    ai_mean = scores[labels == 1].mean()
    if human_mean > ai_mean:
        return -scores, True
    return scores, False


def binarize_labels(raw_labels: np.ndarray, mode: str) -> np.ndarray:
    """Convert raw labels (0=human, 1=ai, -1=ai_edited) to binary for the given mode."""
    if mode == "human_vs_ai":
        return raw_labels.copy()
    elif mode == "human_vs_rest":
        # human=0, everything else=1
        return (raw_labels != 0).astype(int)
    elif mode == "ai_vs_rest":
        # ai_generated=1, everything else=0
        return (raw_labels == 1).astype(int)


def evaluate(true_labels: np.ndarray, pred_labels: np.ndarray) -> dict:
    acc = accuracy_score(true_labels, pred_labels)
    macro_f1 = f1_score(true_labels, pred_labels, average="macro")
    per_class = f1_score(true_labels, pred_labels, average=None, labels=[0, 1])
    cm = confusion_matrix(true_labels, pred_labels, labels=[0, 1])

    # FPR = FP / (FP + TN) — class-0 samples misclassified as class-1
    fp = cm[0, 1]
    tn = cm[0, 0]
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    # FNR = FN / (FN + TP) — class-1 samples misclassified as class-0
    fn = cm[1, 0]
    tp = cm[1, 1]
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "f1_class0": per_class[0],
        "f1_class1": per_class[1],
        "fpr": fpr,
        "fnr": fnr,
        "confusion_matrix": cm,
    }


MODE_CLASS_NAMES = {
    "human_vs_ai": ("Human", "AI"),
    "human_vs_rest": ("Human", "AI/Edited"),
    "ai_vs_rest": ("Human/Edited", "AI"),
}


def print_results(score_col: str, mode: str, metrics: dict, threshold: float):
    c0, c1 = MODE_CLASS_NAMES[mode]
    print(f"\n{'=' * 60}")
    print(f"Binary evaluation: {score_col}  [{mode}]")
    print(f"{'=' * 60}")
    print(f"Threshold (scaled): {threshold:.4f}")
    print(f"Accuracy:  {metrics['accuracy']:.3f}")
    print(f"Macro F1:  {metrics['macro_f1']:.3f}")
    print(f"  {c0}: {metrics['f1_class0']:.3f}")
    print(f"  {c1}: {metrics['f1_class1']:.3f}")
    if mode == "human_vs_ai":
        print(f"FPR: {metrics['fpr']:.3f}")
        print(f"FNR: {metrics['fnr']:.3f}")
    print(f"\nConfusion matrix (rows=true, cols=pred):")
    print(f"              {c0:>12} {c1:>12}")
    for i, name in enumerate([c0, c1]):
        row = metrics["confusion_matrix"][i]
        print(f"  {name:<12} {row[0]:>12} {row[1]:>12}")


def run(
    data_dir: str,
    score_col: str | None = None,
    mode: str = "human_vs_ai",
    val_file: str = "val.csv",
    test_file: str = "test.csv",
):
    val_df = pd.read_csv(os.path.join(data_dir, val_file))
    test_df = pd.read_csv(os.path.join(data_dir, test_file))

    # For human_vs_ai mode, filter out ai_edited rows
    if mode == "human_vs_ai":
        val_df = val_df[val_df["label"] != -1].copy()
        test_df = test_df[test_df["label"] != -1].copy()

    # Auto-detect score columns if none specified
    if score_col is not None:
        score_cols = [score_col]
    else:
        score_cols = [c for c in val_df.columns if c.endswith("_score") and c not in ("cosine_score", "soft_ngrams_score")]
        if not score_cols:
            raise ValueError("No *_score columns found in val set.")
        print(f"Found score columns: {score_cols}")

    all_results = {}

    for col in score_cols:
        for df, name in [(val_df, val_file), (test_df, test_file)]:
            if col not in df.columns:
                raise ValueError(f"{col} not found in {name}. Available: {[c for c in df.columns if c.endswith('_score')]}")

        # --- Calibrate on val ---
        val_binary = binarize_labels(val_df["label"].values, mode)
        val_scores = val_df[col].values

        val_scores, flipped = orient_scores(val_scores, val_binary)
        if flipped:
            print(f"[{col}] Score direction flipped (original: higher = more human)")

        # Min-max scale using val's range, apply same to test
        val_min, val_max = val_scores.min(), val_scores.max()
        if val_max == val_min:
            val_scaled = np.zeros_like(val_scores)
        else:
            val_scaled = (val_scores - val_min) / (val_max - val_min)

        threshold, val_f1 = find_optimal_threshold(val_scaled, val_binary)
        print(f"[{col}] Val calibration — F1: {val_f1:.3f}, threshold: {threshold:.4f}")

        # --- Evaluate on test ---
        test_binary = binarize_labels(test_df["label"].values, mode)
        test_scores = test_df[col].values
        if flipped:
            test_scores = -test_scores

        if val_max == val_min:
            test_scaled = np.zeros_like(test_scores)
        else:
            test_scaled = (test_scores - val_min) / (val_max - val_min)

        preds = (test_scaled >= threshold).astype(int)
        metrics = evaluate(test_binary, preds)
        print_results(col, mode, metrics, threshold)

        all_results[col] = {"metrics": metrics, "threshold": threshold}

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Binary eval for score-based detectors")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory with val.csv and test.csv")
    parser.add_argument("--score_col", type=str, default=None, help="Score column (default: all *_score columns)")
    parser.add_argument("--mode", type=str, default="human_vs_ai", choices=MODES,
                        help="human_vs_ai (filter edited), human_vs_rest, or ai_vs_rest")
    parser.add_argument("--val_file", type=str, default="val.csv")
    parser.add_argument("--test_file", type=str, default="test.csv")
    args = parser.parse_args()

    run(args.data_dir, args.score_col, args.mode, args.val_file, args.test_file)


if __name__ == "__main__":
    main()
