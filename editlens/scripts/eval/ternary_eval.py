"""
Ternary evaluation for binary-score detectors.

Calibrates two thresholds on the val set to split a single detector score
into three classes: human_written, ai_edited, ai_generated.

Score direction is auto-detected: if mean(human) > mean(ai_generated) the
scores are flipped so that higher always means "more AI".
Auto-detects all *_score columns if --score_col is omitted.

Usage:
    python -m scripts.eval.ternary_eval --data_dir data --score_col binoculars_score
    python -m scripts.eval.ternary_eval --data_dir data
"""

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from scripts.eval.threshold import find_optimal_threshold


LABEL_TO_ID = {"human_written": 0, "ai_generated": 1, "ai_edited": 2}


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


def calibrate_thresholds(
    labels: np.ndarray, scaled_scores: np.ndarray
) -> tuple[float, float, float, float]:
    """Find two thresholds on the val set.

    Assumes higher score = more AI (call orient_scores first).
    Returns (human_thresh, ai_thresh, f1_human, f1_ai).
    """
    # Threshold 1: human (0) vs everything else (1, 2)
    binary_human = (labels > 0).astype(int)
    h_thresh, h_f1 = find_optimal_threshold(scaled_scores, binary_human)

    # Threshold 2: ai_generated (1) vs everything else (0, 2)
    binary_ai = (labels == 1).astype(int)
    ai_thresh, ai_f1 = find_optimal_threshold(scaled_scores, binary_ai)

    return h_thresh, ai_thresh, h_f1, ai_f1


def predict_ternary(
    scaled_scores: np.ndarray, h_thresh: float, ai_thresh: float
) -> np.ndarray:
    """Assign ternary labels based on two thresholds.

    Assumes higher score = more AI.
    """
    preds = np.full(len(scaled_scores), 2, dtype=int)  # default: ai_edited
    preds[scaled_scores < h_thresh] = 0   # human
    preds[scaled_scores > ai_thresh] = 1  # ai_generated
    return preds


def evaluate(true_labels: np.ndarray, pred_labels: np.ndarray) -> dict:
    acc = accuracy_score(true_labels, pred_labels)
    macro_f1 = f1_score(true_labels, pred_labels, average="macro")
    per_class = f1_score(true_labels, pred_labels, average=None, labels=[0, 1, 2])
    cm = confusion_matrix(true_labels, pred_labels, labels=[0, 1, 2])
    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "f1_human": per_class[0],
        "f1_ai_generated": per_class[1],
        "f1_ai_edited": per_class[2],
        "confusion_matrix": cm,
    }


def print_results(score_col: str, metrics: dict, h_thresh: float, ai_thresh: float):
    print(f"\n{'=' * 60}")
    print(f"Ternary evaluation: {score_col}")
    print(f"{'=' * 60}")
    print(f"Human threshold (scaled):       {h_thresh:.4f}")
    print(f"AI-generated threshold (scaled): {ai_thresh:.4f}")
    print(f"Accuracy:  {metrics['accuracy']:.3f}")
    print(f"Macro F1:  {metrics['macro_f1']:.3f}")
    print(f"  Human:        {metrics['f1_human']:.3f}")
    print(f"  AI-generated: {metrics['f1_ai_generated']:.3f}")
    print(f"  AI-edited:    {metrics['f1_ai_edited']:.3f}")
    print(f"\nConfusion matrix (rows=true, cols=pred):")
    print(f"              {'Human':>8} {'AI-Gen':>8} {'AI-Edit':>8}")
    for i, name in enumerate(["Human", "AI-Gen", "AI-Edit"]):
        row = metrics["confusion_matrix"][i]
        print(f"  {name:<10} {row[0]:>8} {row[1]:>8} {row[2]:>8}")


def run(data_dir: str, score_col: str | None = None, val_file: str = "val.csv", test_file: str = "test.csv"):
    val_df = pd.read_csv(os.path.join(data_dir, val_file))
    test_df = pd.read_csv(os.path.join(data_dir, test_file))

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
        val_labels = val_df["text_type"].map(LABEL_TO_ID).values
        val_scores = val_df[col].values

        # Auto-detect score direction so higher = more AI
        val_scores, flipped = orient_scores(val_scores, val_labels)
        if flipped:
            print(f"[{col}] Score direction flipped (original: higher = more human)")

        val_scaled = minmax_scale(val_scores)
        h_thresh, ai_thresh, h_f1, ai_f1 = calibrate_thresholds(val_labels, val_scaled)
        print(f"[{col}] Val calibration — human-vs-rest F1: {h_f1:.3f}, ai-vs-rest F1: {ai_f1:.3f}")

        # --- Evaluate on test ---
        test_labels = test_df["text_type"].map(LABEL_TO_ID).values
        test_scores = test_df[col].values
        if flipped:
            test_scores = -test_scores
        test_scaled = minmax_scale(test_scores)

        preds = predict_ternary(test_scaled, h_thresh, ai_thresh)
        metrics = evaluate(test_labels, preds)
        print_results(col, metrics, h_thresh, ai_thresh)

        all_results[col] = {"metrics": metrics, "h_thresh": h_thresh, "ai_thresh": ai_thresh}

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Ternary eval for binary-score detectors")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory with val.csv and test.csv")
    parser.add_argument("--score_col", type=str, default=None, help="Score column (default: all *_score columns)")
    parser.add_argument("--val_file", type=str, default="val.csv")
    parser.add_argument("--test_file", type=str, default="test.csv")
    args = parser.parse_args()

    run(args.data_dir, args.score_col, args.val_file, args.test_file)


if __name__ == "__main__":
    main()
