import numpy as np


def find_optimal_threshold(preds, labels, num_thresholds=1000):
    """
    Find the threshold that maximizes F1 score for binary classification.

    Args:
        preds: Array of prediction scores between 0 and 1.
        labels: Array of binary labels (0 or 1).
        num_thresholds: Number of evenly spaced thresholds to try.

    Returns:
        optimal_threshold: Threshold that maximizes F1 score.
        max_f1: The maximum F1 score achieved.
    """
    preds = np.asarray(preds)
    labels = np.asarray(labels)
    thresholds = np.linspace(0, 1, num_thresholds)

    best_threshold = 0.0
    best_f1 = 0.0

    for threshold in thresholds:
        pred_labels = (preds >= threshold).astype(int)

        tp = np.sum((pred_labels == 1) & (labels == 1))
        fp = np.sum((pred_labels == 1) & (labels == 0))
        fn = np.sum((pred_labels == 0) & (labels == 1))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold, best_f1
