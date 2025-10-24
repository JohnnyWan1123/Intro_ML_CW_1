"""Utility functions for evaluating decision trees via cross-validation.

This module implements the Step 3 evaluation requirements for the coursework:
- An `evaluate` function that returns the accuracy of a trained tree on a test set.
- Helpers for computing confusion matrices and derived classification metrics.
- A k-fold cross-validation routine tailored to the WiFi room classification task.

Only NumPy and the existing decision tree implementation are used, adhering to the
coursework constraints.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Optional

from decision_tree_builder import decision_tree_learning
from tree_node import TreeNode


def evaluate(test_db: np.ndarray, trained_tree: TreeNode) -> float:
    """Compute the accuracy of a trained tree on the supplied test dataset.

    Args:
        test_db: Array with features in the first columns and labels in the last column.
        trained_tree: Root of the trained decision tree.

    Returns:
        Classification accuracy between 0.0 and 1.0.
    """
    if test_db.size == 0:
        return 0.0

    features = test_db[:, :-1]
    labels = test_db[:, -1].astype(int)
    predictions = trained_tree.predict_batch(features)
    return float(np.mean(predictions == labels))


def _build_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, label_values: np.ndarray
) -> np.ndarray:
    """Construct a confusion matrix with rows=actual labels, cols=predicted labels."""
    matrix = np.zeros((label_values.size, label_values.size), dtype=int)
    index_by_label = {label: idx for idx, label in enumerate(label_values)}

    for actual, predicted in zip(y_true, y_pred):
        matrix[index_by_label[int(actual)], index_by_label[int(predicted)]] += 1

    return matrix


def _classification_metrics(confusion_matrix: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute per-class precision, recall, and F1 scores from a confusion matrix."""
    true_positives = np.diag(confusion_matrix).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        predicted_positives = confusion_matrix.sum(axis=0)
        actual_positives = confusion_matrix.sum(axis=1)

        precision = np.divide(true_positives, predicted_positives, out=np.zeros_like(true_positives), where=predicted_positives != 0)
        recall = np.divide(true_positives, actual_positives, out=np.zeros_like(true_positives), where=actual_positives != 0)
        f1_denominator = precision + recall
        f1 = np.divide(2 * precision * recall, f1_denominator, out=np.zeros_like(true_positives), where=f1_denominator != 0)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def k_fold_cross_validation(
    dataset: np.ndarray, k: int = 10, random_seed: Optional[int] = 42
) -> Dict[str, np.ndarray]:
    """Run k-fold cross-validation and aggregate evaluation metrics.

    Args:
        dataset: Complete dataset with features + label column.
        k: Number of folds (default 10 as per specification).
        random_seed: Seed for shuffling to guarantee reproducibility.

    Returns:
        Dictionary containing confusion matrix, per-fold accuracies, overall accuracy,
        and per-class precision/recall/F1 arrays.
    """
    if k < 2:
        raise ValueError("k-fold cross-validation requires k >= 2")
    if dataset.shape[0] < k:
        raise ValueError("Dataset must contain at least k samples for k-fold CV")

    rng = np.random.default_rng(random_seed)
    indices = np.arange(dataset.shape[0])
    rng.shuffle(indices)
    shuffled_dataset = dataset[indices]

    folds = np.array_split(shuffled_dataset, k)
    label_values = np.unique(dataset[:, -1]).astype(int)
    confusion_matrix = np.zeros((label_values.size, label_values.size), dtype=int)
    fold_accuracies = np.zeros(k, dtype=float)

    for fold_idx in range(k):
        test_data = folds[fold_idx]
        train_parts = [folds[i] for i in range(k) if i != fold_idx]
        train_data = np.vstack(train_parts)

        tree = decision_tree_learning(train_data, depth=0)

        features = test_data[:, :-1]
        labels = test_data[:, -1].astype(int)
        predictions = tree.predict_batch(features)

        fold_accuracies[fold_idx] = evaluate(test_data, tree)
        confusion_matrix += _build_confusion_matrix(labels, predictions, label_values)

    overall_accuracy = float(np.trace(confusion_matrix) / np.sum(confusion_matrix))
    metrics = _classification_metrics(confusion_matrix)

    return {
        "labels": label_values,
        "confusion_matrix": confusion_matrix,
        "accuracy_per_fold": fold_accuracies,
        "average_accuracy": overall_accuracy,
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
    }