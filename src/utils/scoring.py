"""Scoring utilities for SOFA model."""

import numpy as np


def class_balance_weights(y):
    """Compute class-balanced sample weights.

    Args:
        y (np.ndarray): Binary target variable.

    Returns:
        np.ndarray: Sample weights for balanced training.
    """
    unique, counts = np.unique(y, return_counts=True)
    n = len(y)
    weights = np.zeros_like(y, dtype=float)
    for c, cnt in zip(unique, counts):
        weights[y == c] = n / (len(unique) * cnt)
    return weights


def compute_score_from_thresholds_weights(X, thresholds, weights):
    """Compute aggregated score from binned features.

    Args:
        X (np.ndarray): Feature matrix (n_samples, n_features).
        thresholds (list): List of threshold arrays, one per feature.
        weights (list): List of weight arrays, one per feature.

    Returns:
        np.ndarray: Aggregated scores (n_samples,).
    """
    n, f = X.shape
    score = np.zeros(n, dtype=float)
    for j in range(f):
        t = thresholds[j]
        w = weights[j]
        bin_idx = np.digitize(X[:, j], t, right=False).astype(int)
        bin_idx = np.clip(bin_idx, 0, len(w) - 1)
        score += w[bin_idx]
    return score


def initial_weights_from_bins(X, y, thresholds_list, k_bins_list, eps=1e-6):
    """Initialize weights using logit of empirical probabilities per bin.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Binary target.
        thresholds_list (list): Thresholds per feature.
        k_bins_list (list): Number of bins per feature.
        eps (float): Smoothing parameter.

    Returns:
        list: Initial weights per feature.
    """
    weights_init = []
    n_features = X.shape[1]

    for j in range(n_features):
        t = thresholds_list[j]
        k_j = k_bins_list[j]
        bin_idx = np.digitize(X[:, j], t, right=False).astype(int)

        # Count samples and positives per bin
        counts = np.bincount(bin_idx, minlength=k_j)
        pos = np.bincount(bin_idx, weights=y, minlength=k_j)

        # Compute smoothed probability
        p = (pos + eps) / (counts + 2 * eps)
        p = np.clip(p, 1e-6, 1 - 1e-6)

        # Logit transformation
        logit = np.log(p / (1 - p))
        logit[counts == 0] = 0.0

        weights_init.append(logit)

    return weights_init