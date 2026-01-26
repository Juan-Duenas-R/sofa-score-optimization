"""Binning utilities using decision trees."""

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import log_loss
import yaml

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)


RANDOM_SEED = config["random_seed"]

def thresholds_from_tree_1d(Xcol, y, max_leaf_nodes, min_samples_leaf):
    """Extract thresholds from a 1D decision tree.

    Args:
        Xcol (np.ndarray): Single feature column.
        y (np.ndarray): Binary target.
        max_leaf_nodes (int): Maximum number of leaf nodes.
        min_samples_leaf (int): Minimum samples per leaf.

    Returns:
        np.ndarray: Sorted array of split thresholds.
    """
    dt = DecisionTreeClassifier(
        max_leaf_nodes=max_leaf_nodes,
        min_samples_leaf=min_samples_leaf,
        random_state=RANDOM_SEED
    )
    try:
        dt.fit(Xcol.reshape(-1, 1), y)
        tree = dt.tree_
        feat = tree.feature
        thr = tree.threshold

        # Extract valid splits
        splits = thr[feat == 0]
        splits = splits[splits != -2]
        splits = np.unique(np.sort(splits))
        return splits
    except Exception:
        return np.array([])


def auto_select_k_bins(
        X_train,
        y_train,
        X_val,
        y_val,
        feature_names,
        Kmax=50,
        min_samples_leaf=16,
        smoothing=1e-2,
        penalty_coef=0.003
):
    """Automatically select optimal number of bins per feature.

    This function tests different numbers of bins (k) for each feature
    independently and selects the k that minimizes validation log-loss
    with a complexity penalty.

    Args:
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training target.
        X_val (np.ndarray): Validation features.
        y_val (np.ndarray): Validation target.
        feature_names (list): Names of features.
        Kmax (int): Maximum number of bins to consider.
        min_samples_leaf (int): Minimum samples per leaf in decision tree.
        smoothing (float): Smoothing parameter for probability estimation.
        penalty_coef (float): Penalty coefficient for model complexity.

    Returns:
        list: Optimal number of bins for each feature.
    """
    n_features = X_train.shape[1]
    best_k_list = []

    # Global baseline probability (for bins with no data)
    global_pos = y_train.sum()
    global_n = len(y_train)
    global_p_smooth = (global_pos + smoothing) / (global_n + 2 * smoothing)

    for j in range(n_features):
        xtr = X_train[:, j]
        xva = X_val[:, j]
        best_k = 2
        best_score = np.inf

        for k in range(2, min(Kmax, len(np.unique(xtr))) + 1):
            # Get thresholds from tree
            t = thresholds_from_tree_1d(
                xtr, y_train,
                max_leaf_nodes=k,
                min_samples_leaf=min_samples_leaf
            )

            # Ensure we have k-1 thresholds
            if t.size < (k - 1):
                missing = (k - 1) - t.size
                qcand = np.unique(np.quantile(xtr, np.linspace(0.01, 0.99, 50)))
                add = []
                for qv in qcand:
                    if qv not in t:
                        add.append(qv)
                    if len(add) >= missing:
                        break
                if add:
                    t = np.sort(np.concatenate([t, np.array(add[:missing])]))

            if t.size != (k - 1):
                t = np.quantile(xtr, np.linspace(0, 1, k + 1))[1:-1]

            # Bin data
            bin_tr = np.digitize(xtr, t, right=False).astype(int)
            bin_va = np.digitize(xva, t, right=False).astype(int)

            # Compute probabilities per bin
            counts = np.bincount(bin_tr, minlength=k)
            pos = np.bincount(bin_tr, weights=y_train, minlength=k)

            p_bin = (pos + smoothing) / (counts + 2 * smoothing)
            p_bin = np.clip(p_bin, 1e-6, 1 - 1e-6)

            # Predict on validation set
            p_va = np.array([
                p_bin[b] if counts[b] > 0 else global_p_smooth
                for b in bin_va
            ])
            p_va = np.clip(p_va, 1e-12, 1 - 1e-12)

            # Compute score with complexity penalty
            try:
                ll = log_loss(y_val, p_va)
            except Exception:
                ll = 1e3

            score_with_penalty = ll + penalty_coef * k

            if score_with_penalty < best_score:
                best_score = score_with_penalty
                best_k = k

        best_k_list.append(int(best_k))
        print(f"Feature '{feature_names[j]}': selected k={best_k}")

    return best_k_list