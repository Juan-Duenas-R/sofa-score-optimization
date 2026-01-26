"""Optimization utilities for integer weight learning."""

import numpy as np
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss

from src.utils.binning import thresholds_from_tree_1d
from src.utils.scoring import (
    compute_score_from_thresholds_weights,
    initial_weights_from_bins,
)
import yaml

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Optimization constants
_INTEGER_LOCAL_SEARCH = config["optimization"]["integer_local_search"]
_INTEGER_LS_MAX_PASSES = config["optimization"]["integer_ls_max_passes"]
_INTEGER_LS_MAX_NO_IMPROVE = config["optimization"]["integer_ls_max_no_improve"]
_INTEGER_LS_STEP = config["optimization"]["integer_ls_step"]
_ENFORCE_POSITIVE_LR_COEF = config["optimization"]["enforce_positive_lr_coef"]

# Scaling constants
TARGET_SCORE_STD = config["scaling"]["target_score_std"]
SCALE_PENALTY_COEF = config["scaling"]["scale_penalty_coef"]
EPS_STD = float(config["scaling"]["eps_std"])


def _clip_round_int(wvec, W_bound):
    """Round weights to integers and clip to bounds.

    Args:
        wvec (np.ndarray): Weight vector.
        W_bound (float): Maximum absolute weight value.

    Returns:
        np.ndarray: Integer weights as floats.
    """
    W = int(W_bound)
    w_int = np.rint(wvec).astype(int)
    w_int = np.clip(w_int, -W, W)
    return w_int.astype(float)


def _discrete_local_search(
        obj_fn,
        w_start,
        W_bound,
        max_passes=3,
        max_no_improve_passes=1,
        step=1
):
    """Discrete local search in integer weight space.

    Performs coordinate descent, testing ±step for each weight component.

    Args:
        obj_fn (callable): Objective function to minimize.
        w_start (np.ndarray): Starting weights.
        W_bound (float): Weight bounds.
        max_passes (int): Maximum number of passes through all coordinates.
        max_no_improve_passes (int): Stop after this many non-improving passes.
        step (int): Step size for coordinate updates.

    Returns:
        np.ndarray: Optimized integer weights.
    """
    W = int(W_bound)
    w_curr = _clip_round_int(w_start, W)
    best_val = float(obj_fn(w_curr))

    no_improve = 0
    for _ in range(max_passes):
        improved = False

        for i in range(len(w_curr)):
            base = w_curr.copy()

            # Test positive step
            cand_plus = base.copy()
            cand_plus[i] = np.clip(cand_plus[i] + step, -W, W)
            cand_plus = _clip_round_int(cand_plus, W)
            v_plus = float(obj_fn(cand_plus))

            # Test negative step
            cand_minus = base.copy()
            cand_minus[i] = np.clip(cand_minus[i] - step, -W, W)
            cand_minus = _clip_round_int(cand_minus, W)
            v_minus = float(obj_fn(cand_minus))

            # Accept best improvement
            if v_plus + 1e-12 < best_val and v_plus <= v_minus:
                w_curr, best_val = cand_plus, v_plus
                improved = True
            elif v_minus + 1e-12 < best_val:
                w_curr, best_val = cand_minus, v_minus
                improved = True

        if not improved:
            no_improve += 1
            if no_improve >= max_no_improve_passes:
                break
        else:
            no_improve = 0

    return w_curr


def run_single_config(
        k_bins_spec,
        min_samples_leaf,
        reg_lambda,
        W_bound,
        X_train,
        y_train,
        X_val,
        y_val,
        w_train,
        min_method,
        bin_penalty_alpha=1.0,
        eps_counts=1e-6
):
    """Run optimization for a single hyperparameter configuration.

    This function:
    1. Creates bins using decision trees
    2. Optimizes continuous weights
    3. Converts to integer weights with scale correction
    4. Refines with discrete local search
    5. Enforces positive logistic regression coefficient

    Args:
        k_bins_spec (int or list): Number of bins per feature.
        min_samples_leaf (int): Minimum samples per leaf in trees.
        reg_lambda (float): L2 regularization for logistic regression.
        W_bound (float): Maximum absolute weight value.
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training target.
        X_val (np.ndarray): Validation features.
        y_val (np.ndarray): Validation target.
        w_train (np.ndarray): Training sample weights.
        min_method (str): Scipy optimization method.
        bin_penalty_alpha (float): Penalty for sparse bins.
        eps_counts (float): Smoothing for bin counts.

    Returns:
        dict: Results containing metrics, thresholds, and weights.
    """
    n_features = X_train.shape[1]

    # Normalize k_bins to list
    if isinstance(k_bins_spec, int):
        k_bins_list = [int(k_bins_spec)] * n_features
    else:
        k_bins_list = list(map(int, list(k_bins_spec)))
        if len(k_bins_list) != n_features:
            raise ValueError(
                "k_bins_spec must be int or list with length = n_features"
            )

    # Create thresholds per feature using decision trees
    thresholds_list = []
    for j in range(n_features):
        k_j = k_bins_list[j]
        t = thresholds_from_tree_1d(
            X_train[:, j],
            y_train,
            max_leaf_nodes=k_j,
            min_samples_leaf=min_samples_leaf
        )

        # Ensure we have k-1 thresholds
        if t.size < (k_j - 1):
            missing = (k_j - 1) - t.size
            qcand = np.unique(
                np.quantile(X_train[:, j], np.linspace(0.01, 0.99, 50))
            )
            add = []
            for qv in qcand:
                if qv not in t:
                    add.append(qv)
                if len(add) >= missing:
                    break
            if add:
                t = np.sort(np.concatenate([t, np.array(add[:missing])]))

        if t.size != (k_j - 1):
            t = np.quantile(X_train[:, j], np.linspace(0, 1, k_j + 1))[1:-1]

        thresholds_list.append(np.sort(t))

    # Initialize weights
    weights_init_list = initial_weights_from_bins(
        X_train, y_train, thresholds_list, k_bins_list
    )
    w0 = np.concatenate(weights_init_list)

    # Helper functions
    def vector_to_weights_list(wvec):
        """Convert flat weight vector to list per feature."""
        out = []
        ptr = 0
        for j in range(n_features):
            k_j = k_bins_list[j]
            out.append(wvec[ptr:ptr + k_j])
            ptr += k_j
        return out

    def compute_score_weights_vector(X, wvec):
        """Compute score from flat weight vector."""
        wlist = vector_to_weights_list(wvec)
        return compute_score_from_thresholds_weights(X, thresholds_list, wlist)

    # Objective function with scale penalty
    def obj_weights(wvec):
        """Objective: log-loss + bin penalty + scale penalty."""
        # Penalty for sparse bins
        penalty_bins = 0.0
        for j in range(n_features):
            k_j = k_bins_list[j]
            counts = np.bincount(
                np.digitize(
                    X_train[:, j],
                    thresholds_list[j],
                    right=False
                ).astype(int),
                minlength=k_j
            )
            penalty_bins += np.sum(1.0 / (counts + eps_counts))
        penalty_bins *= bin_penalty_alpha

        # Compute score
        s_train = compute_score_weights_vector(X_train, wvec).reshape(-1, 1)

        # Scale penalty (breaks scale invariance)
        std_s = float(np.std(s_train))
        penalty_scale = SCALE_PENALTY_COEF * float(
            (std_s - TARGET_SCORE_STD) ** 2
        )

        # Standardize score for logistic regression
        try:
            ss_local = StandardScaler()
            s_train_s = ss_local.fit_transform(s_train)
        except Exception:
            return 1e3 + penalty_bins + penalty_scale

        # Fit logistic regression
        if reg_lambda is None or float(reg_lambda) == 0.0:
            C_reg = 1e12
        else:
            C_reg = max(1e-12, 1.0 / float(reg_lambda))

        try:
            lr = LogisticRegression(
                solver="lbfgs",
                penalty="l2",
                C=C_reg,
                fit_intercept=True,
                max_iter=1000
            )
            lr.fit(s_train_s, y_train, sample_weight=w_train)
            p_train = lr.predict_proba(s_train_s)[:, 1]
            p_train = np.clip(p_train, 1e-12, 1 - 1e-12)
            train_logloss = float(log_loss(y_train, p_train))
        except Exception:
            train_logloss = 1e3

        return float(train_logloss + penalty_bins + penalty_scale)

    # 1. Continuous optimization
    bounds = [(-W_bound, W_bound)] * len(w0)
    res = minimize(
        obj_weights,
        x0=w0,
        method=min_method,
        bounds=bounds,
        options={"maxiter": 1000, "ftol": 1e-6}
    )
    w_opt_cont = res.x.copy()

    # 2. Scale and round to integers
    s_cont = compute_score_weights_vector(X_train, w_opt_cont)
    std_cont = float(np.std(s_cont))
    scale = TARGET_SCORE_STD / max(std_cont, EPS_STD)

    w_scaled = w_opt_cont * scale
    w_opt_int = _clip_round_int(w_scaled, W_bound)

    # 3. Discrete local search
    if _INTEGER_LOCAL_SEARCH:
        w_opt_int = _discrete_local_search(
            obj_fn=obj_weights,
            w_start=w_opt_int,
            W_bound=W_bound,
            max_passes=_INTEGER_LS_MAX_PASSES,
            max_no_improve_passes=_INTEGER_LS_MAX_NO_IMPROVE,
            step=_INTEGER_LS_STEP
        )

    # 4. Evaluate on validation set
    def eval_with_weights(wvec_int):
        """Evaluate weights on validation set."""
        s_train_opt = compute_score_weights_vector(
            X_train, wvec_int
        ).reshape(-1, 1)
        s_val_opt = compute_score_weights_vector(
            X_val, wvec_int
        ).reshape(-1, 1)

        ssf = StandardScaler().fit(s_train_opt)
        s_train_opt_s = ssf.transform(s_train_opt)
        s_val_opt_s = ssf.transform(s_val_opt)

        if reg_lambda is None or float(reg_lambda) == 0.0:
            C_reg = 1e12
        else:
            C_reg = max(1e-12, 1.0 / float(reg_lambda))

        try:
            lr = LogisticRegression(
                solver="lbfgs",
                penalty="l2",
                C=C_reg,
                fit_intercept=True,
                max_iter=1000
            )
            lr.fit(s_train_opt_s, y_train, sample_weight=w_train)
            coef = float(lr.coef_.ravel()[0])

            p_val = lr.predict_proba(s_val_opt_s)[:, 1]
            p_val = np.clip(p_val, 1e-12, 1 - 1e-12)
            ll = float(log_loss(y_val, p_val))
            auc = float(roc_auc_score(y_val, p_val))
            return ll, auc, coef
        except Exception:
            return 1e3, 0.5, 0.0

    val_logloss, val_auc, lr_coef = eval_with_weights(w_opt_int)

    # 5. Enforce positive coefficient
    if _ENFORCE_POSITIVE_LR_COEF and lr_coef < 0:
        w_opt_int = -w_opt_int
        val_logloss, val_auc, lr_coef = eval_with_weights(w_opt_int)

    w_opt_list = vector_to_weights_list(w_opt_int)

    return {
        "val_logloss": float(val_logloss),
        "val_auc": float(val_auc),
        "thresholds": thresholds_list,
        "weights_opt": w_opt_list,
        "k_bins_list": k_bins_list,
        "success": bool(res.success),
        "message": res.message,
        "res_obj": res,
        "lr_coef": float(lr_coef),
        "integer_weights": True,
        "lr_coef_positive_enforced": bool(_ENFORCE_POSITIVE_LR_COEF),
        "target_score_std": float(TARGET_SCORE_STD),
        "scale_penalty_coef": float(SCALE_PENALTY_COEF),
    }