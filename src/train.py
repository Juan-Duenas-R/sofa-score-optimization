"""Main training script for SOFA score optimization."""

import argparse
import pickle
import warnings
from itertools import product
from pathlib import Path
from time import time

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.utils.binning import auto_select_k_bins
from src.utils.optimization import (
    run_single_config,
    BIN_PENALTY_MODE,
    MONOTONICITY_ENABLED,
)
from src.utils.scoring import (
    class_balance_weights,
    compute_score_from_thresholds_weights,
)

warnings.filterwarnings("ignore")

#Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)


RANDOM_SEED = config["random_seed"]
MIN_METHOD = config["min_method"]
TEST_SIZE = config["test_size"]
VAL_SIZE = config["val_size"]

MIN_SAMPLES_LEAF_LIST = config["grid_search"]["min_samples_leaf_list"]
W_BOUND_LIST = config["grid_search"]["w_bound_list"]
REG_LAMBDA_LIST = config["grid_search"]["reg_lambda_list"]
BIN_PENALTY_ALPHA_LIST = config["grid_search"].get(
    "bin_penalty_alpha_list", None
)

AUTO_BIN_KMAX = config["auto_bin"]["kmax"]
AUTO_BIN_MIN_SAMPLES_LEAF = config["auto_bin"]["min_samples_leaf"]
AUTO_BIN_SMOOTHING = config["auto_bin"]["smoothing"]
AUTO_BIN_PENALTY_COEF = config["auto_bin"]["penalty_coef"]

BIN_PENALTY_ALPHA = config["single_config"]["bin_penalty_alpha"]
EPS_COUNTS = float(config["single_config"]["eps_counts"])

FEATURE_NAMES = config["feature_names"]
TARGET = config["target"]
STRATIFY_COL = config["stratify_col"]


def load_data(data_path):
    """Load and split data into train/val/test sets.

    Args:
        data_path (str): Path to parquet or csv file

    Returns:
        tuple: (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    print(f"Loading data from {data_path}...")
    if data_path.endswith(".parquet"):
        df = pd.read_parquet(data_path)
    elif data_path.endswith(".csv"):
        df = pd.read_csv(data_path)
    else:
        raise Exception("Data must be in .parquet or .csv")

    if(STRATIFY_COL is None):
        # Split: 60% train, 20% val, 20% test
        train_val_df, test_df = train_test_split(
            df,
            test_size=TEST_SIZE,
            shuffle=True,
            random_state=RANDOM_SEED
        )

        train_df, val_df = train_test_split(
            train_val_df,
            test_size=VAL_SIZE,
            shuffle=True,
            random_state=RANDOM_SEED
        )
    else:
        # Stratified split: 60% train, 20% val, 20% test
        train_val_df, test_df = train_test_split(
            df,
            test_size=TEST_SIZE,
            stratify=df[STRATIFY_COL],
            shuffle=True,
            random_state=RANDOM_SEED
        )

        train_df, val_df = train_test_split(
            train_val_df,
            test_size=VAL_SIZE,
            stratify=train_val_df[STRATIFY_COL],
            shuffle=True,
            random_state=RANDOM_SEED
        )

    # Extract features and targets
    try:
        X_train = train_df[FEATURE_NAMES].values.astype(float)
        y_train = train_df[TARGET].astype(int).values

        X_val = val_df[FEATURE_NAMES].values.astype(float)
        y_val = val_df[TARGET].astype(int).values

        X_test = test_df[FEATURE_NAMES].values.astype(float)
        y_test = test_df[TARGET].astype(int).values
    except:
        raise Exception("Data must be available to convert to float")

    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    return X_train, y_train, X_val, y_val, X_test, y_test


def perform_grid_search(
        best_k_bins_list,
        X_train,
        y_train,
        X_val,
        y_val,
        w_train
):
    """Perform grid search over hyperparameters.

    Args:
        best_k_bins_list (list): Optimal bins per feature.
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training target.
        X_val (np.ndarray): Validation features.
        y_val (np.ndarray): Validation target.
        w_train (np.ndarray): Training sample weights.

    Returns:
        pd.DataFrame: Results sorted by validation log-loss.
    """
    # alpha is only a live hyperparameter under the corrected penalty; under
    # "legacy" the term has zero gradient, so the grid collapses to one value.
    if BIN_PENALTY_ALPHA_LIST and BIN_PENALTY_MODE == "shrinkage":
        alpha_list = list(BIN_PENALTY_ALPHA_LIST)
    else:
        alpha_list = [BIN_PENALTY_ALPHA]

    grid = list(product(
        MIN_SAMPLES_LEAF_LIST, REG_LAMBDA_LIST, W_BOUND_LIST, alpha_list
    ))
    print(f"\nGrid search: {len(grid)} configurations "
          f"(bin_penalty_mode={BIN_PENALTY_MODE}, "
          f"alpha grid={alpha_list})")

    results = []
    t0 = time()

    for idx, (min_samples_leaf, reg_lambda, W_bound, bin_alpha) in \
            enumerate(grid, 1):
        print(f"\n[{idx}/{len(grid)}] Config: k_bins={best_k_bins_list}, "
              f"min_samples_leaf={min_samples_leaf}, "
              f"reg_lambda={reg_lambda}, W_bound={W_bound}, "
              f"bin_penalty_alpha={bin_alpha}")

        try:
            out = run_single_config(
                best_k_bins_list,
                min_samples_leaf,
                reg_lambda,
                W_bound,
                X_train,
                y_train,
                X_val,
                y_val,
                w_train,
                MIN_METHOD,
                bin_penalty_alpha=bin_alpha,
                eps_counts=EPS_COUNTS
            )

            results.append({
                "k_bins": out["k_bins_list"],
                "min_samples_leaf": min_samples_leaf,
                "reg_lambda": reg_lambda,
                "W_bound": W_bound,
                "bin_penalty_alpha": bin_alpha,
                "val_logloss": out["val_logloss"],
                "val_auc": out["val_auc"],
                "success": out["success"],
                "message": out["message"],
                "thresholds": out["thresholds"],
                "weights_opt": out["weights_opt"]
            })

            print(f"  -> val_logloss = {out['val_logloss']:.6f}, "
                  f"val_auc = {out['val_auc']:.4f} "
                  f"(success={out['success']})")

        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                "k_bins": best_k_bins_list,
                "min_samples_leaf": min_samples_leaf,
                "reg_lambda": reg_lambda,
                "W_bound": W_bound,
                "bin_penalty_alpha": bin_alpha,
                "val_logloss": 1e3,
                "val_auc": 0.5,
                "success": False,
                "message": str(e),
                "thresholds": None,
                "weights_opt": None
            })

    results_df = pd.DataFrame(results).sort_values(
        "val_logloss", ascending=True
    )

    print(f"\nGrid search completed in {time() - t0:.1f}s")
    print("\nTop 5 configurations:")
    print(results_df.head(5))

    return results_df


def refit_final_model(
        best_config,
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test
):
    """Refit final model on train+val and evaluate on test.

    Args:
        best_config (dict): Best hyperparameter configuration.
        X_train, y_train, X_val, y_val, X_test, y_test: Data arrays.

    Returns:
        dict: Final model pipeline and metrics.
    """
    print("\n" + "=" * 60)
    print("Refitting final model on TRAIN+VAL...")
    print("=" * 60)

    # Combine train and val
    X_comb = np.vstack([X_train, X_val])
    y_comb = np.hstack([y_train, y_val])
    w_comb = class_balance_weights(y_comb)

    # Run optimization on combined data
    from src.utils.binning import thresholds_from_tree_1d
    from src.utils.scoring import initial_weights_from_bins
    from src.utils.optimization import (
        _clip_round_int,
        _discrete_local_search,
        TARGET_SCORE_STD,
        SCALE_PENALTY_COEF,
        EPS_STD,
        MONOTONICITY_ENABLED,
        RECENTER_WEIGHTS,
        BIN_PENALTY_MODE,
        make_param_spec,
    )
    from src.utils.constraints import (
        theta_to_weights,
        weights_to_theta,
        theta_bounds,
        round_weights_monotone,
        recenter_weights,
        bin_counts_per_feature,
        sparse_bin_penalty,
        monotonicity_report,
    )
    from scipy.optimize import minimize

    k_bins_list = best_config['k_bins']
    min_samples_leaf = best_config['min_samples_leaf']
    reg_lambda = best_config['reg_lambda']
    W_bound = best_config['W_bound']
    bin_alpha = best_config.get('bin_penalty_alpha', BIN_PENALTY_ALPHA)
    n_features = X_comb.shape[1]

    # Get thresholds on combined data
    thresholds_list_comb = []
    for j in range(n_features):
        k_j = k_bins_list[j]
        t = thresholds_from_tree_1d(
            X_comb[:, j],
            y_comb,
            max_leaf_nodes=k_j,
            min_samples_leaf=min_samples_leaf
        )

        # Ensure k-1 thresholds
        if t.size < (k_j - 1):
            missing = (k_j - 1) - t.size
            qcand = np.unique(
                np.quantile(X_comb[:, j], np.linspace(0.01, 0.99, 50))
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
            t = np.quantile(X_comb[:, j], np.linspace(0, 1, k_j + 1))[1:-1]

        thresholds_list_comb.append(np.sort(t))

    # Initialize weights
    weights_init_list_comb = initial_weights_from_bins(
        X_comb, y_comb, thresholds_list_comb, k_bins_list
    )
    w0_comb = np.concatenate(weights_init_list_comb)

    # Helper functions
    def vector_to_weights_list(wvec):
        out = []
        ptr = 0
        for j in range(n_features):
            k_j = k_bins_list[j]
            out.append(wvec[ptr:ptr + k_j])
            ptr += k_j
        return out

    def compute_score_from_wvec(X, wvec):
        wlist = vector_to_weights_list(wvec)
        return compute_score_from_thresholds_weights(
            X, thresholds_list_comb, wlist
        )

    # Bin counts depend only on the data and the thresholds, never on the
    # weights, so they are computed once instead of on every objective call.
    counts_flat_comb = bin_counts_per_feature(
        X_comb, thresholds_list_comb, k_bins_list
    )
    spec_layout_comb = make_param_spec(k_bins_list)

    # Objective function
    def obj_weights_on_comb(wvec, bin_penalty_alpha=bin_alpha,
                            eps_counts=EPS_COUNTS):
        penalty_bins = sparse_bin_penalty(
            wvec, counts_flat_comb, bin_penalty_alpha, eps_counts,
            mode=BIN_PENALTY_MODE, spec=spec_layout_comb
        )

        s_comb = compute_score_from_wvec(X_comb, wvec).reshape(-1, 1)
        std_s = float(np.std(s_comb))
        penalty_scale = SCALE_PENALTY_COEF * float(
            (std_s - TARGET_SCORE_STD) ** 2
        )

        try:
            ss_local = StandardScaler()
            s_comb_s = ss_local.fit_transform(s_comb)
        except Exception:
            return 1e3 + penalty_bins + penalty_scale

        C_reg = 1e12 if reg_lambda == 0 else max(1e-12, 1.0 / reg_lambda)

        try:
            lr_local = LogisticRegression(
                solver="lbfgs",
                penalty="l2",
                C=C_reg,
                fit_intercept=True,
                max_iter=1000
            )
            lr_local.fit(s_comb_s, y_comb, sample_weight=w_comb)
            p = lr_local.predict_proba(s_comb_s)[:, 1]
            p = np.clip(p, 1e-12, 1 - 1e-12)
            loss = float(log_loss(y_comb, p))
        except Exception:
            loss = 1e3

        return float(loss + penalty_bins + penalty_scale)

    # Optimize continuous weights
    print("\nOptimizing continuous weights...")
    if MONOTONICITY_ENABLED:
        spec_comb = spec_layout_comb
        print(spec_comb.summary())
        theta0_comb = weights_to_theta(w0_comb, spec_comb, W_bound=W_bound)

        def obj_theta_on_comb(theta):
            return obj_weights_on_comb(theta_to_weights(theta, spec_comb))

        res_cont = minimize(
            obj_theta_on_comb,
            x0=theta0_comb,
            method=MIN_METHOD,
            bounds=theta_bounds(spec_comb, W_bound),
            options={"maxiter": 1000, "ftol": 1e-6}
        )
        w_opt_cont = theta_to_weights(res_cont.x, spec_comb)
    else:
        spec_comb = None
        bounds_comb = [(-W_bound, W_bound)] * len(w0_comb)
        res_cont = minimize(
            obj_weights_on_comb,
            x0=w0_comb,
            method=MIN_METHOD,
            bounds=bounds_comb,
            args=(bin_alpha, EPS_COUNTS),
            options={"maxiter": 1000, "ftol": 1e-6}
        )
        w_opt_cont = res_cont.x.copy()

    # Convert to integer weights with scale correction
    print("Converting to integer weights...")
    s_cont = compute_score_from_wvec(X_comb, w_opt_cont)
    std_cont = float(np.std(s_cont))
    scale = TARGET_SCORE_STD / max(std_cont, EPS_STD)
    w_scaled = w_opt_cont * scale
    if MONOTONICITY_ENABLED:
        w_int0 = round_weights_monotone(w_scaled, W_bound)
    else:
        w_int0 = _clip_round_int(w_scaled, W_bound)

    # Discrete local search
    print("Refining with discrete local search...")
    w_opt_int = _discrete_local_search(
        obj_fn=obj_weights_on_comb,
        w_start=w_int0,
        W_bound=int(W_bound),
        max_passes=15,
        max_no_improve_passes=10,
        step=1,
        spec=spec_comb
    )

    w_opt_list_comb = vector_to_weights_list(w_opt_int)

    # Calibrate and evaluate on test
    def fit_calibrator_and_eval(weights_list):
        score_comb = compute_score_from_thresholds_weights(
            X_comb, thresholds_list_comb, weights_list
        ).reshape(-1, 1)
        score_test = compute_score_from_thresholds_weights(
            X_test, thresholds_list_comb, weights_list
        ).reshape(-1, 1)

        ss = StandardScaler()
        score_comb_s = ss.fit_transform(score_comb)
        score_test_s = ss.transform(score_test)

        C_reg_final = 1e12 if reg_lambda == 0 else max(1e-12, 1.0 / reg_lambda)

        lr = LogisticRegression(
            solver="lbfgs",
            penalty="l2",
            C=C_reg_final,
            max_iter=1000
        )
        lr.fit(score_comb_s, y_comb)

        y_prob = lr.predict_proba(score_test_s)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
        ll = log_loss(y_test, np.clip(y_prob, 1e-12, 1 - 1e-12))
        coef = float(lr.coef_.ravel()[0])

        return lr, ss, y_prob, auc, ll, coef

    final_logreg, ss_final, y_test_prob, test_auc, test_logloss, coef = \
        fit_calibrator_and_eval(w_opt_list_comb)

    # Enforce positive coefficient (behaviour unchanged)
    if coef < 0:
        print("\nCoefficient negative, inverting weights...")
        if MONOTONICITY_ENABLED:
            print("  WARNING: sign flip under monotonicity constraints "
                  "reverses the imposed directions.")
        w_opt_list_comb = [(-w) for w in w_opt_list_comb]
        final_logreg, ss_final, y_test_prob, test_auc, test_logloss, coef = \
            fit_calibrator_and_eval(w_opt_list_comb)

    # Presentational recentring + feasibility report
    if MONOTONICITY_ENABLED and spec_comb is not None:
        w_flat_final = np.concatenate(w_opt_list_comb)
        if RECENTER_WEIGHTS:
            w_flat_final = recenter_weights(w_flat_final, spec_comb)
            w_opt_list_comb = vector_to_weights_list(w_flat_final)
            # Recentring is a pure level shift; refit the calibrator so the
            # stored intercept matches the published weights.
            final_logreg, ss_final, y_test_prob, test_auc, test_logloss, \
                coef = fit_calibrator_and_eval(w_opt_list_comb)
        report, mono_ok = monotonicity_report(w_flat_final, spec_comb)
        print("\n" + report)
        if not mono_ok:
            raise RuntimeError(
                "Monotonicity constraints violated in the final model."
            )

    print(f"\nFinal test metrics:")
    print(f"  AUC: {test_auc:.4f}")
    print(f"  Log-loss: {test_logloss:.6f}")
    print(f"  LR coefficient: {coef:.6f}")

    return {
        "thresholds": thresholds_list_comb,
        "weights": w_opt_list_comb,
        "score_scaler": ss_final,
        "final_logistic": final_logreg,
        "test_auc": test_auc,
        "test_logloss": test_logloss,
        "lr_coef": coef
    }


def _eval_split(score, y, scaler, lr, label):
    """Metrics for one split under a fixed score, scaler and calibrator."""
    z = scaler.transform(np.asarray(score, float).reshape(-1, 1))
    p = lr.predict_proba(z)[:, 1]
    return {
        "split": label,
        "n": int(len(y)),
        "event_rate": float(np.mean(y)),
        "auroc": float(roc_auc_score(y, p)),
        "auprc": float(average_precision_score(y, p)),
        "logloss": float(log_loss(y, np.clip(p, 1e-12, 1 - 1e-12),
                                  labels=[0, 1])),
    }

def train_sofa_model(data_path, output_dir="models/"):
    """Main training pipeline.

    Args:
        data_path (str): Path to input data.
        output_dir (str): Directory to save outputs.

    Returns:
        dict: Training results and metrics.
    """
    np.random.seed(RANDOM_SEED)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(data_path)
    w_train = class_balance_weights(y_train)

    # Auto-select bins
    print("\n" + "=" * 60)
    print("Automatic bin selection...")
    print("=" * 60)
    auto_t0 = time()
    best_k_bins_list = auto_select_k_bins(
        X_train, y_train, X_val, y_val, FEATURE_NAMES,
        Kmax=AUTO_BIN_KMAX,
        min_samples_leaf=AUTO_BIN_MIN_SAMPLES_LEAF,
        smoothing=AUTO_BIN_SMOOTHING,
        penalty_coef=AUTO_BIN_PENALTY_COEF
    )
    print(f"Bin selection completed in {time() - auto_t0:.1f}s")
    print(f"Selected bins per feature: {best_k_bins_list}")

    # Grid search
    print("\n" + "=" * 60)
    print("Grid search...")
    print("=" * 60)
    results_df = perform_grid_search(
        best_k_bins_list, X_train, y_train, X_val, y_val, w_train
    )

    # Save grid results
    results_path = output_dir / "grid_search_results.csv"
    results_df["k_bins_str"] = results_df["k_bins"].apply(
        lambda x: ",".join(map(str, x)) if isinstance(x, (list, tuple)) else str(x)
    )
    results_df.to_csv(results_path, index=False)
    print(f"\nGrid results saved to {results_path}")

    # Refit final model
    best_row = results_df.iloc[0]
    best_config = {
        'k_bins': list(map(int, best_row["k_bins"])),
        'min_samples_leaf': int(best_row["min_samples_leaf"]),
        'reg_lambda': float(best_row["reg_lambda"]),
        'W_bound': float(best_row["W_bound"]),
        'bin_penalty_alpha': float(
            best_row.get("bin_penalty_alpha", BIN_PENALTY_ALPHA)
        ),
    }

    final_model = refit_final_model(
        best_config, X_train, y_train, X_val, y_val, X_test, y_test
    )

    # Build final pipeline
    pipeline_final = {
        "feature_names": FEATURE_NAMES,
        "k_bins": best_config['k_bins'],
        "thresholds": final_model["thresholds"],
        "weights": final_model["weights"],
        "score_scaler": final_model["score_scaler"],
        "final_logistic": final_model["final_logistic"],
        "grid_results": results_df,
        "meta": {
            "best_min_samples_leaf": best_config['min_samples_leaf'],
            "best_reg_lambda": best_config['reg_lambda'],
            "best_W_bound": best_config['W_bound'],
            "best_bin_penalty_alpha": best_config['bin_penalty_alpha'],
            "bin_penalty_mode": BIN_PENALTY_MODE,
            "monotonicity_enabled": MONOTONICITY_ENABLED,
            "test_auc": final_model["test_auc"],
            "test_logloss": final_model["test_logloss"],
            "lr_coef": final_model["lr_coef"],
            "integer_weights": True,
            "lr_coef_positive": True,
        }
    }

    # Save pipeline
    model_path = output_dir / "sofa_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(pipeline_final, f)
    print(f"\nFinal model saved to {model_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"Best configuration: {best_config}")
    print(f"Test AUC: {final_model['test_auc']:.4f}")
    print(f"Test Log-loss: {final_model['test_logloss']:.6f}")
    print("\nThresholds and weights per feature:")
    for name, t, w in zip(FEATURE_NAMES, final_model["thresholds"], final_model["weights"]):
        w_int = np.array(w, dtype=int)
        print(f"  {name:15s}: thresholds={np.round(t, 2)}, weights={w_int}")

    return pipeline_final


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Train SOFA score model with integer weights"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="./data/synthetic_data.csv",
        help="Path to input data (parquet file)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./models/",
        help="Output directory for models"
    )

    args = parser.parse_args()
    train_sofa_model(args.data, args.output)


if __name__ == "__main__":
    main()