"""Evaluation script for trained SOFA models."""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    log_loss,
    confusion_matrix,
    classification_report,
    roc_curve,
)

from src.utils.scoring import compute_score_from_thresholds_weights
from src.train import load_data

FEATURE_NAMES = [
    "PaFi", "GCS", "meanbp", "platelets", "creatinine", "bilirubin",
    "norepinephrine", "epinephrine", "dopamine", "dobutamine"
]

def load_model(model_path):
    """Load trained model pipeline.

    Args:
        model_path (str): Path to pickled model.

    Returns:
        dict: Model pipeline.
    """
    with open(model_path, "rb") as f:
        pipeline = pickle.load(f)
    return pipeline


def predict(pipeline, X):
    """Generate predictions using trained pipeline.

    Args:
        pipeline (dict): Trained model pipeline.
        X (np.ndarray): Feature matrix.

    Returns:
        tuple: (scores, probabilities)
    """
    # Compute raw score
    raw_score = compute_score_from_thresholds_weights(
        X,
        pipeline["thresholds"],
        pipeline["weights"]
    ).reshape(-1, 1)

    # Standardize and apply logistic regression
    score_scaled = pipeline["score_scaler"].transform(raw_score)
    proba = pipeline["final_logistic"].predict_proba(score_scaled)[:, 1]

    return raw_score.ravel(), proba


def evaluate_model(model_path, data_path, output_dir=None):
    """Evaluate model on test data.

    Args:
        model_path (str): Path to trained model.
        test_data_path (str): Path to test data.
        output_dir (str, optional): Directory to save results.

    Returns:
        dict: Evaluation metrics.
    """
    print("=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)

    # Load model and data
    print(f"\nLoading model from {model_path}...")
    pipeline = load_model(model_path)

    _, _, _, _, X_test, y_test = load_data(data_path)

    print(f"Test set size: {len(y_test)}")

    # Generate predictions
    print("\nGenerating predictions...")
    scores, probas = predict(pipeline, X_test)
    preds = (probas >= 0.5).astype(int)

    # Calculate metrics
    auc = roc_auc_score(y_test, probas)
    logloss = log_loss(y_test, np.clip(probas, 1e-12, 1 - 1e-12))
    cm = confusion_matrix(y_test, preds)

    # ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, probas)

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS (probability thresholds = 0.5 )")
    print("=" * 60)
    print(f"\nAUC: {auc:.4f}")
    print(f"Log-loss: {logloss:.6f}")
    print(f"\nConfusion Matrix:")
    print(cm)
    print(f"\nClassification Report:")
    print(classification_report(y_test, preds, target_names=["Alive", "Deceased"]))

    # Print model info
    print("\n" + "=" * 60)
    print("MODEL INFORMATION")
    print("=" * 60)
    print(f"\nNumber of bins per feature: {pipeline['k_bins']}")
    print(f"LR coefficient: {pipeline['meta']['lr_coef']:.4f}")

    print("\nFeature thresholds and weights:")
    for name, t, w in zip(FEATURE_NAMES, pipeline["thresholds"], pipeline["weights"]):
        w_int = np.array(w, dtype=int)
        print(f"\n{name}:")
        print(f"  Thresholds: {np.round(t, 2)}")
        print(f"  Weights: {w_int}")

    # Prepare metrics dict
    metrics = {
        "auc": float(auc),
        "log_loss": float(logloss),
        "confusion_matrix": cm.tolist(),
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "roc_thresholds": thresholds.tolist(),
        "n_test": len(y_test),
        "prevalence": float(y_test.mean()),
    }

    # Save results if output_dir provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save metrics as CSV
        metrics_path = output_dir / "test_metrics.csv"
        df_metrics = pd.DataFrame([metrics])
        df_metrics.to_csv(metrics_path, index=False)

        print(f"\nMetrics saved to {metrics_path}")

        # Save predictions
        results_df = pd.DataFrame({
            "true_label": y_test,
            "predicted_proba": probas,
            "predicted_label": preds,
            "raw_score": scores
        })
        results_path = output_dir / "test_predictions.csv"
        results_df.to_csv(results_path, index=False)
        print(f"Predictions saved to {results_path}")

    return metrics


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Evaluate trained SOFA model"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/sofa_model.pkl",
        help="Path to trained model"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="./data/synthetic_data.csv",
        help="Path to test data"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./results/",
        help="Output directory for results"
    )

    args = parser.parse_args()
    evaluate_model(args.model, args.data, args.output)


if __name__ == "__main__":
    main()