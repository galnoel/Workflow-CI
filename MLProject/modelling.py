"""
modelling.py — CI Retraining Script (Workflow-CI / Kriteria 3)
===============================================================
Retraining script adapted from Kriteria 2's tuning script for CI reproducibility.
Uses GridSearchCV with manual MLflow logging.

Usage:
    python modelling.py --data_path telco_customer_churn_preprocessing
"""

import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from mlflow.models import infer_signature
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV


SCRIPT_DIR = Path(__file__).resolve().parent


def load_data(data_path: str):
    """Load preprocessed train/test data."""
    resolved_path = Path(data_path)
    if not resolved_path.is_absolute():
        resolved_path = SCRIPT_DIR / resolved_path

    required = ["X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv"]
    missing = [name for name in required if not (resolved_path / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing processed data files in {resolved_path}: {missing}"
        )

    X_train = pd.read_csv(resolved_path / "X_train.csv")
    X_test = pd.read_csv(resolved_path / "X_test.csv")
    y_train = pd.read_csv(resolved_path / "y_train.csv").values.ravel()
    y_test = pd.read_csv(resolved_path / "y_test.csv").values.ravel()
    if list(X_train.columns) != list(X_test.columns):
        raise ValueError("Training and test feature columns do not match")
    if X_train.shape[1] != 30:
        raise ValueError(f"Expected 30 model features, got {X_train.shape[1]}")
    if X_train.isna().any().any() or X_test.isna().any().any():
        raise ValueError("Processed feature data contains missing values")
    if set(y_train) != {0, 1} or set(y_test) != {0, 1}:
        raise ValueError("Target data must contain exactly the labels 0 and 1")
    X_train = X_train.astype(float)
    X_test = X_test.astype(float)
    return X_train, X_test, y_train, y_test


def train_and_log(data_path: str, run_id_output: str):
    """Train with tuning and log everything to MLflow."""
    X_train, X_test, y_train, y_test = load_data(data_path)
    feature_names = list(X_train.columns)
    print(f"Training: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")

    # `mlflow run` supplies MLFLOW_RUN_ID. When the script is called directly,
    # create/use the named local experiment instead.
    if not os.environ.get("MLFLOW_RUN_ID"):
        mlflow.set_experiment("telco-churn-ci")

    # Hyperparameter search
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [10, 20, None],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
    }

    grid = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid,
        cv=5,
        scoring="f1",
        n_jobs=-1,
        verbose=1,
    )
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    # Compute metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "best_cv_f1": grid.best_score_,
    }

    with mlflow.start_run(run_name="ci_retrain"):
        # Log params
        for k, v in grid.best_params_.items():
            mlflow.log_param(k, v)

        # Log metrics
        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        # Log model
        input_example = X_train.head(5)
        signature = infer_signature(input_example, best_model.predict(input_example))
        mlflow.sklearn.log_model(
            best_model,
            "model",
            signature=signature,
            input_example=input_example,
        )

        # Log artifacts
        artifact_dir = SCRIPT_DIR / "artifacts"
        artifact_dir.mkdir(parents=True, exist_ok=True)

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["No Churn", "Churn"],
                    yticklabels=["No Churn", "Churn"], ax=ax)
        ax.set_title("Confusion Matrix")
        confusion_matrix_path = artifact_dir / "confusion_matrix.png"
        fig.savefig(confusion_matrix_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        mlflow.log_artifact(str(confusion_matrix_path))

        # ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, lw=2, label=f"AUC = {metrics['roc_auc']:.4f}")
        ax.plot([0, 1], [0, 1], "--", color="gray")
        ax.set_title("ROC Curve")
        ax.legend()
        roc_curve_path = artifact_dir / "roc_curve.png"
        fig.savefig(roc_curve_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        mlflow.log_artifact(str(roc_curve_path))

        # Classification report
        report = classification_report(y_test, y_pred, target_names=["No Churn", "Churn"])
        classification_report_path = artifact_dir / "classification_report.txt"
        with classification_report_path.open("w", encoding="utf-8") as f:
            f.write(report)
        mlflow.log_artifact(str(classification_report_path))

        run_id = mlflow.active_run().info.run_id
        output_path = Path(run_id_output)
        if not output_path.is_absolute():
            output_path = SCRIPT_DIR / output_path
        output_path.write_text(run_id + "\n", encoding="utf-8")
        print(f"\nRun ID: {run_id}")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

    return run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="telco_customer_churn_preprocessing")
    parser.add_argument("--run_id_output", type=str, default="run_id.txt")
    args = parser.parse_args()
    train_and_log(args.data_path, args.run_id_output)
