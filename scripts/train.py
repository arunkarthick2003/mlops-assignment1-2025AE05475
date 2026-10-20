"""Compare candidate models with cross-validation and log MLflow experiments."""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

from heart_mlops.data import FEATURES, load_or_download
from heart_mlops.model import build_model, validate_features

SEARCH_SPACES = {
    "logistic_regression": {"classifier__C": [0.1, 1.0, 10.0]},
    "random_forest": {
        "classifier__max_depth": [None, 5],
        "classifier__min_samples_leaf": [1, 3],
    },
}
def main() -> None:
    output_dir = Path("artifacts")
    output_dir.mkdir(exist_ok=True)
    frame = load_or_download("data/raw/processed.cleveland.data")
    features = validate_features(frame)
    labels = frame["target"]
    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("uci-heart-disease-cleveland")
    sns.set_theme(style="whitegrid", palette="colorblind")
    results: list[dict[str, object]] = []
    best_score = float("-inf")
    best_pipeline = None
    best_name = ""

    for model_name, search_space in SEARCH_SPACES.items():
        search = GridSearchCV(
            build_model(model_name), search_space, scoring="roc_auc", cv=cv, n_jobs=1, refit=True
        )
        with mlflow.start_run(run_name=model_name) as run:
            search.fit(x_train, y_train)
            predictions = search.predict(x_test)
            probabilities = search.predict_proba(x_test)[:, 1]
            metrics = {
                "cv_roc_auc": float(search.best_score_),
                "test_accuracy": float(accuracy_score(y_test, predictions)),
                "test_precision": float(precision_score(y_test, predictions, zero_division=0)),
                "test_recall": float(recall_score(y_test, predictions, zero_division=0)),
                "test_roc_auc": float(roc_auc_score(y_test, probabilities)),
            }
            mlflow.log_params({"model": model_name, "cv_folds": 5, "seed": 42, **search.best_params_})
            mlflow.log_metrics(metrics)
            mlflow.log_text(classification_report(y_test, predictions, digits=4), "classification_report.txt")
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            sns.heatmap(
                confusion_matrix(y_test, predictions), annot=True, fmt="d", cmap="crest", ax=axes[0]
            )
            axes[0].set(title="Confusion matrix", xlabel="Predicted", ylabel="Actual")
            false_positive, true_positive, _ = roc_curve(y_test, probabilities)
            axes[1].plot(false_positive, true_positive, label=f"AUC = {metrics['test_roc_auc']:.3f}")
            axes[1].plot([0, 1], [0, 1], linestyle="--", color="grey")
            axes[1].set(title="ROC curve", xlabel="False positive rate", ylabel="True positive rate")
            axes[1].legend()
            fig.tight_layout()
            plot_path = output_dir / f"{model_name}_evaluation.png"
            fig.savefig(plot_path, dpi=160)
            plt.close(fig)
            mlflow.log_artifact(str(plot_path))

            results.append({"model": model_name, **metrics, "best_params": search.best_params_, "run_id": run.info.run_id})
            if search.best_score_ > best_score:
                best_score = float(search.best_score_)
                best_pipeline = search.best_estimator_
                best_name = model_name

    if best_pipeline is None:
        raise RuntimeError("No model completed training.")
    model_path = output_dir / "heart_disease_model.joblib"
    import joblib

    joblib.dump(best_pipeline, model_path)
    (output_dir / "training_summary.json").write_text(
        json.dumps({"selected_model": best_name, "features": FEATURES, "runs": results}, indent=2),
        encoding="utf-8",
    )
    with mlflow.start_run(run_name="selected-model-package"):
        mlflow.log_param("selected_model", best_name)
        mlflow.log_artifact(str(model_path))
        mlflow.log_artifact(str(output_dir / "training_summary.json"))
    print(json.dumps({"selected_model": best_name, "runs": results}, indent=2))


if __name__ == "__main__":
    main()