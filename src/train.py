import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import mlflow.sklearn


def load_processed_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Processed data file not found: {path}")
    if path.stat().st_size == 0:
        raise RuntimeError(f"Processed data file is empty: {path}")
    return pd.read_csv(path)


def prepare_features(df: pd.DataFrame):
    df = df.copy()
    if "is_high_risk" not in df.columns:
        raise KeyError("Target column 'is_high_risk' not found in processed data")

    # Drop obvious non-feature columns
    drop_cols = [c for c in ["CustomerId", "TransactionId", "TransactionStartTime"] if c in df.columns]
    X = df.drop(columns=drop_cols + ["is_high_risk"])

    # Keep numeric columns only for modeling simplicity
    X = X.select_dtypes(include=[np.number]).fillna(0)
    y = df["is_high_risk"].astype(int)
    return X, y


def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    probs = None
    try:
        probs = model.predict_proba(X_test)[:, 1]
    except Exception:
        pass

    metrics = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "precision": float(precision_score(y_test, preds, zero_division=0)),
        "recall": float(recall_score(y_test, preds, zero_division=0)),
        "f1": float(f1_score(y_test, preds, zero_division=0)),
    }
    if probs is not None:
        metrics["roc_auc"] = float(roc_auc_score(y_test, probs))
    else:
        metrics["roc_auc"] = None
    return metrics


def train_and_log(processed_csv: Path, random_state: int = 42, test_size: float = 0.2):
    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / processed_csv
    df = load_processed_data(data_path)

    X, y = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    mlflow.set_experiment("credit-risk-model")

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)

    models = [
        ("logistic_regression", LogisticRegression(solver="liblinear", random_state=random_state), {
            "C": [0.01, 0.1, 1.0, 10.0]
        }),
        ("random_forest", RandomForestClassifier(random_state=random_state), {
            "n_estimators": [50, 100],
            "max_depth": [5, 10, None]
        })
    ]

    best_overall = {"score": -1, "name": None, "model": None, "metrics": None}

    for name, estimator, params in models:
        with mlflow.start_run(run_name=name):
            gs = GridSearchCV(estimator, params, cv=cv, scoring="roc_auc", n_jobs=-1)
            gs.fit(X_train, y_train)

            best = gs.best_estimator_
            metrics = evaluate_model(best, X_test, y_test)

            # Log params and metrics
            mlflow.log_param("model", name)
            mlflow.log_param("best_params", json.dumps(gs.best_params_))
            for k, v in metrics.items():
                mlflow.log_metric(k, v if v is not None else -1)

            # Log model artifact
            mlflow.sklearn.log_model(best, artifact_path="model")

            if metrics.get("roc_auc") is not None and metrics["roc_auc"] > best_overall["score"]:
                best_overall.update({"score": metrics["roc_auc"], "name": name, "model": best, "metrics": metrics})

    # Save best model to disk for convenience
    if best_overall["model"] is not None:
        out_path = project_root / "models"
        out_path.mkdir(exist_ok=True)
        model_file = out_path / f"best_model_{best_overall['name']}.pkl"
        mlflow.sklearn.save_model(best_overall["model"], str(model_file))

    return best_overall


def main():
    parser = argparse.ArgumentParser(description="Train credit risk models and log to MLflow")
    parser.add_argument(
        "--processed-csv",
        default="data/processed/processed_data.csv",
        help="Relative path to processed CSV from project root"
    )
    args = parser.parse_args()

    result = train_and_log(Path(args.processed_csv))
    print(f"Best model: {result['name']} with ROC-AUC: {result['score']}")


if __name__ == "__main__":
    main()
