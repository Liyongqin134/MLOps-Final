"""Train RandomForest on digits data and log the experiment with MLflow.

Logs git commit SHA, DVC dataset version hash, hyperparameters, metrics
(accuracy, weighted F1) and the trained model artifact.
"""
import argparse
import json
import os
import subprocess

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

DATA_PATH = "data/digits.csv"
MODEL_PATH = "ml/registry/model.pkl"
DVC_POINTER = "data/digits.csv.dvc"

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")  # MLflow 3.x: allow local file backend


def git_commit_sha():
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()


def dvc_data_hash():
    try:
        with open(DVC_POINTER) as f:
            meta = json.load(f)
        return meta["outs"][0]["md5"]
    except (OSError, KeyError, ValueError):
        return "not-tracked"


def train(n_estimators, max_depth, experiment_name, run_name):
    mlflow.set_tracking_uri(os.path.abspath("mlruns"))
    mlflow.set_experiment(experiment_name)

    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["target"]).values
    y = df["target"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("dataset_version", dvc_data_hash())
        mlflow.log_param("git_commit", git_commit_sha())

        model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=42
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="weighted")
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_weighted", f1)

        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump(model, MODEL_PATH)
        mlflow.log_artifact(MODEL_PATH, artifact_path="model")
        mlflow.sklearn.log_model(model, "model", serialization_format="pickle")

        print(
            f"Logged run '{run_name}': n_estimators={n_estimators}, "
            f"max_depth={max_depth}, accuracy={acc:.4f}, f1={f1:.4f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and log a RandomForest experiment")
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--experiment", default="digits-experiments")
    parser.add_argument("--run-name", default="rf_run")
    args = parser.parse_args()
    train(args.n_estimators, args.max_depth, args.experiment, args.run_name)
