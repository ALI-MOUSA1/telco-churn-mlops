"""
train.py
Trains Logistic Regression, Random Forest, XGBoost and CatBoost on each
dataset version (v1, v2, v3), using K-Fold CV + a validation set for model
selection, and a held-out test set for the single final evaluation.
Everything is logged to MLflow.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

from data_loader import load_data
from preprocessing import run as run_preprocessing
from features import run as run_features
import mlflow_utils

SEED = 42
TARGET_COLUMN = "Churn Value"

# Only v1 needs preprocessing/feature-engineering to be generated first;
# v2 and v3 are produced from the pipeline steps.
DATASET_VERSIONS = ["v1", "v2", "v3"]

MODEL_CONFIGS = {
    "LogisticRegression": {
        "flavor": "sklearn",
        "build": lambda: LogisticRegression(max_iter=1000, random_state=SEED),
        "params": {"max_iter": 1000, "solver": "lbfgs"},
    },
    "RandomForest": {
        "flavor": "sklearn",
        "build": lambda: RandomForestClassifier(
            n_estimators=300, max_depth=8, random_state=SEED
        ),
        "params": {"n_estimators": 300, "max_depth": 8},
    },
    "XGBoost": {
        "flavor": "xgboost",
        "build": lambda: XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            eval_metric="logloss", random_state=SEED
        ),
        "params": {"n_estimators": 300, "max_depth": 5, "learning_rate": 0.05},
    },
    "CatBoost": {
        "flavor": "catboost",
        "build": lambda: CatBoostClassifier(
            iterations=300, depth=6, learning_rate=0.05,
            random_state=SEED, verbose=False
        ),
        "params": {"iterations": 300, "depth": 6, "learning_rate": 0.05},
    },
}


def _prepare_raw_v1_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    v1 is the raw, untouched dataset (kept exactly as delivered, per the
    project spec). To be able to train a model on it at all we only apply
    the minimal step of dropping obvious non-numeric identifier columns and
    numerically encoding Yes/No/target columns -- NO cleaning, NO imputation,
    NO feature engineering happens here. This keeps v1 a fair "raw" baseline.
    """
    df = df.copy()
    drop_cols = [c for c in [
        "CustomerID", "Count", "Country", "State", "City", "Zip Code",
        "Lat Long", "Churn Label", "Churn Score", "Churn Reason"
    ] if c in df.columns]
    df = df.drop(columns=drop_cols)

    df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce")
    df = df.dropna()

    obj_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
    obj_cols = [c for c in obj_cols if c != TARGET_COLUMN]
    df = pd.get_dummies(df, columns=obj_cols, drop_first=True)
    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)
    return df


def get_dataset(version: str) -> pd.DataFrame:
    if version == "v1":
        raw = load_data("v1")
        return _prepare_raw_v1_numeric(raw)
    elif version == "v2":
        return load_data("v2")
    elif version == "v3":
        return load_data("v3")
    else:
        raise ValueError(version)


def split_data(df: pd.DataFrame):
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    # 60% train / 20% val / 20% test, stratified, fixed seed
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=SEED
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=SEED
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def compute_metrics(y_true, y_pred, y_proba):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }


def train_and_evaluate_one(model_name, config, dataset_version, X_train, X_val, X_test, y_train, y_val, y_test):
    print(f"\n--- {model_name} | {dataset_version} ---")

    # 1. Cross-validation on train, for stability check / model selection signal
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    model_cv = config["build"]()
    cv_scores = cross_val_score(model_cv, X_train, y_train, cv=cv, scoring="f1")
    print(f"CV F1 scores: {cv_scores} | mean={cv_scores.mean():.4f}")

    # 2. Fit on train, check validation set (this is where you'd tune
    #    hyperparameters across multiple candidate configs in a fuller search)
    model = config["build"]()
    model.fit(X_train, y_train)
    val_pred = model.predict(X_val)
    val_f1 = f1_score(y_val, val_pred)
    print(f"Validation F1: {val_f1:.4f}")

    # 3. Refit on train+val, single final evaluation on test
    X_trainval = pd.concat([X_train, X_val])
    y_trainval = pd.concat([y_train, y_val])
    final_model = config["build"]()
    final_model.fit(X_trainval, y_trainval)

    test_pred = final_model.predict(X_test)
    test_proba = final_model.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, test_pred, test_proba)
    metrics["cv_f1_mean"] = float(cv_scores.mean())
    metrics["validation_f1"] = float(val_f1)
    print(f"Test metrics: {metrics}")

    run_id = mlflow_utils.log_run(
        model=final_model,
        model_name=model_name,
        dataset_version=dataset_version,
        params=config["params"],
        metrics=metrics,
        seed=SEED,
        y_true=y_test,
        y_pred=test_pred,
        flavor=config["flavor"],
    )
    return run_id, metrics


def train_models(dataset_versions=None):
    """Train every model on every requested dataset version, logging each to MLflow."""
    mlflow_utils.setup_mlflow()

    if dataset_versions is None:
        dataset_versions = DATASET_VERSIONS

    results = []
    for version in dataset_versions:
        df = get_dataset(version)
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

        for model_name, config in MODEL_CONFIGS.items():
            run_id, metrics = train_and_evaluate_one(
                model_name, config, version, X_train, X_val, X_test, y_train, y_val, y_test
            )
            results.append({
                "run_id": run_id, "model": model_name,
                "dataset_version": version, **metrics
            })

    results_df = pd.DataFrame(results)
    print("\n=== Summary ===")
    print(results_df.sort_values("f1_score", ascending=False).to_string(index=False))
    return results_df


if __name__ == "__main__":
    # Make sure v2/v3 exist before training on them
    run_preprocessing()
    run_features()
    train_models()