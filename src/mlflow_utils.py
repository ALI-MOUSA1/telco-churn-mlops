"""
mlflow_utils.py
Thin wrapper around MLflow so training/evaluation code stays clean.
"""

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.catboost
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

EXPERIMENT_NAME = "telco_churn_prediction"


def setup_mlflow(tracking_uri: str = "sqlite:///mlflow.db"):
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)


def log_run(model, model_name, dataset_version, params, metrics, seed, y_true, y_pred, flavor="sklearn"):
    """
    Logs a single MLflow run: tags, params, metrics, confusion matrix figure, and the model itself.
    """
    with mlflow.start_run(run_name=f"{model_name}_{dataset_version}"):
        mlflow.set_tag("model_name", model_name)
        mlflow.set_tag("dataset_version", dataset_version)

        log_params = dict(params)
        log_params["seed"] = seed
        mlflow.log_params(log_params)

        mlflow.log_metrics(metrics)

        fig = _plot_confusion_matrix(y_true, y_pred, model_name, dataset_version)
        mlflow.log_figure(fig, "confusion_matrix.png")
        plt.close(fig)

        if flavor == "xgboost":
            mlflow.xgboost.log_model(model, "model")
        elif flavor == "catboost":
            mlflow.catboost.log_model(model, "model")
        else:
            mlflow.sklearn.log_model(model, "model")

        return mlflow.active_run().info.run_id


def _plot_confusion_matrix(y_true, y_pred, model_name, dataset_version):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"])
    ax.set_title(f"{model_name} | {dataset_version}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.tight_layout()
    return fig


def register_best_model(run_id: str, model_name: str = "ChurnModel", stage: str = "Production"):
    """
    Registers the model from a given run into the MLflow Model Registry
    and promotes it to the given stage (e.g. "Production").
    """
    model_uri = f"runs:/{run_id}/model"
    result = mlflow.register_model(model_uri, model_name)

    client = mlflow.tracking.MlflowClient()
    client.transition_model_version_stage(
        name=model_name,
        version=result.version,
        stage=stage,
        archive_existing_versions=True,
    )
    print(f"Registered '{model_name}' version {result.version} -> stage '{stage}'")
    return result