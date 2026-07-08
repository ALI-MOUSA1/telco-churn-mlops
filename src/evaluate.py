"""
evaluate.py
Queries MLflow for all logged runs of this experiment, builds a comparison
table across models x dataset versions, and identifies/returns the best run
(by F1-score on the test set, then ROC-AUC as tiebreak).
"""

import pandas as pd
import mlflow

from mlflow_utils import EXPERIMENT_NAME


def get_all_runs() -> pd.DataFrame:
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise RuntimeError(
            f"Experiment '{EXPERIMENT_NAME}' not found. Run train.py first."
        )
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    return runs


def evaluate(metric: str = "metrics.f1_score", tiebreak_metric: str = "metrics.roc_auc") -> pd.Series:
    """
    Compares all logged runs and returns the row (as a pandas Series) of the
    best-performing run.
    """
    runs = get_all_runs()
    if runs.empty:
        raise RuntimeError("No MLflow runs found. Run train.py first.")

    runs_sorted = runs.sort_values([metric, tiebreak_metric], ascending=False)

    cols_to_show = [
        "tags.model_name", "tags.dataset_version",
        "metrics.accuracy", "metrics.precision", "metrics.recall",
        "metrics.f1_score", "metrics.roc_auc", "run_id",
    ]
    cols_to_show = [c for c in cols_to_show if c in runs_sorted.columns]

    print("\n=== All runs, best first ===")
    print(runs_sorted[cols_to_show].to_string(index=False))

    best_run = runs_sorted.iloc[0]
    print(f"\nBest run: {best_run.get('tags.model_name')} on "
          f"{best_run.get('tags.dataset_version')} "
          f"(f1={best_run.get('metrics.f1_score'):.4f}, "
          f"roc_auc={best_run.get('metrics.roc_auc'):.4f})")
    return best_run


if __name__ == "__main__":
    evaluate()