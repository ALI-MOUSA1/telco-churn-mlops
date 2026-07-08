"""
run_pipeline.py
Single entry point that runs the full MLOps pipeline end to end:
load raw data -> preprocess -> feature engineering -> train models
-> evaluate/compare -> register best model in MLflow.

Usage:
    python run_pipeline.py                # run everything
    python run_pipeline.py --register      # also register the best model
                                            # to the MLflow Model Registry
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from preprocessing import run as preprocess          # noqa: E402
from features import run as feature_engineering      # noqa: E402
from train import train_models                       # noqa: E402
from evaluate import evaluate                         # noqa: E402
from mlflow_utils import setup_mlflow, register_best_model  # noqa: E402


def load_data():
    """v1 (raw) is already committed under data/v1/ -- nothing to fetch here,
    this step exists to keep the pipeline stages explicit as required."""
    print("Step 1/6: load_data -> using data/v1/telco_churn_raw.csv")


def log_to_mlflow(best_run, do_register: bool):
    """Final bookkeeping step: optionally promote the best run to Production."""
    if do_register:
        register_best_model(best_run["run_id"], model_name="ChurnModel", stage="Production")
    else:
        print("Skipping model registration (pass --register to enable).")


def main(do_register: bool = False):
    load_data()

    print("Step 2/6: preprocess -> building data/v2/")
    preprocess()

    print("Step 3/6: feature_engineering -> building data/v3/")
    feature_engineering()

    print("Step 4/6: train_models -> training on v1, v2, v3")
    setup_mlflow()
    train_models()

    print("Step 5/6: evaluate -> comparing all MLflow runs")
    best_run = evaluate()

    print("Step 6/6: log_to_mlflow -> finalize")
    log_to_mlflow(best_run, do_register)

    print("\nPipeline complete. Run `mlflow ui` and open http://localhost:5000 to inspect results.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--register", action="store_true",
                         help="Register the best model to the MLflow Model Registry as Production")
    args = parser.parse_args()
    main(do_register=args.register)