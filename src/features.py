"""
features.py
Feature engineering step: builds new features from the cleaned dataset (v2)
and normalizes numeric columns to produce the final feature set (v3).
"""

import os
import json
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

from data_loader import load_data, DATA_DIR

TARGET_COLUMN = "Churn Value"

# Base numeric columns that exist right after preprocessing
NUMERIC_BASE_COLS = ["Tenure Months", "Monthly Charges", "Total Charges", "CLTV"]


def _add_new_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Average monthly spend so far (guards against tenure = 0)
    df["Avg_Monthly_Spend"] = df["Total Charges"] / df["Tenure Months"].replace(0, 1)

    # Tenure buckets: new / established / loyal customers
    df["Tenure_Bucket"] = pd.cut(
        df["Tenure Months"],
        bins=[-1, 12, 24, 48, 1000],
        labels=[0, 1, 2, 3],
    ).astype(int)

    # Count of subscribed "add-on" services (proxy for engagement/stickiness)
    service_cols = [c for c in df.columns if c.startswith((
        "Online Security_", "Online Backup_", "Device Protection_",
        "Tech Support_", "Streaming TV_", "Streaming Movies_"
    )) and c.endswith("Yes")]
    if service_cols:
        df["Num_Services_Subscribed"] = df[service_cols].sum(axis=1)
    else:
        df["Num_Services_Subscribed"] = 0

    # Monthly charges relative to tenure (spend intensity)
    df["Charges_Per_Tenure"] = df["Monthly Charges"] / (df["Tenure Months"] + 1)

    return df


def feature_engineering(df: pd.DataFrame, scaler: StandardScaler = None, fit_scaler: bool = True):
    """
    Add new engineered features and normalize numeric columns.

    Returns (df_transformed, fitted_scaler)
    """
    df = df.copy()
    df = _add_new_features(df)

    numeric_cols = NUMERIC_BASE_COLS + [
        "Avg_Monthly_Spend", "Charges_Per_Tenure"
    ]
    numeric_cols = [c for c in numeric_cols if c in df.columns]

    if fit_scaler or scaler is None:
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    else:
        df[numeric_cols] = scaler.transform(df[numeric_cols])

    return df, scaler


def run(save: bool = True):
    clean_df = load_data("v2")
    feat_df, scaler = feature_engineering(clean_df, fit_scaler=True)

    if save:
        out_dir = os.path.join(DATA_DIR, "v3")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "telco_churn_features.csv")
        feat_df.to_csv(out_path, index=False)

        scaler_path = os.path.join(out_dir, "scaler.joblib")
        joblib.dump(scaler, scaler_path)

        print(f"Saved feature-engineered dataset to {out_path} | shape={feat_df.shape}")
        print(f"Saved fitted scaler to {scaler_path}")

    return feat_df, scaler


if __name__ == "__main__":
    run()