"""
preprocessing.py
Cleans the raw Telco Churn dataset (v1) and produces the cleaned version (v2):
- Drops unnecessary / leakage columns
- Handles missing values
- Encodes categorical variables to numeric
"""

import os
import pandas as pd

from data_loader import load_data, DATA_DIR

# Columns that carry no predictive signal (IDs, constants, geo detail)
DROP_COLUMNS = [
    "CustomerID",
    "Count",        # constant = 1 for every row
    "Country",      # constant = "United States"
    "State",        # constant = "California"
    "City",
    "Zip Code",
    "Lat Long",
    "Latitude",
    "Longitude",
]

# Columns that LEAK the target and must never be used as features:
# - Churn Label is just Churn Value spelled as Yes/No
# - Churn Score is a pre-computed churn probability (would make the task trivial)
# - Churn Reason is only populated for customers who already churned
LEAKAGE_COLUMNS = ["Churn Label", "Churn Score", "Churn Reason"]

TARGET_COLUMN = "Churn Value"

# Categorical columns to encode (binary Yes/No style)
BINARY_YES_NO_COLUMNS = [
    "Senior Citizen", "Partner", "Dependents", "Phone Service",
    "Paperless Billing",
]

# Categorical columns with more than 2 categories -> one-hot encode
MULTI_CATEGORY_COLUMNS = [
    "Gender", "Multiple Lines", "Internet Service", "Online Security",
    "Online Backup", "Device Protection", "Tech Support", "Streaming TV",
    "Streaming Movies", "Contract", "Payment Method",
]


def _clean_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    """Total Charges is read as text and has a few blank strings for new customers."""
    df["Total Charges"] = df["Total Charges"].astype(str).str.strip()
    df["Total Charges"] = df["Total Charges"].replace("", pd.NA)
    df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce")
    # Blank Total Charges only happens for customers with 0 tenure -> fill with 0
    df["Total Charges"] = df["Total Charges"].fillna(0.0)
    return df


def _encode_binary_yes_no(df: pd.DataFrame) -> pd.DataFrame:
    for col in BINARY_YES_NO_COLUMNS:
        if col in df.columns:
            df[col] = df[col].map({"Yes": 1, "No": 0}).fillna(df[col])
            # Senior Citizen already comes as Yes/No in this file; handle safely
            df[col] = df[col].replace({"Yes": 1, "No": 0})
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw dataframe: drop unneeded/leakage columns, fix missing values,
    encode categoricals to numeric. Returns a fully numeric dataframe ready
    for feature engineering / modeling.
    """
    df = df.copy()

    # 1. Drop irrelevant + leakage columns
    cols_to_drop = [c for c in DROP_COLUMNS + LEAKAGE_COLUMNS if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    # 2. Handle missing values
    df = _clean_total_charges(df)
    df = df.dropna()  # drop any remaining stray missing rows

    # 3. Encode categorical -> numeric
    df = _encode_binary_yes_no(df)
    multi_cols_present = [c for c in MULTI_CATEGORY_COLUMNS if c in df.columns]
    df = pd.get_dummies(df, columns=multi_cols_present, drop_first=True)

    # Make sure one-hot booleans become 0/1 ints
    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)

    return df


def run(save: bool = True) -> pd.DataFrame:
    raw_df = load_data("v1")
    clean_df = preprocess(raw_df)

    if save:
        out_dir = os.path.join(DATA_DIR, "v2")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "telco_churn_clean.csv")
        clean_df.to_csv(out_path, index=False)
        print(f"Saved cleaned dataset to {out_path} | shape={clean_df.shape}")

    return clean_df


if __name__ == "__main__":
    run()