"""
data_loader.py
Loads a specific version of the Telco Churn dataset from data/v{version}/.
"""

import os
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

# Expected filename for each dataset version
VERSION_FILES = {
    "v1": "telco_churn_raw.csv",
    "v2": "telco_churn_clean.csv",
    "v3": "telco_churn_features.csv",
}


def load_data(version: str) -> pd.DataFrame:
    """
    Load the dataset for a given version ('v1', 'v2', or 'v3').

    Parameters
    ----------
    version : str
        One of 'v1', 'v2', 'v3'.

    Returns
    -------
    pd.DataFrame
    """
    if version not in VERSION_FILES:
        raise ValueError(f"Unknown version '{version}'. Expected one of {list(VERSION_FILES)}")

    file_path = os.path.join(DATA_DIR, version, VERSION_FILES[version])
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Could not find {file_path}. "
            f"Did you run preprocessing/feature_engineering to generate '{version}' yet?"
        )

    df = pd.read_csv(file_path)
    return df


if __name__ == "__main__":
    df = load_data("v1")
    print(df.shape)
    print(df.head())