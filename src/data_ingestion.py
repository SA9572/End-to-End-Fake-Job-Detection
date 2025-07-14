import pandas as pd
import os
from datetime import datetime


# src/data_ingestion.py

import pandas as pd

def load_data(path):
    try:
        df = pd.read_csv(path)
        print(f"[INFO] Loaded data with shape: {df.shape}")
        return df
    except Exception as e:
        print(f"[ERROR] Failed to load data: {e}")
        return None


def load_raw_data(file_path: str) -> pd.DataFrame:
    """Load raw CSV dataset"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"🚫 File not found at: {file_path}")
    df = pd.read_csv(file_path)
    print(f"✅ Raw data loaded successfully with shape: {df.shape}")
    return df

def save_artifact(df: pd.DataFrame, save_path: str):
    """Save DataFrame as CSV artifact"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"📦 Data saved to: {save_path}")

def main():
    # File paths
    raw_file_path = "data/raw/fake_job_postings.csv"
    artifact_save_path = "artifacts/raw_data.csv"

    # Load and Save
    df = load_raw_data(raw_file_path)
    save_artifact(df, artifact_save_path)

if __name__ == "__main__":
    main()
