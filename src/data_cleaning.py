import pandas as pd
import os

def load_raw_artifact(file_path: str) -> pd.DataFrame:
    """Load raw data from artifact"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ File not found at: {file_path}")
    df = pd.read_csv(file_path)
    print(f"✅ Raw artifact loaded: {df.shape}")
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Perform data cleaning steps"""
    print("🧹 Starting data cleaning...")

    # Drop duplicates
    df = df.drop_duplicates()
    
    # Drop columns with too many missing values (e.g. 50%+ missing)
    threshold = 0.5
    df = df.loc[:, df.isnull().mean() < threshold]

    # Drop rows where target is missing
    df = df[df['fraudulent'].notna()]

    # Fill NA with 'unknown' for object/text columns
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].fillna('unknown')

    # Fill numeric columns with median
    for col in df.select_dtypes(include=['int64', 'float64']).columns:
        df[col] = df[col].fillna(df[col].median())

    print(f"✅ Cleaned data shape: {df.shape}")
    return df

def save_clean_data(df: pd.DataFrame, save_path: str):
    """Save cleaned dataset"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"📁 Clean data saved to: {save_path}")

def main():
    input_path = "artifacts/raw_data.csv"
    output_path = "artifacts/clean_data.csv"

    df_raw = load_raw_artifact(input_path)
    df_clean = clean_data(df_raw)
    save_clean_data(df_clean, output_path)

if __name__ == "__main__":
    main()
