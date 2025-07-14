import pandas as pd
import numpy as np
import os
import joblib

# Save vectorizer and encoder for prediction
# (Moved to after fitting in preprocess function)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

def load_clean_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ Cleaned data not found at: {path}")
    df = pd.read_csv(path)
    print(f"✅ Loaded clean data: {df.shape}")
    return df

def preprocess(df: pd.DataFrame):
    print("🔄 Starting data transformation...")

    # Target
    y = df["fraudulent"].values

    # Combine important text columns (you can choose more)
    df["text"] = df["title"].astype(str) + " " + df["company_profile"].astype(str) + " " + df["description"].astype(str) + " " + df["requirements"].astype(str)

    # Vectorize the text column
    # Vectorize the text column
    vectorizer = TfidfVectorizer(max_features=500)
    X_text = vectorizer.fit_transform(df["text"]).toarray()
    print(f"📝 Text vectorized shape: {X_text.shape}")

    # Save the fitted vectorizer
    os.makedirs("models", exist_ok=True)
    joblib.dump(vectorizer, "models/vectorizer.pkl")

    # Encode categorical column: employment_type (optional)
    if "employment_type" in df.columns:
        df["employment_type"] = df["employment_type"].astype(str)
        le = LabelEncoder()
        employment_encoded = le.fit_transform(df["employment_type"]).reshape(-1, 1)
        print(f"📊 Encoded employment_type")
        # Save the fitted encoder
        joblib.dump(le, "models/employment_encoder.pkl")
        # Combine with text features
        X = np.concatenate([X_text, employment_encoded], axis=1)
    else:
        X = X_text
    print(f"✅ Final feature matrix shape: {X.shape}")
    return X, y

def save_numpy_arrays(X: np.ndarray, y: np.ndarray, X_path: str, y_path: str):
    np.save(X_path, X)
    np.save(y_path, y)
    print(f"💾 Saved features to {X_path} and labels to {y_path}")

def main():
    input_path = "artifacts/clean_data.csv"
    X_path = "artifacts/X.npy"
    y_path = "artifacts/y.npy"

    df = load_clean_data(input_path)
    X, y = preprocess(df)
    save_numpy_arrays(X, y, X_path, y_path)

if __name__ == "__main__":
    main()
