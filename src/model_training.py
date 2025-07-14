import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

def load_data(X_path: str, y_path: str):
    if not os.path.exists(X_path) or not os.path.exists(y_path):
        raise FileNotFoundError("❌ Feature or label file not found.")
    X = np.load(X_path)
    y = np.load(y_path)
    print(f"✅ Data loaded: X={X.shape}, y={y.shape}")
    return X, y

def train_model(X_train, y_train):
    print("🎯 Training Logistic Regression model...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    print("✅ Model training complete.")
    return model

def evaluate_model(model, X_test, y_test):
    print("📊 Evaluating model...")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Accuracy: {acc:.4f}")
    print("\n🔍 Classification Report:\n")
    print(classification_report(y_test, y_pred))

def save_model(model, path: str):
    joblib.dump(model, path)
    print(f"💾 Model saved to: {path}")

def main():
    X_path = "artifacts/X.npy"
    y_path = "artifacts/y.npy"
    model_path = "models/model.pkl"

    X, y = load_data(X_path, y_path)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    save_model(model, model_path)

if __name__ == "__main__":
    main()
