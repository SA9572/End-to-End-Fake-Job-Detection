import numpy as np
import os
import joblib
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_auc_score,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def load_model(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError("❌ Model not found.")
    model = joblib.load(path)
    print(f"✅ Model loaded from: {path}")
    return model

def load_data(X_path: str, y_path: str):
    if not os.path.exists(X_path) or not os.path.exists(y_path):
        raise FileNotFoundError("❌ Feature or label file missing.")
    X = np.load(X_path)
    y = np.load(y_path)
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def evaluate(model, X_test, y_test, save_report_path: str = None):
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"✅ Accuracy: {acc:.4f}")
    print("🔍 Classification Report:\n", report)

    # Save report
    if save_report_path:
        with open(save_report_path, "w") as f:
            f.write(f"Accuracy: {acc:.4f}\n\n")
            f.write("Classification Report:\n")
            f.write(report)
        print(f"📁 Evaluation report saved to: {save_report_path}")

    # Plot confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.savefig("artifacts/confusion_matrix.png")
    print("📊 Confusion matrix saved to: artifacts/confusion_matrix.png")
    plt.show()

def main():
    model_path = "models/model.pkl"
    X_path = "artifacts/X.npy"
    y_path = "artifacts/y.npy"
    report_path = "artifacts/evaluation_report.txt"

    model = load_model(model_path)
    _, X_test, _, y_test = load_data(X_path, y_path)

    evaluate(model, X_test, y_test, save_report_path=report_path)

if __name__ == "__main__":
    main()
