import joblib
import numpy as np

# Load the trained model and transformers
model = joblib.load("models/model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")
encoder = joblib.load("models/employment_encoder.pkl")

def preprocess_input(input_dict):
    """
    Prepares new job posting input for prediction.
    input_dict should contain: title, company_profile, description, requirements, employment_type
    """
    # Combine text
    text = (
        input_dict.get("title", "") + " " +
        input_dict.get("company_profile", "") + " " +
        input_dict.get("description", "") + " " +
        input_dict.get("requirements", "")
    )

    # Vectorize text
    X_text = vectorizer.transform([text]).toarray()

    # Encode employment_type
    emp_type = input_dict.get("employment_type", "unknown")
    emp_encoded = encoder.transform([emp_type]).reshape(-1, 1)

    # Combine features
    X_final = np.concatenate([X_text, emp_encoded], axis=1)

    return X_final

def predict_job_posting(input_dict):
    X = preprocess_input(input_dict)
    prediction = model.predict(X)[0]
    proba = model.predict_proba(X)[0][prediction]
    label = "Fake" if prediction == 1 else "Real"
    print(f"🔍 Prediction: {label} ({proba*100:.2f}%)")
    return label, proba

if __name__ == "__main__":
    # Example input
    sample = {
        "title": "Data Scientist",
        "company_profile": "Leading tech company hiring AI engineers",
        "description": "Looking for a data scientist with experience in ML, Python, and deep learning.",
        "requirements": "Bachelor’s degree, 3+ years of experience",
        "employment_type": "Full-time"
    }

    predict_job_posting(sample)
