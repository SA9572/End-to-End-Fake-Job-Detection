import os
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Locate dataset from project root
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(base_dir, ("data/raw/fake_job_postings.csv"))

# Load dataset
df = pd.read_csv(data_path)

# Fill missing values in text columns
text_fields = [
    'title', 'location', 'department', 'salary_range', 'company_profile',
    'description', 'requirements', 'benefits', 'employment_type',
    'required_experience', 'required_education', 'industry', 'function'
]
df[text_fields] = df[text_fields].fillna("")

# Combine text fields
df['text'] = df[text_fields].agg(" ".join, axis=1)

# Target
y = df['fraudulent']

# Vectorizer with fixed features
vectorizer = TfidfVectorizer(stop_words="english", max_features=500)
X = vectorizer.fit_transform(df['text'])

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# Save both in models/ folder
model_path = os.path.join(base_dir, "models", "model.pkl")
vectorizer_path = os.path.join(base_dir, "models", "vectorizer.pkl")
joblib.dump(model, model_path)
joblib.dump(vectorizer, vectorizer_path)

print("✅ Model and vectorizer retrained and saved with exactly 500 features.")



