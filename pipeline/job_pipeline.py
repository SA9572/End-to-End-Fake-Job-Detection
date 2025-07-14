import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
import joblib

def build_pipeline():
    # Feature categories
    text_features = ['title', 'description', 'requirements']
    categorical_features = ['employment_type', 'required_experience', 'required_education', 'industry', 'function']
    numeric_features = ['telecommuting', 'has_company_logo', 'has_questions']

    # Pipelines for each type of feature
    text_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', max_features=300))
    ])

    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Combine all transformers
    preprocessor = ColumnTransformer([
        ('text_title', text_pipeline, 'title'),
        ('text_desc', text_pipeline, 'description'),
        ('text_req', text_pipeline, 'requirements'),
        ('cat', categorical_pipeline, categorical_features),
        ('num', numeric_pipeline, numeric_features)
    ])

    # Final pipeline with classifier
    pipeline = Pipeline([
        ('preprocess', preprocessor),
        ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    return pipeline

def train_pipeline(X, y, model_path='models/fake_job_model.pkl'):
    pipeline = build_pipeline()
    pipeline.fit(X, y)
    joblib.dump(pipeline, model_path)
    print(f"Model pipeline saved to {model_path}")

def predict_single(sample_input, model_path='models/fake_job_model.pkl'):
    pipeline = joblib.load(model_path)
    return pipeline.predict([sample_input])[0]
