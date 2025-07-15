# End-to-End-Fake-Job-Detection

# Fake Job Posting Detection


### A complete end-to-end machine learning project that identifies and classifies job listings as real or fake based on their content. This solution uses NLP, feature engineering, and classification algorithms, and is deployed as a web application using Flask.

## 🚀 Project Overview

### Fake job postings are a growing issue across online platforms. This project aims to build a machine learning pipeline to:

#### Detect and classify fake vs. real job postings

#### Provide actionable predictions through a user-friendly web interface 

## 🧰 Technologies Used

Python 3.10

Flask

Scikit-learn

Pandas

Matplotlib & Seaborn

TF-IDF Vectorization

Joblib

Render (deployment)

🧠 Model

Text Vectorization: TfidfVectorizer with 500 features

Classifier: Random Forest (tuned with GridSearchCV)

Evaluation Metrics: Accuracy, Precision, Recall, F1-score

📊 Exploratory Data Analysis Highlights

Real jobs tend to have structured, detailed descriptions

Fake jobs often use vague or persuasive language (e.g. "earn fast", "no experience")

Features like has_company_logo, telecommuting, and has_questions are important

🖥️ Web App Features

Input fields for job details (title, description, requirements, etc.)

Checkbox fields for binary values (has logo, telecommute, etc.)

Returns prediction label (Fake or Real) and confidence score
