from flask import Flask, render_template, request
import joblib

# Initialize Flask app
app = Flask(__name__)

# Load model and vectorizer from models/ folder
model = joblib.load("models/model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

# Combine important text fields into a single string
def combine_text_fields(data):
    fields = [
        'title', 'location', 'department', 'salary_range', 'company_profile',
        'description', 'requirements', 'benefits', 'employment_type',
        'required_experience', 'required_education', 'industry', 'function'
    ]
    return " ".join([data.get(field, "") for field in fields])

# Preprocess form input using the trained vectorizer (no extra binary features)
def preprocess_input(data):
    combined_text = combine_text_fields(data)
    vectorized_text = vectorizer.transform([combined_text])
    return vectorized_text

# Home route: render the form
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

# Predict route: handle form submission
@app.route("/predict", methods=["POST"])
def predict():
    form_data = request.form
    X = preprocess_input(form_data)

    prediction = model.predict(X)[0]
    proba = model.predict_proba(X)[0][prediction] * 100

    result = "Fake" if prediction == 1 else "Real"
    return render_template("index.html", prediction=result, probability=f"{proba:.2f}")

# # Run the app
if __name__ == "__main__":
    app.run(debug=True)