from flask import Flask, request, render_template
import joblib
import re
import string

# Load the trained model and vectorizer
model = joblib.load("xgb_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Function to clean input text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Initialize Flask app
app = Flask(__name__,template_folder='templates')

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        review_text = request.form["review"]
        cleaned_text = clean_text(review_text)
        text_vector = vectorizer.transform([cleaned_text])
        pred = model.predict(text_vector)[0]
        prediction = "Genuine" if pred == 1 else "Fraudulent"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)