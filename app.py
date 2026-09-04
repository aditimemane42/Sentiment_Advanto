import os
import pickle
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

MODEL_PATH = "model.pkl"

# Load the saved model/vectorizer pipeline
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.route("/", methods=["GET", "POST"])
def home():
    sentiment = None
    review_text = ""

    if request.method == "POST":
        review_text = request.form.get("review", "")
        if review_text and model:
            try:
                # Predict sentiment using the unpickled pipeline
                prediction = model.predict([review_text])[0]
                
                # Format prediction for display
                if str(prediction).lower() in ["1", "positive", "pos"]:
                    sentiment = "Positive"
                elif str(prediction).lower() in ["0", "negative", "neg"]:
                    sentiment = "Negative"
                else:
                    sentiment = str(prediction)
            except Exception as e:
                sentiment = f"Error during prediction: {e}"

    return render_template("index.html", sentiment=sentiment, review=review_text)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
