from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os

app = Flask(__name__)
CORS(app)

# Load your trained ML model
model = joblib.load("model/rf_parking_model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    message = data.get("message", "")

    # Dummy prediction logic — Replace this with actual model prediction if available
    prediction = "Slot 14 is available near you 🚗 (You asked: {})".format(message)

    return jsonify({"reply": prediction})

if __name__ == "__main__":
    # This line makes it work on Render
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
