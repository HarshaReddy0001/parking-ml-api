from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os

app = Flask(__name__)
CORS(app)

# Load your trained model (only once when app starts)
model_path = os.path.join("model", "rf_parking_model.pkl")
model = joblib.load(model_path)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    message = data.get("message", "")

    # For now, let’s simulate prediction (you will later plug real logic)
    # Replace this with your real feature extraction
    prediction = model.predict([[6, 6, 2]])  # <- replace with real features extracted from `message`

    return jsonify({
        "reply": f"Slot {prediction[0]} is available near you 🚗 (You asked: {message})"
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(debug=False, host="0.0.0.0", port=port)
