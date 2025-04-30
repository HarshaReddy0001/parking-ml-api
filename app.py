from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os

app = Flask(__name__)
CORS(app)

# ✅ Load your trained model using correct absolute path
model_path = os.path.join(os.path.dirname(__file__), "model", "rf_parking_model.pkl")
model = joblib.load(model_path)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        message = data.get("message", "")

        # 🔁 TODO: Replace with actual preprocessing logic
        # For testing, let's just simulate input features:
        # Example dummy input: [day_of_week, hour_of_day, zone]
        example_features = [[6, 18, 2]]  # Replace this with actual logic from `message`

        prediction = model.predict(example_features)

        return jsonify({
            "reply": f"Slot {prediction[0]} is available near you 🚗\n(You asked: {message})"
        })
    except Exception as e:
        print("Prediction error:", e)
        return jsonify({"reply": "❌ Sorry, something went wrong."}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(debug=False, host="0.0.0.0", port=port)
