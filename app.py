from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os
import re
from datetime import datetime

app = Flask(__name__)
CORS(app)

# ✅ Load trained model
model_path = os.path.join(os.path.dirname(__file__), "model", "rf_parking_model.pkl")
model = joblib.load(model_path)

# 🧠 NLP feature extractor
def extract_features_from_text(text):
    text = text.lower()

    # 1. Day
    days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    day_features = [1 if day in text else 0 for day in days]

    # 2. Hour
    hour_match = re.search(r'(\d{1,2})(?:\s*pm|\s*am)?', text)
    hour = int(hour_match.group(1)) if hour_match else 12
    if "pm" in text and hour != 12:
        hour += 12
    if "am" in text and hour == 12:
        hour = 0

    # 3. Peak hour (assume peak hours are 17–20)
    peak_hour = 1 if 17 <= hour <= 20 else 0

    # 4. Paid/Free (1 for Paid, 0 for Free)
    is_paid = 1 if "paid" in text or "pay" in text else 0

    # 5. Cleaning weekday (only one active based on keyword)
    cleaning_days = ["mon", "tue", "wed", "thu", "fri"]
    cleaning_features = [1 if f"cleaning {d}" in text or f"{d} cleaning" in text else 0 for d in cleaning_days]

    # 6. Start and end cleaning hour (dummy default)
    cleaning_start_hour = 8
    cleaning_end_hour = 10

    # 7. Zone ID from street (example logic)
    if "hiram" in text:
        zone = 2
    elif "main" in text:
        zone = 3
    else:
        zone = 1

    # 8. Weekend
    weekend = 1 if "saturday" in text or "sunday" in text else 0

    # 9. Popularity & Safety (basic dummy values)
    popularity = 2  # assume 0 to 3 scale
    cleaning_risk = 1
    safety = 3

    # Final input vector (based on your ML training format)
    features = [
        hour,
        is_paid,
        day_features[0],  # Monday
        day_features[1],  # Tuesday
        day_features[2],  # Wednesday
        day_features[3],  # Thursday
        day_features[4],  # Friday
        day_features[5],  # Saturday
        day_features[6],  # Sunday
        *cleaning_features,  # mon to fri
        cleaning_start_hour,
        cleaning_end_hour,
        peak_hour,
        weekend,
        popularity,
        cleaning_risk,
        safety
    ]
    return [features]

# 🔮 Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        message = data.get("message", "")

        features = extract_features_from_text(message)
        prediction = model.predict(features)

        return jsonify({
            "reply": f"Slot {prediction[0]} is available near you 🚗\n(You asked: {message})"
        })
    except Exception as e:
        print("Prediction error:", e)
        return jsonify({"reply": "❌ Sorry, something went wrong."}), 500

# 🚀 Start server
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(debug=False, host="0.0.0.0", port=port)
