from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os
import re
import pandas as pd
from datetime import datetime

app = Flask(__name__)
CORS(app)

# ✅ Load trained model
model_path = os.path.join(os.path.dirname(__file__), "model", "rf_parking_model.pkl")
model = joblib.load(model_path)

# 🧠 NLP feature extractor

def extract_features_from_text(text):
    text = text.lower()

    # 1. Slot Number (assumed if mentioned)
    slot_match = re.search(r'slot\s+(\d+)', text)
    slot_number = int(slot_match.group(1)) if slot_match else 0

    # 2. Timing (hour extraction)
    hour_match = re.search(r'(\d{1,2})(?:\s*(am|pm))?', text)
    hour = int(hour_match.group(1)) if hour_match else 12
    meridiem = hour_match.group(2) if hour_match else ""
    if meridiem == "pm" and hour != 12:
        hour += 12
    if meridiem == "am" and hour == 12:
        hour = 0

    # 3. Latitude & Longitude (based on street keywords)
    if "hiram" in text:
        latitude, longitude = 40.745, -74.17
    elif "main" in text:
        latitude, longitude = 40.750, -74.16
    else:
        latitude, longitude = 40.740, -74.15

    # 4. Paid/Free
    paid_free = 1 if "paid" in text or "pay" in text else 0

    # 5. Peak Hour (17:00–20:00)
    peak_hour = 1 if 17 <= hour <= 20 else 0

    # 6. Weekend
    weekend = 1 if "saturday" in text or "sunday" in text else 0

    # 7. Street Popularity
    if any(word in text for word in ["church", "gas station", "mall", "market", "main", "ground", "shops", "walmart"]):
        popularity = 3
    elif "school" in text or "college" in text:
        popularity = 2
    else:
        popularity = 1

    # 8. Cleaning Risk
    cleaning_risk = 1 if "cleaning" in text else 0

    # 9. Safety Category
    if any(word in text for word in ["unsafe", "dark", "accident"]):
        safety = 1
    elif "quiet" in text:
        safety = 2
    elif "moderate" in text:
        safety = 3
    elif "light" in text:
        safety = 4
    else:
        safety = 5

    # 10. Status (if known)
    status_encoded = 1 if "filled" in text or "occupied" in text else 0

    # Final feature DataFrame with original trained column names
    return pd.DataFrame([[
        slot_number, hour, latitude, longitude, paid_free,
        peak_hour, weekend, popularity, cleaning_risk,
        safety, status_encoded
    ]], columns=[
        'Slot Number', 'Timing', 'Latitude', 'Longitude', 'Paid/Free',
        'peak_hour', 'weekend', 'street_popularity', 'cleaning_risk',
        'safety_category', 'status_encoded'
    ])

# 🔮 Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        message = data.get("message", "")

        input_df = extract_features_from_text(message)
        prediction = model.predict(input_df)

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
