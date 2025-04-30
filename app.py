from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    question = data.get("message", "")
    return jsonify({
        "reply": f"Slot 14 is available near you 🚗 (You asked: {question})"
    })

if __name__ == '__main__':
    app.run(debug=True, port=8000)
