from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained model and scaler
bundle = joblib.load("mobile_price_model.pkl")

model = bundle["model"]
scaler = bundle["scaler"]

# Features used during model training
FEATURES = [
    "battery_power",
    "blue",
    "clock_speed",
    "dual_sim",
    "fc",
    "four_g",
    "int_memory",
    "m_dep",
    "mobile_wt",
    "n_cores",
    "pc",
    "px_height",
    "px_width",
    "ram",
    "sc_h",
    "sc_w",
    "talk_time",
    "three_g",
    "touch_screen",
    "wifi"
]


@app.get("/")
def health():
    return jsonify({
        "status": "ok",
        "service": "mobile-price-prediction"
    })


@app.post("/predict")
def predict():

    data = request.get_json(silent=True)

    if not data:
        return jsonify({
            "error": "JSON request body is required"
        }), 400

    # Check for missing features
    missing_fields = [
        feature for feature in FEATURES
        if feature not in data
    ]

    if missing_fields:
        return jsonify({
            "error": "Missing required fields",
            "missing_fields": missing_fields
        }), 400

    try:
        # Keep feature order exactly the same as training
        input_data = pd.DataFrame(
            [[data[feature] for feature in FEATURES]],
            columns=FEATURES
        )

        # Apply the same scaler used during training
        input_scaled = scaler.transform(input_data)

        # Make prediction
        prediction = int(model.predict(input_scaled)[0])

        return jsonify({
            "prediction": prediction,
            "price_range": prediction
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
