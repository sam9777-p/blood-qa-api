from flask import Flask, request, jsonify
import lightgbm as lgb
import pandas as pd

app = Flask(__name__)

# Load the model
model = lgb.Booster(model_file="eligibility_model.txt")

# Define feature columns (must match training data)
FEATURE_COLUMNS = [
    "Age", "HemoglobinLevel", "Weight", "BloodPressureSys",
    "BloodPressureDia", "PulseRate", "PreviousDonationInterval",
    "HasChronicIllness", "IsSmoker"
]

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    try:
        # Convert JSON to DataFrame
        df = pd.DataFrame([data], columns=FEATURE_COLUMNS)

        # Predict
        pred_prob = model.predict(df)[0]
        pred = int(pred_prob > 0.5)

        return jsonify({
            "eligible": bool(pred),
            "confidence": round(pred_prob, 4)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
