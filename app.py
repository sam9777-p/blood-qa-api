from flask import Flask, request, jsonify
import numpy as np

app = Flask(__name__)

BMI_data = [18.5, 22.0, 24.0, 28.0, 30.0]
blood_volume_data = [450, 500, 520, 560, 590]

def predict_blood_volume(bmi_input):
    x = np.array(BMI_data)
    y = np.array(blood_volume_data)
    n = len(x)
    sum_x, sum_y = x.sum(), y.sum()
    sum_x2, sum_xy = (x ** 2).sum(), (x * y).sum()
    m = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
    b = (sum_y - m * sum_x) / n
    return round(m * bmi_input + b, 2)

@app.route("/")
def index():
    return "Blood Donation Eligibility Linear Regression API is running."

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    bmi = data.get("bmi")
    if bmi is None:
        return jsonify({"error": "Please provide a BMI value."}), 400
    prediction = predict_blood_volume(float(bmi))
    return jsonify({"bmi": bmi, "predicted_blood_volume": prediction})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
