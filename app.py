from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)

# Load the trained model and encoder
model = joblib.load("Linear Regression/model/insurance_model.pkl")
encoder = joblib.load("Linear Regression/model/onehot_encoder.pkl")


@app.route("/")
def home():
    return render_template("gui.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        # Extract features from request
        age = float(data["age"])
        sex = data["sex"]
        bmi = float(data["bmi"])
        children = int(data["children"])
        smoker = data["smoker"]
        region = data["region"]

        # One-hot encoding for sex and region
        encoded_features = encoder.transform([[sex, region]])

        # Encoded smoker
        encoded_smoker = 1 if smoker == "yes" else 0

        # Feature engineering (same as training)
        bmi_age = bmi * age
        smoker_bmi = encoded_smoker * bmi
        smoker_age = encoded_smoker * age
        bmi_age_smoker = bmi * age * encoded_smoker

        # Age group
        if age <= 30:
            age_group = 0
        elif age <= 40:
            age_group = 1
        elif age <= 50:
            age_group = 2
        elif age <= 60:
            age_group = 3
        else:
            age_group = 4

        # BMI category
        if bmi <= 18.5:
            bmi_category = 0
        elif bmi <= 25:
            bmi_category = 1
        elif bmi <= 30:
            bmi_category = 2
        else:
            bmi_category = 3

        # Combine all features in correct order
        features = np.array(
            [
                [
                    age,
                    bmi,
                    children,
                    encoded_features[0][0],  # sex_female
                    encoded_features[0][1],  # sex_male
                    encoded_features[0][2],  # region_northeast
                    encoded_features[0][3],  # region_northwest
                    encoded_features[0][4],  # region_southeast
                    encoded_features[0][5],  # region_southwest
                    encoded_smoker,
                    bmi_age,
                    smoker_bmi,
                    smoker_age,
                    bmi_age_smoker,
                    age_group,
                    bmi_category,
                ]
            ]
        )

        # Make prediction (result is in log scale)
        prediction_log = model.predict(features)[0]

        # Convert back from log scale
        prediction_original = np.expm1(prediction_log)

        return jsonify(
            {"success": True, "prediction": round(float(prediction_original), 2)}
        )

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True, port=5000)

