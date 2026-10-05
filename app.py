from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

# ============================================================
# LOAD MODEL
# ============================================================

MODEL_PATH = os.path.join(
    "models",
    "model.pkl"
)

model_package = joblib.load(
    MODEL_PATH
)

regression_model = model_package[
    "regression_model"
]

classification_model = model_package[
    "classification_model"
]

features = model_package[
    "features"
]


# ============================================================
# HOME
# ============================================================

@app.route("/", methods=["GET"])
def home():

    return jsonify({
        "message": "Stock Prediction API is running",
        "endpoints": {
            "/": "API status",
            "/test": "Test API",
            "/predict": "Stock prediction"
        }
    })


# ============================================================
# TEST
# ============================================================

@app.route("/test", methods=["GET"])
def test():

    return jsonify({
        "status": "success",
        "message": "API working successfully"
    })


# ============================================================
# PREDICTION
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        # ----------------------------------------------------
        # Get JSON data
        # ----------------------------------------------------

        data = request.get_json()

        if not data:

            return jsonify({
                "status": "error",
                "message": "No JSON data received"
            }), 400

        # ----------------------------------------------------
        # Convert input to DataFrame
        # ----------------------------------------------------

        input_data = pd.DataFrame(
            [data]
        )

        # ----------------------------------------------------
        # Check missing features
        # ----------------------------------------------------

        missing_features = [

            feature

            for feature in features

            if feature not in input_data.columns
        ]

        if missing_features:

            return jsonify({

                "status": "error",

                "message":
                    "Missing required features",

                "missing_features":
                    missing_features
            }), 400

        # ----------------------------------------------------
        # Keep features in correct order
        # ----------------------------------------------------

        input_data = input_data[
            features
        ]

        # ----------------------------------------------------
        # Convert values to numeric
        # ----------------------------------------------------

        input_data = input_data.apply(
            pd.to_numeric,
            errors="coerce"
        )

        # ----------------------------------------------------
        # Check invalid values
        # ----------------------------------------------------

        if input_data.isnull().any().any():

            return jsonify({

                "status": "error",

                "message":
                    "Input contains missing or non-numeric values"
            }), 400

        # ====================================================
        # REGRESSION
        # ====================================================

        next_close = (
            regression_model
            .predict(input_data)[0]
        )

        # ====================================================
        # CLASSIFICATION
        # ====================================================

        prediction = int(

            classification_model
            .predict(input_data)[0]
        )

        # ====================================================
        # PROBABILITY
        # ====================================================

        probability = (

            classification_model
            .predict_proba(input_data)[0]
        )

        down_probability = float(
            probability[0]
        )

        up_probability = float(
            probability[1]
        )

        # ====================================================
        # BUY / SELL SIGNAL
        # ====================================================

        if prediction == 1:

            signal = "BUY"

        else:

            signal = "SELL"

        # ====================================================
        # RESPONSE
        # ====================================================

        return jsonify({

            "status": "success",

            "predicted_next_close":
                round(
                    float(next_close),
                    4
                ),

            "prediction":
                prediction,

            "signal":
                signal,

            "probability": {

                "down":
                    round(
                        down_probability,
                        4
                    ),

                "up":
                    round(
                        up_probability,
                        4
                    )
            }
        })

    except Exception as e:

        return jsonify({

            "status": "error",

            "message":
                str(e)

        }), 500


# ============================================================
# RUN FLASK
# ============================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )