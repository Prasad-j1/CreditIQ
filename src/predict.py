# src/predict.py
# Handles loading the trained pipeline and running predictions on new input.
# This file knows nothing about Flask, HTML, or forms — it only knows about
# the ML pipeline. That separation is what makes it reusable and testable.

import joblib
import pandas as pd


def load_pipeline(model_path='flask_app/model/credit_risk_pipeline.pkl'):
    """Load the trained pipeline (preprocessing + model) from disk."""
    return joblib.load(model_path)


def make_prediction(pipeline, input_data: dict):
    """
    Takes a dictionary of raw feature values, runs it through the pipeline,
    and returns the predicted label and probability of bad credit.
    The pipeline itself handles all scaling/encoding internally — this
    function never touches preprocessing directly.
    """
    input_df = pd.DataFrame([input_data])

    prediction = pipeline.predict(input_df)[0]
    probability = pipeline.predict_proba(input_df)[0][1]  # probability of class 1 ("bad")

    label = "Bad Credit" if prediction == 1 else "Good Credit"

    return {
        'prediction': label,
        'probability_bad': round(probability * 100, 2)
    }