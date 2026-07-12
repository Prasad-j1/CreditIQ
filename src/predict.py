# src/predict.py
# Handles loading the trained pipeline and running predictions on new input.
# This file knows nothing about Flask, HTML, or forms — it only knows about
# the ML pipeline. That separation is what makes it reusable and testable.

import joblib
import pandas as pd
from src.utils import RISK_THRESHOLD


def load_pipeline(model_path='flask_app/model/credit_risk_pipeline.pkl'):
    """Load the trained pipeline (preprocessing + model) from disk."""
    return joblib.load(model_path)


def make_prediction(pipeline, input_data: dict):
    """
    Takes a dictionary of raw feature values, runs it through the pipeline,
    and returns the predicted label and probability of bad credit.

    Uses RISK_THRESHOLD instead of sklearn's default 0.5 cutoff, since our
    business problem prioritizes catching bad-credit customers (recall)
    over minimizing false alarms. See utils.py for the reasoning.
    """
    input_df = pd.DataFrame([input_data])

    probability = pipeline.predict_proba(input_df)[0][1]  # probability of class 1 ("bad")

    prediction = 1 if probability >= RISK_THRESHOLD else 0
    label = "Bad Credit" if prediction == 1 else "Good Credit"

    return {
        'prediction': label,
        'probability_bad': round(probability * 100, 2)
    }
