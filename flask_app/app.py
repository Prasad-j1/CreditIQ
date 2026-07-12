# flask_app/app.py

import sys
import os
from flask import Flask, render_template, request, jsonify

# Allow importing from src/ folder (sibling to flask_app/)
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from predict import load_pipeline, make_prediction
from utils import get_risk_category, get_explanation_factors, get_confidence_score, get_recommendation

app = Flask(__name__)

# Load the pipeline once, when the app starts — not on every request.
pipeline = load_pipeline('flask_app/model/credit_risk_pipeline.pkl')

# Fields we expect from the frontend, and simple server-side validation rules.
# We validate again here even though the frontend already validates —
# never trust the client alone. A request could bypass the browser entirely.
REQUIRED_FIELDS = [
    'Age', 'Sex', 'Job', 'Housing', 'Saving accounts',
    'Checking account', 'Credit amount', 'Duration', 'Purpose'
]


def validate_input(data):
    """Returns an error message string if invalid, or None if valid."""
    for field in REQUIRED_FIELDS:
        if field not in data or data[field] in (None, ''):
            return f"Missing required field: {field}"

    try:
        age = float(data['Age'])
        if age < 18 or age > 100:
            return "Age must be between 18 and 100"
    except (ValueError, TypeError):
        return "Age must be a valid number"

    try:
        credit_amount = float(data['Credit amount'])
        if credit_amount <= 0:
            return "Credit amount must be greater than 0"
    except (ValueError, TypeError):
        return "Credit amount must be a valid number"

    try:
        duration = float(data['Duration'])
        if duration <= 0 or duration > 120:
            return "Duration must be between 1 and 120 months"
    except (ValueError, TypeError):
        return "Duration must be a valid number"

    try:
        job = int(data['Job'])
        if job not in (0, 1, 2, 3):
            return "Invalid job/skill level"
    except (ValueError, TypeError):
        return "Job/skill level must be a valid number"

    return None


@app.route('/')
def home():
    return render_template('dashboard.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('prediction.html')

    # POST — form was submitted via fetch() as JSON
    try:
        data = request.get_json()

        if data is None:
            return jsonify({'error': 'Invalid or missing JSON body'}), 400

        error = validate_input(data)
        if error:
            return jsonify({'error': error}), 400

        # Build the exact structure the pipeline expects
        input_data = {
            'Age': float(data['Age']),
            'Sex': data['Sex'],
            'Job': int(data['Job']),
            'Housing': data['Housing'],
            'Saving accounts': data['Saving accounts'],
            'Checking account': data['Checking account'],
            'Credit amount': float(data['Credit amount']),
            'Duration': float(data['Duration']),
            'Purpose': data['Purpose']
        }

        result = make_prediction(pipeline, input_data)
        risk_info = get_risk_category(result['probability_bad'])
        factors = get_explanation_factors(input_data)
        confidence = get_confidence_score(result['probability_bad'])
        recommendation = get_recommendation(risk_info['category'])

        return jsonify({
            'prediction': result['prediction'],
            'probability_bad': result['probability_bad'],
            'risk_category': risk_info['category'],
            'risk_color': risk_info['color'],
            'confidence': confidence,
            'factors': factors,
            'recommendation': recommendation
        })

    except Exception as e:
        # Catch-all safety net — never let the server crash silently or
        # leak a raw traceback to the frontend.
        return jsonify({'error': 'Something went wrong while processing your request.'}), 500


if __name__ == '__main__':
    app.run(debug=True)