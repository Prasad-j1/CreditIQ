# src/utils.py
# Small, reusable helper functions used by the Flask app.
# Keeping these separate from app.py and predict.py keeps each file focused
# on one job: predict.py = ML logic, utils.py = business/display logic, app.py = routing.

RISK_THRESHOLD = 0.25
JOB_LABELS = {
    0: 'Unskilled & Non-resident',
    1: 'Unskilled & Resident',
    2: 'Skilled',
    3: 'Highly Skilled / Management'
}

# Dataset averages, computed once from training data (Phase 2 describe()).
# Used to judge whether an applicant's numeric values are "above average".
FEATURE_AVERAGES = {
    'Age': 35.5,
    'Credit amount': 3271.0,
    'Duration': 20.9
}


def get_risk_category(probability_bad):
    """
    Takes probability of bad credit (0-100 scale) and returns
    a risk category label and a color code for the UI.
    Thresholds are centered around the model's natural 50% decision
    boundary: 0-30 = confidently good, 30-60 = the uncertain zone
    right around the boundary, 60-100 = confidently bad.
    """
    if probability_bad < 30:
        return {'category': 'Low Risk', 'color': 'green'}
    elif probability_bad < 60:
        return {'category': 'Medium Risk', 'color': 'orange'}
    else:
        return {'category': 'High Risk', 'color': 'red'}


def get_confidence_score(probability_bad):
    """
    Confidence = how far the model's probability is from the uncertain
    50% midpoint, scaled back to a 50-100% range.
    E.g. 91.87% bad -> model is very confident -> high confidence score.
    E.g. 52% bad -> model is barely leaning either way -> low confidence score.
    """
    distance_from_midpoint = abs(probability_bad - 50)
    confidence = 50 + distance_from_midpoint
    return round(confidence, 2)


def get_recommendation(risk_category):
    """Returns a plain-language recommendation based on risk category."""
    messages = {
        'Low Risk': "This loan appears relatively safe based on the applicant's profile. Standard approval process is recommended.",
        'Medium Risk': "This application shows a mix of risk factors. Additional financial verification is recommended before approval.",
        'High Risk': "This applicant has a higher probability of default based on historical patterns. Manual review and stricter terms are recommended."
    }
    return messages.get(risk_category, "Unable to generate a recommendation for this profile.")


def get_explanation_factors(input_data: dict):
    """
    Returns a list of human-readable factors that likely influenced
    the prediction, based on the applicant's actual input values.
    'effect': 'positive' = good for the applicant (lowers risk)
    'effect': 'negative' = raises risk
    """
    factors = []

    if input_data.get('Checking account') == 'none':
        factors.append({'text': 'No checking account on record', 'effect': 'positive'})
    elif input_data.get('Checking account') == 'rich':
        factors.append({'text': 'Strong checking account balance', 'effect': 'positive'})
    elif input_data.get('Checking account') == 'little':
        factors.append({'text': 'Low checking account balance', 'effect': 'negative'})

    if input_data.get('Saving accounts') == 'rich':
        factors.append({'text': 'Rich savings account', 'effect': 'positive'})
    elif input_data.get('Saving accounts') == 'little':
        factors.append({'text': 'Limited savings', 'effect': 'negative'})

    if input_data.get('Purpose') == 'education':
        factors.append({'text': 'Education loan purpose', 'effect': 'negative'})

    if input_data.get('Housing') == 'own':
        factors.append({'text': 'Applicant owns their home', 'effect': 'positive'})
    elif input_data.get('Housing') == 'rent':
        factors.append({'text': 'Applicant rents their home', 'effect': 'negative'})

    try:
        if float(input_data.get('Duration', 0)) > FEATURE_AVERAGES['Duration']:
            factors.append({'text': 'Longer than average loan duration', 'effect': 'negative'})
    except (ValueError, TypeError):
        pass

    try:
        if float(input_data.get('Credit amount', 0)) > FEATURE_AVERAGES['Credit amount']:
            factors.append({'text': 'Higher than average credit amount', 'effect': 'negative'})
    except (ValueError, TypeError):
        pass

    return factors