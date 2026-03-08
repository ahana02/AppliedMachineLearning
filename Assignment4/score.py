import joblib
import numpy as np
from typing import Tuple
import os


# Load model and vectorizer once (global load)
MODEL_PATH = os.path.join("model", "best_model.pkl")
VECTORIZER_PATH = os.path.join("model", "vectorizer.pkl")

if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
else:
    model = None
    vectorizer = None


def score(text: str, model_input=None, threshold: float = 0.5) -> Tuple[np.bool_, float]:
    """
    Score a text and return:
    - prediction (numpy.bool_)
    - propensity (float)
    """

    # Required assertions for tests
    assert type(text) == str
    assert ((type(threshold) == float) or type(threshold) == int) and (0 <= threshold <= 1)

    if model is None or vectorizer is None:
        raise Exception("Model or vectorizer not loaded")

    # Transform text
    text_vectorized = vectorizer.transform([text])

    # Get probability of spam (class 1)
    propensity = model.predict_proba(text_vectorized)[0][1]

    # Apply threshold
    prediction = np.bool_(propensity >= threshold)

    return prediction, float(propensity)