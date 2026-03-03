import pytest
import numpy as np
import requests
import time
import subprocess
from score import score
from app import app


# -------------------------
# SCORE FUNCTION TESTS
# -------------------------

def test_smoke_test():
    try:
        result = score("Example text", threshold=0.5)
    except Exception as e:
        pytest.fail(f"Score function crashed: {e}")

    assert isinstance(result, tuple)
    assert len(result) == 2


def test_format_test():
    prediction, probability = score("Example text", threshold=0.7)

    assert isinstance(prediction, np.bool_)
    assert isinstance(probability, float)


def test_prediction_0_or_1():
    prediction, _ = score("Example text", threshold=0.5)
    assert int(prediction) in (0, 1)


def test_propensity_between_0_and_1():
    _, propensity = score("Example text", threshold=0.5)
    assert 0 <= propensity <= 1


def test_when_threshold_0_prediction_always_1():
    prediction, _ = score("Any random text", threshold=0)
    assert int(prediction) == 1


def test_when_threshold_1_prediction_always_0():
    prediction, _ = score("Any random text", threshold=1)
    assert int(prediction) == 0


def test_obvious_spam_gives_prediction_1():
    text = """Congratulations! You have won a free lottery ticket.
              Claim now before offer expires."""
    prediction, _ = score(text, threshold=0.5)
    assert int(prediction) == 1


def test_obvious_non_spam_gives_prediction_0():
    text = "Don't forget about tomorrow's meeting at 10 AM."
    prediction, _ = score(text, threshold=0.5)
    assert int(prediction) == 0


# -------------------------
# FLASK APP TESTS
# -------------------------

@pytest.fixture
def client():
    with app.test_client() as client:
        yield client


def test_homepage(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"Spam Classifier" in response.data


def test_prediction_form_data(client):
    response = client.post("/", data={"text": "Win a free vacation now!"})
    assert response.status_code == 200
    assert "prediction" in response.json
    assert "propensity" in response.json


def test_prediction_json(client):
    response = client.post("/", json={"text": "You have won a lottery!"})
    assert response.status_code == 200
    assert "prediction" in response.json
    assert "propensity" in response.json


def test_missing_text(client):
    response = client.post("/", json={})
    assert response.status_code == 400
    assert response.json == {"error": "No input text provided"}


# -------------------------
# LIVE SERVER TEST (Optional but required in many assignments)
# -------------------------

def test_flask_live():
    process = subprocess.Popen(["python", "app.py"])
    time.sleep(2)

    payload = {"text": "Congratulations! You won a prize!"}
    response = requests.post("http://127.0.0.1:5000/", data=payload)

    assert response.status_code == 200
    data = response.json()

    assert "prediction" in data
    assert "propensity" in data

    process.terminate()