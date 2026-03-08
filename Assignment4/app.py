from flask import Flask, request, jsonify
from score import score
import warnings

warnings.filterwarnings("ignore")

app = Flask(__name__)


# ---------------------------------------------------
# Home Route (Form + API support)
# ---------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def home():

    if request.method == "POST":
        try:
            # Handle JSON request
            if request.is_json:
                data = request.get_json()

                if not data or "text" not in data or not data["text"].strip():
                    return jsonify({"error": "No input text provided"}), 400

                text = data["text"].strip()

            # Handle Form request
            else:
                text = request.form.get("text", "").strip()

                if not text:
                    return jsonify({"error": "No input text provided"}), 400

            prediction, probability = score(text, threshold=0.5)

            return jsonify({
                "prediction": int(prediction),
                "propensity": float(probability)
            })

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # GET request → Show simple HTML page
    return """
        <html>
        <head><title>Spam Classifier</title></head>
        <body>
            <h1>Spam Classifier</h1>
            <form method="post">
                <input type="text" name="text" required>
                <input type="submit">
            </form>
        </body>
        </html>
    """


# ---------------------------------------------------
# API Route (Required for Docker test)
# ---------------------------------------------------
@app.route("/score", methods=["POST"])
def score_endpoint():
    try:
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400

        data = request.get_json()

        if not data or "text" not in data or not data["text"].strip():
            return jsonify({"error": "No input text provided"}), 400

        text = data["text"].strip()

        prediction, probability = score(text, threshold=0.5)

        return jsonify({
            "prediction": int(prediction),
            "propensity": float(probability)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------
# Run App
# ---------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)