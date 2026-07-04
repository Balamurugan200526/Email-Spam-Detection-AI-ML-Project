"""
predict.py
------------
Standalone prediction script. Loads the saved model + vectorizer and lets a
user classify a custom email message as Spam or Ham, with a confidence score.

Usage (interactive):
    python predict.py

Usage (single message from command line):
    python predict.py "Congratulations! You have won a free prize, click here"
"""

import os
import sys

import joblib

sys.path.append(os.path.dirname(__file__))
from preprocessing import clean_text

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
MODEL_DIR = os.path.join(BASE_DIR, "models")

_model = None
_vectorizer = None


def _load_artifacts():
    global _model, _vectorizer
    if _model is None or _vectorizer is None:
        _model = joblib.load(os.path.join(MODEL_DIR, "spam_classifier.pkl"))
        _vectorizer = joblib.load(os.path.join(MODEL_DIR, "vectorizer.pkl"))
    return _model, _vectorizer


def predict_email(raw_text: str) -> dict:
    """
    Classify a single raw email message.

    Returns
    -------
    dict with keys: label ("Spam"/"Ham"), confidence (float 0-1),
    spam_probability, ham_probability
    """
    model, vectorizer = _load_artifacts()

    cleaned = clean_text(raw_text)
    features = vectorizer.transform([cleaned])

    prediction = model.predict(features)[0]
    label = "Spam" if prediction == 1 else "Ham"

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(features)[0]
        ham_prob, spam_prob = proba[0], proba[1]
    else:
        # Fallback for models without predict_proba
        spam_prob = float(prediction)
        ham_prob = 1 - spam_prob

    confidence = spam_prob if label == "Spam" else ham_prob

    return {
        "label": label,
        "confidence": round(float(confidence) * 100, 2),
        "spam_probability": round(float(spam_prob) * 100, 2),
        "ham_probability": round(float(ham_prob) * 100, 2),
        "cleaned_text": cleaned,
    }


def main():
    if len(sys.argv) > 1:
        message = " ".join(sys.argv[1:])
        result = predict_email(message)
        print(f"\nMessage : {message}")
        print(f"Prediction : {result['label']}")
        print(f"Confidence : {result['confidence']}%")
        print(f"(Spam: {result['spam_probability']}% | Ham: {result['ham_probability']}%)")
        return

    print("=== Email Spam Detector -- Interactive Mode ===")
    print("Type an email message and press Enter (or 'quit' to exit).\n")
    while True:
        message = input("Enter email text: ").strip()
        if message.lower() in ("quit", "exit"):
            break
        if not message:
            continue
        result = predict_email(message)
        print(f"  -> Prediction: {result['label']} "
              f"(Confidence: {result['confidence']}%)\n")


if __name__ == "__main__":
    main()
