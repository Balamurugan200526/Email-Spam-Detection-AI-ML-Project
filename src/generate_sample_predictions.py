"""
generate_sample_predictions.py
--------------------------------
Generates outputs/prediction_results.csv -- a sample of test-set predictions
(actual vs predicted label + confidence) for reporting purposes.
"""

import os
import sys

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(__file__))
from preprocessing import clean_text

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "spam_dataset.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_PATH = os.path.join(BASE_DIR, "outputs", "prediction_results.csv")
RANDOM_STATE = 42


def main():
    df = pd.read_csv(DATASET_PATH)
    df["Cleaned_Text"] = df["Email_Text"].apply(clean_text)
    df["Label_Num"] = df["Label"].map({"Ham": 0, "Spam": 1})

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=RANDOM_STATE, stratify=df["Label_Num"])

    model = joblib.load(os.path.join(MODEL_DIR, "spam_classifier.pkl"))
    vectorizer = joblib.load(os.path.join(MODEL_DIR, "vectorizer.pkl"))

    X_test = vectorizer.transform(test_df["Cleaned_Text"])
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

    result_df = test_df[["Email_ID", "Subject", "Sender", "Label"]].copy()
    result_df["Predicted_Label"] = ["Spam" if p == 1 else "Ham" for p in preds]
    if proba is not None:
        result_df["Confidence_Percent"] = [
            round(max(p) * 100, 2) for p in proba
        ]
    result_df["Correct"] = result_df["Label"] == result_df["Predicted_Label"]

    result_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(result_df)} sample predictions -> {OUTPUT_PATH}")
    print(f"Test-set accuracy check: {result_df['Correct'].mean()*100:.2f}%")


if __name__ == "__main__":
    main()
