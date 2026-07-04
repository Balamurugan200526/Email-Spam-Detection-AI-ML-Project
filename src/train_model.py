"""
train_model.py
----------------
End-to-end training pipeline:
    1. Load dataset
    2. Clean text (preprocessing.py)
    3. Extract features -- compares CountVectorizer vs TF-IDF (feature_extraction.py)
    4. Train 6 classifiers: Naive Bayes, Logistic Regression, Decision Tree,
       Random Forest, SVM, KNN
    5. Evaluate all models, pick the best performing one
    6. Save the best model + vectorizer with Joblib for later use in
       predict.py and the Streamlit app.
"""

import os
import sys
import time
import json

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score)

sys.path.append(os.path.dirname(__file__))
from preprocessing import clean_text
from feature_extraction import fit_transform_features

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "spam_dataset.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
REPORT_DIR = os.path.join(BASE_DIR, "outputs", "reports")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

RANDOM_STATE = 42


def load_and_clean_data():
    print("Loading dataset...")
    df = pd.read_csv(DATASET_PATH)
    print(f"  {len(df)} records loaded.")

    print("Cleaning text (lowercasing, removing punctuation/numbers, "
          "tokenizing, removing stopwords, lemmatizing)...")
    df["Cleaned_Text"] = df["Email_Text"].apply(clean_text)
    df["Label_Num"] = df["Label"].map({"Ham": 0, "Spam": 1})
    return df


def compare_feature_methods(df):
    """Compare CountVectorizer vs TF-IDF using a fixed Logistic Regression
    model, to decide which feature-extraction approach to use going forward."""
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df["Cleaned_Text"], df["Label_Num"], test_size=0.2,
        random_state=RANDOM_STATE, stratify=df["Label_Num"])

    results = {}
    for method in ["count", "tfidf"]:
        X_train, X_test, _ = fit_transform_features(
            X_train_text, X_test_text, method=method, max_features=3000)
        clf = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        acc = accuracy_score(y_test, preds)
        results[method] = acc
        print(f"  {method.upper():6s} -> Logistic Regression accuracy: {acc:.4f}")

    best_method = max(results, key=results.get)
    print(f"Best feature extraction method: {best_method.upper()}\n")
    return best_method, results


def train_all_models(X_train, X_test, y_train, y_test):
    """Train and evaluate all 6 required classifiers."""
    models = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
        "Support Vector Machine": SVC(kernel="linear", probability=True, random_state=RANDOM_STATE),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    }

    results = {}
    trained_models = {}

    for name, model in models.items():
        start = time.time()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        elapsed = time.time() - start

        results[name] = {
            "Accuracy": round(accuracy_score(y_test, preds), 4),
            "Precision": round(precision_score(y_test, preds), 4),
            "Recall": round(recall_score(y_test, preds), 4),
            "F1-Score": round(f1_score(y_test, preds), 4),
            "Train_Time_Seconds": round(elapsed, 3),
        }
        trained_models[name] = model
        print(f"  {name:24s} | Acc: {results[name]['Accuracy']:.4f} | "
              f"Prec: {results[name]['Precision']:.4f} | "
              f"Rec: {results[name]['Recall']:.4f} | "
              f"F1: {results[name]['F1-Score']:.4f} | "
              f"Time: {elapsed:.2f}s")

    return results, trained_models


def main():
    df = load_and_clean_data()

    print("\nComparing feature extraction methods (CountVectorizer vs TF-IDF)...")
    best_method, feature_comparison = compare_feature_methods(df)

    print("Splitting data into train/test sets (80/20 stratified split)...")
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df["Cleaned_Text"], df["Label_Num"], test_size=0.2,
        random_state=RANDOM_STATE, stratify=df["Label_Num"])

    print(f"Extracting features using {best_method.upper()}...")
    X_train, X_test, vectorizer = fit_transform_features(
        X_train_text, X_test_text, method=best_method, max_features=3000)

    print("\nTraining and evaluating 6 classifiers...")
    results, trained_models = train_all_models(X_train, X_test, y_train, y_test)

    # Pick best model by F1-score (balances precision & recall -- important
    # for spam detection where both false positives and false negatives matter)
    best_model_name = max(results, key=lambda k: results[k]["F1-Score"])
    best_model = trained_models[best_model_name]
    print(f"\nBest model selected: {best_model_name} "
          f"(F1-Score: {results[best_model_name]['F1-Score']})")

    # Save best model + vectorizer
    joblib.dump(best_model, os.path.join(MODEL_DIR, "spam_classifier.pkl"))
    joblib.dump(vectorizer, os.path.join(MODEL_DIR, "vectorizer.pkl"))
    print(f"Saved model  -> models/spam_classifier.pkl")
    print(f"Saved vectorizer -> models/vectorizer.pkl")

    # Save comparison report as JSON + CSV for the notebook/report to use
    report = {
        "feature_extraction_comparison": feature_comparison,
        "best_feature_method": best_method,
        "model_comparison": results,
        "best_model": best_model_name,
    }
    with open(os.path.join(REPORT_DIR, "model_comparison.json"), "w") as f:
        json.dump(report, f, indent=2)

    results_df = pd.DataFrame(results).T
    results_df.to_csv(os.path.join(REPORT_DIR, "model_comparison.csv"))
    print(f"Saved comparison report -> outputs/reports/model_comparison.json / .csv")

    # Save the test split predictions for the evaluation module to reuse
    return best_model, vectorizer, X_test, y_test, best_model_name, results


if __name__ == "__main__":
    main()
