"""
evaluation.py
---------------
Detailed evaluation of the final saved model: confusion matrix,
classification report, and ROC curve. Produces the plots used in the
project report and saves a text classification report.
"""

import os
import sys

import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, classification_report,
                              roc_curve, auc)

sys.path.append(os.path.dirname(__file__))
from preprocessing import clean_text

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "spam_dataset.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
GRAPH_DIR = os.path.join(BASE_DIR, "outputs", "graphs")
REPORT_DIR = os.path.join(BASE_DIR, "outputs", "reports")
RANDOM_STATE = 42


def load_test_split():
    df = pd.read_csv(DATASET_PATH)
    df["Cleaned_Text"] = df["Email_Text"].apply(clean_text)
    df["Label_Num"] = df["Label"].map({"Ham": 0, "Spam": 1})

    _, X_test_text, _, y_test = train_test_split(
        df["Cleaned_Text"], df["Label_Num"], test_size=0.2,
        random_state=RANDOM_STATE, stratify=df["Label_Num"])
    return X_test_text, y_test


def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"], ax=ax)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix - Best Model")
    path = os.path.join(GRAPH_DIR, "confusion_matrix.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")
    return cm


def plot_roc_curve(y_test, y_proba):
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve - Best Model")
    ax.legend(loc="lower right")
    path = os.path.join(GRAPH_DIR, "roc_curve.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")
    return roc_auc


def main():
    print("Loading saved model and vectorizer...")
    model = joblib.load(os.path.join(MODEL_DIR, "spam_classifier.pkl"))
    vectorizer = joblib.load(os.path.join(MODEL_DIR, "vectorizer.pkl"))

    X_test_text, y_test = load_test_split()
    X_test = vectorizer.transform(X_test_text)

    y_pred = model.predict(X_test)

    print("\nClassification Report:")
    report_text = classification_report(y_test, y_pred, target_names=["Ham", "Spam"])
    print(report_text)

    with open(os.path.join(REPORT_DIR, "classification_report.txt"), "w") as f:
        f.write(report_text)

    plot_confusion_matrix(y_test, y_pred)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        roc_auc = plot_roc_curve(y_test, y_proba)
        print(f"ROC-AUC: {roc_auc:.4f}")


if __name__ == "__main__":
    main()
