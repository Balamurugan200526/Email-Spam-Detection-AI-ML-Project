"""
visualization.py
------------------
Generates all Exploratory Data Analysis (EDA) charts used in the project
report and Jupyter notebook. Each function saves its figure into
outputs/graphs/ and also displays it inline when run in a notebook.
"""

import os
from collections import Counter

import matplotlib
matplotlib.use("Agg")  # safe for headless script execution
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from wordcloud import WordCloud

sns.set_style("whitegrid")
GRAPH_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "graphs")
os.makedirs(GRAPH_DIR, exist_ok=True)


def _save(fig, name):
    path = os.path.join(GRAPH_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")


def plot_class_distribution(df: pd.DataFrame, label_col: str = "Label"):
    """Spam vs Ham distribution (bar chart)."""
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(x=label_col, data=df, hue=label_col,
                  palette=["#e74c3c", "#2ecc71"], legend=False, ax=ax)
    ax.set_title("Spam vs Ham Distribution")
    ax.set_xlabel("Class")
    ax.set_ylabel("Number of Emails")
    _save(fig, "class_distribution.png")
    plt.close(fig)


def plot_message_length_distribution(df: pd.DataFrame):
    """Distribution of message length, split by class."""
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(data=df, x="Message_Length", hue="Label", bins=40, kde=True,
                 palette=["#e74c3c", "#2ecc71"], ax=ax)
    ax.set_title("Message Length Distribution by Class")
    _save(fig, "message_length_distribution.png")
    plt.close(fig)


def plot_feature_correlation(df: pd.DataFrame):
    """Correlation heatmap of numeric metadata features."""
    numeric_cols = ["Message_Length", "Number_of_Links",
                     "Number_of_Special_Characters", "Number_of_Capital_Letters"]
    fig, ax = plt.subplots(figsize=(6, 5))
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    ax.set_title("Correlation Between Numeric Features")
    _save(fig, "correlation_heatmap.png")
    plt.close(fig)


def plot_wordcloud(texts, title, filename):
    """Word cloud of the most frequent words in a set of texts."""
    combined = " ".join(texts)
    wc = WordCloud(width=800, height=400, background_color="white",
                    colormap="viridis").generate(combined)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(title)
    _save(fig, filename)
    plt.close(fig)


def plot_top_keywords(texts, title, filename, top_n: int = 15):
    """Bar chart of the top-N most frequent words in a set of cleaned texts."""
    words = " ".join(texts).split()
    counts = Counter(words).most_common(top_n)
    words_list, freqs = zip(*counts) if counts else ([], [])

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=list(freqs), y=list(words_list), hue=list(words_list),
                palette="viridis", legend=False, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Frequency")
    _save(fig, filename)
    plt.close(fig)


def plot_numeric_feature_by_class(df: pd.DataFrame, feature: str, filename: str):
    """Boxplot comparing a numeric feature across Spam vs Ham."""
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(x="Label", y=feature, data=df, hue="Label",
                palette=["#e74c3c", "#2ecc71"], legend=False, ax=ax)
    ax.set_title(f"{feature.replace('_', ' ')} by Class")
    _save(fig, filename)
    plt.close(fig)


if __name__ == "__main__":
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "dataset", "spam_dataset.csv"))
    plot_class_distribution(df)
    plot_message_length_distribution(df)
    plot_feature_correlation(df)
    plot_numeric_feature_by_class(df, "Number_of_Links", "links_by_class.png")
    plot_numeric_feature_by_class(df, "Number_of_Capital_Letters", "capitals_by_class.png")

    spam_texts = df[df["Label"] == "Spam"]["Email_Text"].tolist()
    ham_texts = df[df["Label"] == "Ham"]["Email_Text"].tolist()
    plot_wordcloud(spam_texts, "Spam Email Word Cloud", "spam_wordcloud.png")
    plot_wordcloud(ham_texts, "Ham Email Word Cloud", "ham_wordcloud.png")
    print("All EDA visualizations generated.")
