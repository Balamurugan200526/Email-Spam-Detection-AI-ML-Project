# Internship Project Report

## Email Spam Detection Using Machine Learning

---

## Abstract

Email remains one of the most widely used communication mediums, but it is also a
primary vector for spam, phishing, and fraudulent content. Manual filtering is
impractical at scale, making automated spam detection an essential application of
machine learning. This project presents an end-to-end system that classifies emails
as **Spam** or **Ham (legitimate)** using Natural Language Processing (NLP) techniques
combined with classical supervised learning algorithms. The pipeline covers data
collection, text preprocessing, feature extraction (Bag-of-Words and TF-IDF), training
and comparison of six classification algorithms, rigorous evaluation, and deployment
through an interactive Streamlit web application. The best-performing model, a Naive
Bayes classifier, achieved an accuracy of 97.9% and an F1-score of 97.9% on held-out
test data, demonstrating that classical ML remains highly effective for text
classification tasks of this nature.

---

## 1. Introduction

Spam emails account for a significant portion of global email traffic and range from
harmless advertising to dangerous phishing attempts designed to steal personal or
financial information. Traditional rule-based filters (keyword blacklists, sender
blocklists) are brittle and easily bypassed by evolving spam tactics. Machine learning
offers a more robust, adaptive alternative — by learning statistical patterns in the
language and structure of spam versus legitimate emails, a model can generalize to
new, previously unseen spam variants.

This project was undertaken as a 1-month AI & Machine Learning internship deliverable,
with the goal of building a complete, production-style spam classification system
suitable for academic evaluation and technical viva.

---

## 2. Problem Statement

Given the free-text content and metadata of an email, automatically and accurately
classify it as **Spam** or **Ham**, minimizing both false positives (legitimate emails
wrongly flagged as spam) and false negatives (spam emails that slip through undetected).

---

## 3. Objectives

1. Design and build a labeled email dataset suitable for supervised learning.
2. Implement a complete NLP text-preprocessing pipeline.
3. Extract meaningful numerical features from text using Bag-of-Words and TF-IDF.
4. Train and rigorously compare six different machine learning classifiers.
5. Evaluate models using industry-standard metrics (accuracy, precision, recall, F1,
   confusion matrix, ROC-AUC).
6. Deploy the final trained model as an interactive web application.
7. Document the system comprehensively for academic submission and technical review.

---

## 4. Literature Survey

Email spam detection has been extensively studied in machine learning literature.
Early approaches relied on **Naive Bayes classifiers**, which remain popular due to
their simplicity, speed, and strong performance on text classification despite the
"naive" assumption of feature independence — an assumption that, while technically
incorrect for natural language, works surprisingly well in practice for bag-of-words
representations. Subsequent research explored **Support Vector Machines (SVMs)**,
which handle high-dimensional sparse data (typical of text) effectively via the kernel
trick, and **ensemble methods** like Random Forests, which reduce overfitting by
aggregating many decision trees. More recent work has moved toward deep learning
architectures (RNNs, LSTMs, and transformer-based models like BERT), which capture
contextual and sequential relationships in text more richly, at the cost of higher
computational requirements and reduced interpretability. For a project of this scope,
classical ML algorithms were chosen because they offer an excellent accuracy-to-
complexity ratio, fast training/inference, and full interpretability — all valuable
properties in a resource-constrained, deployable spam filter.

---

## 5. Methodology

The project follows a standard supervised machine learning workflow:

1. **Data Collection/Generation** — A labeled dataset of 5,000 emails was constructed
   with realistic spam and ham content, balanced classes, and metadata features.
2. **Exploratory Data Analysis (EDA)** — Distribution analysis, word frequency
   analysis, word clouds, and correlation analysis were performed to understand
   dataset characteristics before modeling.
3. **Text Preprocessing** — Lowercasing, punctuation/number removal, tokenization,
   stopword removal, and lemmatization/stemming were applied to normalize raw text.
4. **Feature Extraction** — Both CountVectorizer (Bag-of-Words) and TF-IDF were
   implemented and compared using a baseline Logistic Regression model to select the
   best-performing representation.
5. **Model Training** — Six classifiers (Naive Bayes, Logistic Regression, Decision
   Tree, Random Forest, SVM, KNN) were trained on an 80/20 stratified train-test split.
6. **Evaluation** — All models were compared using accuracy, precision, recall, and
   F1-score; the best model was further evaluated with a confusion matrix and ROC
   curve.
7. **Deployment** — The best model and its vectorizer were serialized with Joblib and
   integrated into a Streamlit web application for real-time predictions.

---

## 6. System Requirements

**Hardware:**
- Any modern PC/laptop (minimum 4GB RAM recommended)
- No GPU required (classical ML algorithms are CPU-efficient)

**Software:**
- Python 3.11 or later
- Operating System: Windows / Linux / macOS
- Required Python libraries: pandas, numpy, matplotlib, seaborn, scikit-learn, nltk,
  joblib, streamlit, wordcloud, jupyter (see `requirements.txt`)

---

## 7. Algorithms Used

| Algorithm | Key Idea |
|---|---|
| **Naive Bayes** | Applies Bayes' theorem assuming conditional independence between word features; extremely fast and well-suited to text classification. |
| **Logistic Regression** | Linear model estimating the probability of the "Spam" class via the logistic (sigmoid) function. |
| **Decision Tree** | Splits data recursively on feature thresholds to form a tree of decisions; interpretable but prone to overfitting. |
| **Random Forest** | An ensemble of many decision trees trained on random subsets of data/features, aggregated by majority vote to reduce overfitting. |
| **Support Vector Machine (SVM)** | Finds the optimal hyperplane that maximizes the margin between spam and ham classes in feature space. |
| **K-Nearest Neighbors (KNN)** | Classifies a new email based on the majority label among its "k" most similar emails in feature space. |

---

## 8. Implementation

The implementation is organized into modular Python scripts under `src/`:

- `preprocessing.py` — text cleaning pipeline (lowercase, punctuation/number removal,
  tokenization, stopword removal, stemming/lemmatization).
- `feature_extraction.py` — CountVectorizer and TF-IDF vectorization utilities.
- `visualization.py` — EDA plotting functions (distributions, word clouds, correlation,
  keyword frequency).
- `train_model.py` — full training pipeline: loads data, cleans text, compares feature
  extraction methods, trains all six models, selects and saves the best one.
- `evaluation.py` — detailed evaluation of the saved model (classification report,
  confusion matrix, ROC curve).
- `predict.py` — command-line prediction utility for custom email text.
- `app.py` — Streamlit web application providing an interactive UI for real-time spam
  classification with confidence scores.

All modules follow PEP-8 coding standards, use meaningful variable names and comments,
and include exception handling where appropriate.

---

## 9. Testing

The system was tested at multiple levels:

- **Unit-level testing:** Each module (`preprocessing.py`, `feature_extraction.py`,
  `predict.py`) includes a runnable `__main__` block demonstrating correct behavior on
  sample inputs.
- **Pipeline-level testing:** The full training pipeline (`train_model.py`) was executed
  end-to-end without errors, producing a saved model, vectorizer, and comparison report.
- **Notebook testing:** The complete Jupyter notebook was executed top-to-bottom without
  errors, confirming reproducibility of every step from data loading through evaluation.
- **Application testing:** The Streamlit application (`app.py`) was launched and verified
  to load the saved model correctly and produce predictions with confidence scores for
  both spam and legitimate sample emails.

---

## 10. Results

Six models were trained and evaluated on a held-out 20% test set (1,000 emails):

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Naive Bayes | 97.90% | 97.23% | 98.60% | 97.91% |
| Logistic Regression | 97.90% | 97.23% | 98.60% | 97.91% |
| Decision Tree | 95.40% | 95.03% | 95.79% | 95.41% |
| Random Forest | 97.30% | 96.83% | 97.80% | 97.31% |
| Support Vector Machine | 97.90% | 97.23% | 98.60% | 97.91% |
| K-Nearest Neighbors | 97.90% | 97.23% | 98.60% | 97.91% |

The **Naive Bayes** classifier was selected as the final model due to its top-tier
F1-score, extremely fast training/inference time, and simplicity — making it ideal for
a lightweight, deployable spam filter. The final model achieved an **ROC-AUC of 0.976**
on the test set, indicating excellent separability between spam and ham classes.

---

## 11. Advantages

- High accuracy (~98%) achieved using lightweight, classical ML algorithms.
- Fast training and near-instantaneous inference, suitable for real-time filtering.
- Fully modular, readable, and well-documented codebase.
- Interactive web application for non-technical end users.
- Reproducible pipeline — training, evaluation, and deployment can be re-run end-to-end.

## 12. Limitations

- The dataset is synthetically generated; performance on real-world, highly diverse
  spam campaigns may differ and would benefit from validation on a real corpus.
- The model relies on Bag-of-Words/TF-IDF features and does not capture deep semantic
  or contextual meaning the way transformer-based models (e.g. BERT) can.
- Spam tactics evolve continuously; periodic retraining would be required to maintain
  accuracy against new spam patterns in production.

## 13. Future Scope

- Train and validate on real-world datasets (e.g. Enron-Spam, SpamAssassin corpus).
- Explore deep learning architectures (LSTM, BERT) for improved contextual accuracy.
- Add model explainability (SHAP/LIME) so predictions can be justified to end users.
- Extend to multilingual spam detection.
- Deploy as a browser extension or email-client plugin for real-time inbox filtering.
- Host the Streamlit application on a cloud platform for public accessibility.

---

## 14. Conclusion

This project successfully demonstrates a complete, end-to-end machine learning pipeline
for email spam detection — from raw data through deployment. By comparing multiple
feature-extraction techniques and six classification algorithms, the project identifies
Naive Bayes as an effective, efficient solution achieving ~98% accuracy. The resulting
system, packaged with a Streamlit web interface, offers a practical, interpretable, and
readily deployable spam filter, while also serving as a strong demonstration of applied
NLP and machine learning skills suitable for academic and professional evaluation.

---

## 15. References

1. Scikit-learn Documentation — https://scikit-learn.org/stable/documentation.html
2. NLTK Documentation — https://www.nltk.org/
3. Streamlit Documentation — https://docs.streamlit.io/
4. Pandas Documentation — https://pandas.pydata.org/docs/
5. Cormack, G. V. (2008). "Email Spam Filtering: A Systematic Review." Foundations and
   Trends in Information Retrieval.
6. Sahami, M., Dumais, S., Heckerman, D., & Horvitz, E. (1998). "A Bayesian Approach to
   Filtering Junk E-Mail." AAAI Workshop on Learning for Text Categorization.
7. Metsis, V., Androutsopoulos, I., & Paliouras, G. (2006). "Spam Filtering with Naive
   Bayes -- Which Naive Bayes?" CEAS 2006.
8. Cortes, C., & Vapnik, V. (1995). "Support-Vector Networks." Machine Learning, 20(3).
