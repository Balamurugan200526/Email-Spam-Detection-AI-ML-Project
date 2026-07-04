# 📧 Email Spam Detection Using Machine Learning

An end-to-end, internship-quality machine learning project that classifies emails as
**Spam** or **Ham (legitimate)**, complete with EDA, NLP preprocessing, multi-model
comparison, evaluation, and a deployed Streamlit web application.

---

## 📌 Project Overview

Spam emails are unsolicited, often malicious or fraudulent messages that waste time,
compromise security, and clutter inboxes. This project builds a complete machine
learning pipeline — from raw text to a deployed web app — that automatically detects
spam emails using Natural Language Processing (NLP) and classical ML algorithms.

## 🎯 Objectives

- Build a machine learning model to classify emails as **Spam** or **Ham**.
- Implement a complete end-to-end ML pipeline: preprocessing → feature extraction →
  training → evaluation → deployment.
- Compare multiple algorithms and feature-extraction techniques to select the best
  performing combination.
- Deploy the final model as an interactive Streamlit web application.

## 🛠️ Technologies Used

| Category | Tools |
|---|---|
| Language | Python 3.11 |
| Data Handling | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, WordCloud |
| NLP | NLTK (stopwords, tokenization, stemming, lemmatization) |
| Machine Learning | Scikit-learn |
| Model Persistence | Joblib |
| Web App | Streamlit |
| Notebook | Jupyter Notebook |

## 📊 Dataset Information

- **File:** `dataset/spam_dataset.csv`
- **Records:** 5,000 emails (~50/50 Spam/Ham split)
- **Columns:**
  - `Email_ID` — unique identifier
  - `Email_Text` — raw email body text
  - `Sender` — sender's email address
  - `Subject` — email subject line
  - `Message_Length` — character count of the email
  - `Number_of_Links` — count of embedded URLs
  - `Number_of_Special_Characters` — count of symbols like `!`, `$`, `#`
  - `Number_of_Capital_Letters` — count of uppercase characters
  - `Label` — target class (`Spam` / `Ham`)

The dataset was synthetically generated to closely mimic real-world spam and legitimate
email patterns, including realistic vocabulary overlap and ~3% label noise, so that model
performance reflects a genuine (not trivially perfect) classification problem. See
`dataset/generate_dataset.py` for full generation logic.

## 📂 Folder Structure

```
Email_Spam_Detection/
│
├── dataset/
│   ├── spam_dataset.csv
│   └── generate_dataset.py
│
├── notebooks/
│   └── Email_Spam_Detection.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── feature_extraction.py
│   ├── train_model.py
│   ├── evaluation.py
│   ├── predict.py
│   └── visualization.py
│
├── models/
│   ├── spam_classifier.pkl
│   └── vectorizer.pkl
│
├── outputs/
│   ├── graphs/
│   ├── reports/
│   └── prediction_results.csv
│
├── app.py
├── requirements.txt
├── README.md
└── report_content.md
```

## ⚙️ Installation

```bash
# 1. Clone or download the project folder
cd Email_Spam_Detection

# 2. (Recommended) create a virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download required NLTK data (one-time)
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('wordnet'); nltk.download('omw-1.4')"
```

## ▶️ Running Instructions

**1. (Optional) Regenerate the dataset**
```bash
cd dataset
python generate_dataset.py
```

**2. Train the model** (cleans text, compares features, trains & evaluates 6 models, saves the best one)
```bash
cd src
python train_model.py
```

**3. Run detailed evaluation** (confusion matrix, classification report, ROC curve)
```bash
python evaluation.py
```

**4. Generate EDA visualizations**
```bash
python visualization.py
```

**5. Try predictions from the command line**
```bash
python predict.py "Congratulations! You have won a free prize, click here now!"
# or run interactively:
python predict.py
```

**6. Launch the Streamlit web app**
```bash
cd ..                     # back to project root
streamlit run app.py
```

**7. Explore the full pipeline in Jupyter**
```bash
jupyter notebook notebooks/Email_Spam_Detection.ipynb
```

## 📈 Results

Six classifiers were trained and compared using CountVectorizer/TF-IDF features:

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Naive Bayes | 97.90% | 97.23% | 98.60% | 97.91% |
| Logistic Regression | 97.90% | 97.23% | 98.60% | 97.91% |
| Decision Tree | 95.40% | 95.03% | 95.79% | 95.41% |
| Random Forest | 97.30% | 96.83% | 97.80% | 97.31% |
| Support Vector Machine | 97.90% | 97.23% | 98.60% | 97.91% |
| K-Nearest Neighbors | 97.90% | 97.23% | 98.60% | 97.91% |

**Best Model:** Naive Bayes (selected for its top F1-score, fast training/inference, and
strong fit for high-dimensional sparse text data).

Full metrics, confusion matrix, and ROC curve are available in `outputs/reports/` and
`outputs/graphs/`.

## 🔮 Future Scope

- Incorporate deep learning approaches (LSTM, BERT) for improved contextual understanding.
- Expand the dataset with real-world labeled email corpora (e.g. Enron, SpamAssassin).
- Add multilingual spam detection support.
- Integrate the model as a browser extension or email-client plugin for real-time filtering.
- Add explainability (e.g. SHAP/LIME) to show *why* an email was flagged as spam.
- Deploy the Streamlit app to the cloud (Streamlit Community Cloud / GCP / AWS) for public access.

## 👤 Author

Balamurugan S — B.Tech Information Technology, A.V.C College of Engineering
GitHub: [github.com/Balamurugan200526](https://github.com/Balamurugan200526)
LinkedIn: [linkedin.com/in/balamurugan-s-468387337](https://linkedin.com/in/balamurugan-s-468387337)

---
*Developed as part of a 1-month AI & Machine Learning Internship project.*
