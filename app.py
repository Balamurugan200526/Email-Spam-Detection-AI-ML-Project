"""
app.py
--------
Streamlit web application for the Email Spam Detection project.

Run with:
    streamlit run app.py
"""

import os
import json

import joblib
import streamlit as st
import pandas as pd

from src.preprocessing import clean_text
from src.pdf_utils import extract_text_from_pdf

BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "models")
REPORT_DIR = os.path.join(BASE_DIR, "outputs", "reports")

st.set_page_config(
    page_title="Email Spam Detector",
    page_icon="📧",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS for a polished, modern UI
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(90deg, #4e54c8, #8f94fb);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .sub-title {
        text-align: center;
        color: #6c757d;
        font-size: 1.05rem;
        margin-top: 0;
        margin-bottom: 1.5rem;
    }
    .result-spam {
        background-color: #ffe5e5;
        border-left: 6px solid #e74c3c;
        padding: 1.2rem;
        border-radius: 10px;
        font-size: 1.3rem;
        font-weight: 700;
        color: #c0392b;
        text-align: center;
    }
    .result-ham {
        background-color: #e6f9ee;
        border-left: 6px solid #2ecc71;
        padding: 1.2rem;
        border-radius: 10px;
        font-size: 1.3rem;
        font-weight: 700;
        color: #1e8449;
        text-align: center;
    }
    .metric-card {
        background-color: #f5f6fa;
        border-radius: 10px;
        padding: 0.8rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model_artifacts():
    model = joblib.load(os.path.join(MODEL_DIR, "spam_classifier.pkl"))
    vectorizer = joblib.load(os.path.join(MODEL_DIR, "vectorizer.pkl"))
    return model, vectorizer


@st.cache_data
def load_comparison_report():
    path = os.path.join(REPORT_DIR, "model_comparison.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def predict(text, model, vectorizer):
    cleaned = clean_text(text)
    features = vectorizer.transform([cleaned])
    pred = model.predict(features)[0]
    label = "Spam" if pred == 1 else "Ham"

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(features)[0]
        ham_prob, spam_prob = proba[0] * 100, proba[1] * 100
    else:
        spam_prob = float(pred) * 100
        ham_prob = 100 - spam_prob

    confidence = spam_prob if label == "Spam" else ham_prob
    return label, confidence, spam_prob, ham_prob, cleaned


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("ℹ️ About This Project")
    st.write(
        "**Email Spam Detection Using Machine Learning** is an end-to-end "
        "NLP pipeline that classifies emails as **Spam** or **Ham** "
        "(legitimate) using classic ML algorithms."
    )
    st.markdown("---")
    st.subheader("🛠️ Technologies")
    st.write("Python · Scikit-learn · NLTK · Pandas · Streamlit")

    report = load_comparison_report()
    if report:
        st.markdown("---")
        st.subheader("🏆 Best Model")
        best = report["best_model"]
        best_metrics = report["model_comparison"][best]
        st.write(f"**{best}**")
        st.write(f"Accuracy: **{best_metrics['Accuracy']*100:.2f}%**")
        st.write(f"F1-Score: **{best_metrics['F1-Score']*100:.2f}%**")

    st.markdown("---")
    st.caption("Built as part of a 1-month AI & ML Internship project.")


# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------
st.markdown('<p class="main-title">📧 Email Spam Detector</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Paste an email or upload a PDF/TXT file to check whether it is Spam or Ham</p>',
            unsafe_allow_html=True)

try:
    model, vectorizer = load_model_artifacts()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"⚠️ Could not load the trained model. Please run `python src/train_model.py` "
             f"first to generate models/spam_classifier.pkl.\n\nDetails: {e}")

tab_paste, tab_upload = st.tabs(["✍️ Paste Text", "📎 Upload File"])

email_text = ""
extracted_from_file = False

with tab_paste:
    pasted_text = st.text_area(
        "✉️ Email Text",
        height=180,
        placeholder="Paste or type the email content here...\n\n"
                    "Example: 'Congratulations! You have won a free prize, click here to claim now!!!'",
        key="pasted_text",
    )

with tab_upload:
    uploaded_file = st.file_uploader(
        "Upload an email as a PDF or TXT file",
        type=["pdf", "txt"],
        help="Upload a PDF (e.g. an email exported/printed to PDF) or a plain text file.",
    )
    if uploaded_file is not None:
        try:
            if uploaded_file.type == "application/pdf" or uploaded_file.name.lower().endswith(".pdf"):
                extracted_text = extract_text_from_pdf(uploaded_file)
            else:
                extracted_text = uploaded_file.read().decode("utf-8", errors="ignore")

            if not extracted_text.strip():
                st.warning("⚠️ No extractable text was found in this file. If it's a "
                           "scanned/image-only PDF (no text layer), text extraction "
                           "won't work -- try a text-based PDF or paste the content manually.")
            else:
                st.success(f"✅ Extracted {len(extracted_text)} characters from "
                           f"**{uploaded_file.name}**.")
                with st.expander("📄 Preview extracted text"):
                    st.text(extracted_text[:3000] + ("..." if len(extracted_text) > 3000 else ""))
                email_text = extracted_text
                extracted_from_file = True
        except Exception as e:
            st.error(f"⚠️ Could not read this file: {e}")

# If the user has pasted text, prefer that (it's what's visibly in front of them)
# unless they're actively using the uploaded file with no pasted text.
if pasted_text.strip():
    email_text = pasted_text

col1, col2 = st.columns([1, 1])
with col1:
    predict_clicked = st.button("🔍 Predict", use_container_width=True, type="primary")
with col2:
    clear_clicked = st.button("🗑️ Clear", use_container_width=True)

if clear_clicked:
    st.rerun()

if predict_clicked:
    if not model_loaded:
        st.warning("Model not available.")
    elif not email_text.strip():
        st.warning("⚠️ Please paste email text or upload a file before predicting.")
    else:
        label, confidence, spam_prob, ham_prob, cleaned = predict(email_text, model, vectorizer)

        if label == "Spam":
            st.markdown(f'<div class="result-spam">🚨 This email is classified as: SPAM</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="result-ham">✅ This email is classified as: HAM (Legitimate)</div>',
                        unsafe_allow_html=True)

        st.write("")
        m1, m2, m3 = st.columns(3)
        m1.metric("Confidence", f"{confidence:.2f}%")
        m2.metric("Spam Probability", f"{spam_prob:.2f}%")
        m3.metric("Ham Probability", f"{ham_prob:.2f}%")

        st.progress(int(spam_prob))
        st.caption(f"Spam likelihood: {spam_prob:.2f}%")

        with st.expander("🔬 View cleaned/preprocessed text (used by the model)"):
            st.code(cleaned if cleaned else "(empty after cleaning)")

st.markdown("---")

if report:
    st.subheader("📊 Model Performance Comparison")
    comp_df = pd.DataFrame(report["model_comparison"]).T
    comp_df = comp_df[["Accuracy", "Precision", "Recall", "F1-Score"]]
    comp_df = (comp_df * 100).round(2)
    st.dataframe(comp_df.style.highlight_max(color="#d4f7dc", axis=0), use_container_width=True)
    st.caption(f"Feature extraction method used: **{report['best_feature_method'].upper()}** "
               f"| Best model: **{report['best_model']}**")

st.markdown("---")
st.caption("Email Spam Detection Using Machine Learning — Internship Project")