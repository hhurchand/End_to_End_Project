import streamlit as st
from pathlib import Path
import pandas as pd
import random
import sys
from src.models.predict import predict_label

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
DATA = ROOT / "data" / "merged_spam.csv"

st.set_page_config(page_title="Email Spam Verifier", page_icon="📧", layout="centered")
st.title("📧 Email Spam Verifier")

@st.cache_data
def load_sample(df_path: Path):
    df = pd.read_csv(df_path)
    text_col = "text" if "text" in df.columns else df.columns[0]
    label_col = "target" if "target" in df.columns else df.columns[-1]
    return df.rename(columns={text_col: "text", label_col: "label"})

df = load_sample(DATA)
col1, col2 = st.columns(2)
with col1:
    model_name = st.selectbox("Select model", ["rf", "linearsvc", "logreg"])
with col2:
    sample_type = st.radio("Choose:", ["Random Spam", "Random Ham", "Manual Entry"], horizontal=True)

prefill = ""
if sample_type != "Manual Entry":
    df["y"] = df["label"].astype(str).str.lower().apply(lambda v: 1 if "spam" in v else 0)
    pool = df[df["y"] == (1 if sample_type == "Random Spam" else 0)]
    if not pool.empty:
        prefill = pool["text"].sample(1, random_state=random.randint(0, 9999)).iloc[0]

st.subheader("✉️ Enter or edit email content")
user_text = st.text_area("Email text:", value=prefill, height=200)

if st.button("🔍 Predict"):
    if not user_text.strip():
        st.warning("Please enter text first.")
    else:
        label, proba = predict_label(user_text, model_name)
        st.subheader(f"Prediction: **{label}**")
        if proba is not None:
            st.caption(f"Confidence (Spam probability): {proba:.3f}")
        if label == "Spam":
            st.error("🚨 This looks like SPAM.")
        else:
            st.success("✅ This looks like HAM.")
