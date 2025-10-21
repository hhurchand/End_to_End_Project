import streamlit as st
from pathlib import Path
import pandas as pd
import random
import sys

# Make sure we can import src.*
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.models.predict import predict_label

DATA = ROOT / "data" / "merged_spam.csv"

st.set_page_config(page_title="Email Spam Verifier", page_icon="📧", layout="centered")

st.title("📧 Email Spam Verifier")
st.caption("LogReg • LinearSVC • RandomForest (MLflow-trained)")

# Load a small sample of the merged data for random suggestions
@st.cache_data
def load_sample(df_path: Path, n: int = 2000):
    df = pd.read_csv(df_path)
    # Try to auto-detect columns (same logic as main.py)
    text_col = "text" if "text" in df.columns else df.select_dtypes("object").columns[0]
    label_col = "target" if "target" in df.columns else df.columns[-1]
    return df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label"})

if not DATA.exists():
    st.error("Merged dataset not found. Run `python main.py` first to create `data/merged_spam.csv`.")
    st.stop()

df = load_sample(DATA)

# Convert label to binary-ish string for convenience
def label_to_int(v):
    s = str(v).strip().lower()
    if s in {"1", "spam", "true", "yes"} or "spam" in s:
        return 1
    return 0

df["y"] = df["label"].map(label_to_int)

col1, col2 = st.columns(2)
with col1:
    model_name = st.selectbox("Select model", options=["rf", "linearsvc", "logreg"], index=0,
                              help="Pick the trained model artifact to use.")
with col2:
    sample_type = st.radio("Sample", ["Random Spam", "Random Ham"], horizontal=True)

# Prefill text from a random sample
if sample_type == "Random Spam":
    pool = df[df["y"] == 1]
else:
    pool = df[df["y"] == 0]

prefill = pool["text"].sample(1, random_state=random.randint(0, 10_000)).iloc[0] if not pool.empty else ""
user_text = st.text_area("Enter email text", value=prefill, height=200)

if st.button("🔍 Predict"):
    if not user_text.strip():
        st.warning("Please enter some text first.")
    else:
        label, proba = predict_label(user_text, model_name=model_name)
        st.subheader(f"Prediction: **{label}**")
        if proba is not None:
            st.write(f"Confidence (Spam probability): **{proba:.3f}**")
        # A little UX
        if label == "Spam":
            st.error("This looks like spam.")
        else:
            st.success("This looks like ham (not spam).")

st.markdown("---")
st.caption("Tip: retrain with `python main.py` to refresh the models. Then reload this page.")
