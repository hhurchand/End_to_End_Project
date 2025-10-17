from pathlib import Path
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from src.data import load_config, load_dataset
from src.utils import clean_text, find_first_existing

# ------------------ PAGE SETUP ------------------
st.set_page_config(page_title="SPAM OR HAM DETECTOR", page_icon="📧", layout="centered")

# --- Custom CSS for blue theme + rounded components ---
st.markdown(
    """
    <style>
        /* Overall background */
        [data-testid="stAppViewContainer"] {
            background-color: #e8f2ff;
        }
        [data-testid="stHeader"], [data-testid="stToolbar"] {
            background-color: #e8f2ff;
        }
        [data-testid="stSidebar"] {
            background-color: #e8f2ff;
        }

        /* Titles and headings */
        h1, h2, h3 {
            color: #003366;
        }

        /* Text input & text area */
        textarea, input[type="text"] {
            background-color: white !important;
            border-radius: 12px !important;
            border: 1px solid #aac7ff !important;
            padding: 8px !important;
        }

        /* Buttons */
        button[kind="primary"], div.stButton > button {
            background-color: #4a90e2;
            color: white;
            border-radius: 10px;
            border: none;
            padding: 0.5rem 1rem;
            font-weight: 600;
        }
        button[kind="primary"]:hover, div.stButton > button:hover {
            background-color: #3a7bd5;
            color: white;
        }

        /* Success messages */
        .stAlert {
            border-radius: 10px;
        }

        /* Dataframes */
        [data-testid="stDataFrame"] {
            border-radius: 10px;
            background-color: white;
        }

        /* File uploader */
        [data-testid="stFileUploader"] section {
            background-color: #f8fbff;
            border-radius: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------ APP HEADER ------------------
st.title("📧 SPAM OR HAM")
st.caption("TF-IDF + classic ML (LogReg • Naive Bayes • Random Forest)")

# ------------------ LOAD CONFIG & DATA ------------------
ROOT = Path(__file__).resolve().parent
CFG = load_config(ROOT / "params.yaml")

df, text_col, label_col = load_dataset(CFG, ROOT)
df = df.dropna(subset=[text_col, label_col]).reset_index(drop=True)
X = df[text_col].astype(str).tolist()
y = df[label_col].astype(int).values

# ------------------ SIDEBAR SETTINGS ------------------
st.sidebar.header("⚙️")
model_choice = st.sidebar.selectbox(
    "Choose model:",
    ["Logistic Regression", "Naive Bayes", "Random Forest"]
)

# ------------------ TRAIN MODEL ------------------
vec = TfidfVectorizer(ngram_range=(1, 2), max_features=20000, strip_accents="unicode")
X_tf = vec.fit_transform(X)

if model_choice == "Logistic Regression":
    model = LogisticRegression(max_iter=1000, class_weight="balanced", C=0.5, random_state=123)
elif model_choice == "Naive Bayes":
    model = MultinomialNB(alpha=1.0)
else:
    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)

model.fit(X_tf, y)
st.sidebar.success(f"✅ Model trained: {model_choice}")

# ------------------ SINGLE EMAIL PREDICTION ------------------
st.subheader("Enter an Email")
txt = st.text_area("Enter a message:", height=140, placeholder="Type an email here...")

if st.button("Predict"):
    if not txt.strip():
        st.warning("Please enter a message.")
    else:
        X1 = vec.transform([clean_text(txt)])
        pred = int(model.predict(X1)[0])
        st.success("Prediction: **SPAM (1)** 🚩" if pred == 1 else "Prediction: **HAM (0)** ✅")

# ------------------ CSV UPLOAD ------------------
st.subheader("Load a Dataset")
up = st.file_uploader("Upload CSV", type=["csv"])

if up is not None:
    dfu = pd.read_csv(up, encoding="utf-8", engine="python")
    col = find_first_existing(dfu.columns.tolist(), CFG.get("dataset", {}).get("text_columns", []))
    if col is None:
        obj_cols = dfu.select_dtypes(include=["object"]).columns.tolist()
        col = obj_cols[0] if obj_cols else None
    if col is None:
        st.error("No text column found.")
    else:
        dfu[col] = dfu[col].astype(str).apply(clean_text)
        Xu = vec.transform(dfu[col])
        preds = model.predict(Xu).astype(int)
        dfu["predicted_label"] = preds
        st.dataframe(dfu.head(20), use_container_width=True)
        st.download_button(
            "⬇️ Download predictions",
            dfu.to_csv(index=False).encode("utf-8"),
            "predictions.csv",
            "text/csv",
        )
