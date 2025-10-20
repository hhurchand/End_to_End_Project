# app/streamlit_app.py
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# --- Page Config ---
st.set_page_config(page_title="Spam or Ham Checker", page_icon="📩", layout="centered")

# --- Custom CSS ---
st.markdown("""
    <style>
        .main-title {
            text-align: center;
            font-size: 2.5rem;
            color: #4CAF50;
            margin-bottom: 0.5rem;
        }
        .subtext {
            text-align: center;
            color: #888;
            font-size: 1rem;
            margin-bottom: 2rem;
        }
        .result-box {
            border-radius: 20px;
            padding: 2rem;
            text-align: center;
            font-size: 2rem;
            font-weight: bold;
            animation: fadeIn 0.5s ease-in-out;
        }
        .ham {
            background: linear-gradient(135deg, #D4EDDA, #A3E4D7);
            color: #155724;
        }
        .spam {
            background: linear-gradient(135deg, #F8D7DA, #F5B7B1);
            color: #721C24;
        }
        .shake {
            animation: shake 0.4s ease-in-out;
        }
        @keyframes shake {
            0% { transform: translateX(0); }
            25% { transform: translateX(-8px); }
            50% { transform: translateX(8px); }
            75% { transform: translateX(-8px); }
            100% { transform: translateX(0); }
        }
        @keyframes fadeIn {
            from {opacity: 0;}
            to {opacity: 1;}
        }
    </style>
""", unsafe_allow_html=True)

# --- Load CSV ---
csv_path = "data/raw/spam_Emails_data.csv"
df = pd.read_csv(csv_path)
df['text'] = df['text'].fillna("").str.strip()
df['label'] = df['label'].str.lower()

# --- Train Simple Model ---
vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(df["text"])
y = df["label"]
model = MultinomialNB()
model.fit(X, y)

# --- UI ---
st.markdown("<div class='main-title'>📩 Spam or Ham Checker</div>", unsafe_allow_html=True)
st.markdown("<div class='subtext'>Type a message below to check if it’s in the dataset or let our model predict it!</div>", unsafe_allow_html=True)

input_text = st.text_area("💬 Enter your message:")

if st.button("🚀 Check Message"):
    if input_text.strip() == "":
        st.warning("⚠️ Please enter a message.")
    else:
        match = df[df['text'] == input_text.strip()]
        if not match.empty:
            label = match['label'].values[0]
            if label == "ham":
                st.markdown("<div class='result-box ham shake'>✅ HAM — This message looks safe!</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='result-box spam shake'>🚫 SPAM — Suspicious message detected!</div>", unsafe_allow_html=True)
        else:
            # --- Predict using model ---
            X_input = vectorizer.transform([input_text])
            prediction = model.predict(X_input)[0]
            if prediction == "ham":
                st.markdown("<div class='result-box ham shake'>✅ HAM (Predicted) — Looks like a safe message!</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='result-box spam shake'>🚫 SPAM (Predicted) — This seems suspicious!</div>", unsafe_allow_html=True)
