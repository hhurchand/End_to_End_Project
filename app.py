# app/streamlit_app.py
import streamlit as st
import pandas as pd

# Load CSV
csv_path = "data/raw/spam_Emails_data.csv"
df = pd.read_csv(csv_path)
df['text'] = df['text'].fillna("").str.strip()  # clean up whitespace

st.title("📩 Spam or Ham Checker")
st.write("Type a message below, and we'll check if it matches any messages in the dataset.")

# Input from user
input_text = st.text_area("Enter message text:")

if st.button("Check Message"):
    if input_text.strip() == "":
        st.warning("Please enter a message.")
    else:
        # Exact match lookup
        match = df[df['text'] == input_text.strip()]
        if not match.empty:
            label = match['label'].values[0]
            emoji = "✅ Ham" if label.lower() == "ham" else "🚫 Spam"
            st.success(f"Message found in dataset: {emoji}")
        else:
            st.info("Message not found in the dataset.")
