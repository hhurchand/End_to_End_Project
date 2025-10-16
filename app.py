import streamlit as st
import re
import string
import pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from tensorflow import keras

st.title("Spam Email Checker")

# Load model
model = keras.models.load_model('models\dl_model.keras')
with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)

# Transform functions
def transform_text(text):
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    text = text.strip()
    for punc in string.punctuation:
        text = text.replace(punc, "")
    return text

def tokenize_text(text):
    tokens = word_tokenize(text)
    english_stop_words = stopwords.words("english")
    tokens = [word for word in tokens if word not in english_stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)

# Text input
email_text = st.text_area("Enter email text:")

# Predict button
if st.button("Check Email"):
    if email_text.strip():
        # Transform and tokenize
        transformed = transform_text(email_text)
        tokenized = tokenize_text(transformed)
        
        # Vectorize
        X = tfidf_vectorizer.transform([tokenized])
        
        # Predict
        prediction = model.predict(X.toarray(), verbose=0)
        spam_prob = prediction[0][0]
        
        # Show result
        if spam_prob > 0.5:
            st.error(f"SPAM : Confidence: {spam_prob*100:.2f}%")
        else:
            st.success(f"HAM : Confidence: {(1-spam_prob)*100:.2f}%")
    else:
        st.warning("Please enter some text")