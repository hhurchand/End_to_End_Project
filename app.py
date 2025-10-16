import streamlit as st
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from typing import Any

from yaml import safe_load
from constants import PARAMS

with open(PARAMS, encoding="utf-8") as f:
    config = safe_load(f)


def load_model(model_name):
    """
    Loads a machine learning model from the 'models' directory based on the model name provided.

    Parameters:
        model_name (str): The name of the model file (without '.pkl' extension)

    Returns:
        model: A deserialized machine learning model object
    """
    model_file = os.path.join("models", f"{model_name}.pkl")
    with open(model_file, "rb") as f:
        model = pickle.load(f)
    return model


def load_tfidf():
    """
    Loads the pre-trained TF-IDF vectorizer used during model training.

    Returns:
        vectorizer: A deserialized TfidfVectorizer object
    """
    tfidf = os.path.join("models", "tfidf_vectorize.pkl")
    with open(tfidf, "rb") as f:
        vectorize = pickle.load(f)
    return vectorize



def process_text(text:str, tfidf_vectorize:TfidfVectorizer):
    """
    Transforms input text into TF-IDF features using the provided vectorizer.

    Parameters:
        text (str): Input message to classify
        vectorizer: TF-IDF vectorizer object

    Returns:
        sparse matrix: Transformed TF-IDF feature vector
    """
    return tfidf_vectorize.transform([text])


def main():
    """
    Main function to run the Streamlit Spam Detection app.
    Allows user input, model selection, and displays prediction.
    """
    st.title("Spam Detector")
    st.write("Type a message and choose a model to see if it's Spam or Ham.")

    text = st.text_input("Enter your message:")

    model_option = st.selectbox(
        "Choose a model:",
        ["RandomForestClassifier", "MultinomialNB", "LogisticRegression"]
    )

    if st.button("Predict"):
        if text.strip() == "":
            st.warning("Please type a message.")
        else:
            vectorizer= load_tfidf()
            model = load_model(model_option)
            text_transformed = process_text(text, vectorizer)
            prediction = model.predict(text_transformed)[0]
            if prediction == 1:
                st.error("Result: Spam :x:")
            else:
                st.success("Result: Ham :white_check_mark:")

if __name__ == "__main__":
    main()
