import pickle

def predict_text(text):
    """Load saved model/vectorizer and predict a new message."""
    with open("final_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    X = vectorizer.transform([text])
    pred = model.predict(X)[0]
    return "SPAM" if pred == 1 else "HAM"
