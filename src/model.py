import pickle
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

test_df = pd.read_csv("data/processed/test.csv")
X_test_text = test_df["text"].fillna("").astype(str)
y_test = test_df["label"]

with open("models/tfidf_vectorizer.pickle", "rb") as f:
    vectorizer = pickle.load(f)

X_test_tfidf = vectorizer.transform(X_test_text)

for name in ["logistic_regression", "random_forest", "multinomial_nb"]:
    with open(f"models/{name}.pickle", "rb") as f:
        model = pickle.load(f)
    preds = model.predict(X_test_tfidf)
    print(f"\n=== {name.replace('_', ' ')} (Test) ===")
    print(f"Accuracy: {accuracy_score(y_test, preds):.4f}")
    print(classification_report(y_test, preds, digits=2, zero_division=0))
