import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report

# Load test data
test_df = pd.read_csv("data/processed/test.csv")
X_test_text = test_df["text"].fillna("").astype(str)
y_test = test_df["label"]

# Load vectorizer + models
vectorizer = joblib.load("models/vectorizer.joblib")
X_test_tfidf = vectorizer.transform(X_test_text)

# Evaluate models
for name in ["Logistic_Regression", "Random_Forest", "Multinomial_NB"]:
    model = joblib.load(f"models/{name}.joblib")
    preds = model.predict(X_test_tfidf)
    print(f"\n=== {name.replace('_', ' ')} (Test) ===")
    print(f"Accuracy: {accuracy_score(y_test, preds):.4f}")
    print(classification_report(y_test, preds))
