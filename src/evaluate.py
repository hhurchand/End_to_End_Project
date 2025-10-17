import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report

# Load and prepare data
df = pd.read_csv("data/processed/train.csv")
X = df["text"].fillna("").astype(str)
y = df["label"]

# Vectorize text
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2), stop_words="english")
X_tfidf = vectorizer.fit_transform(X)

# Models
models = {
    "Logistic_Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random_Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Multinomial_NB": MultinomialNB()
}

# Train + evaluate + save
for name, model in models.items():
    model.fit(X_tfidf, y)
    preds = model.predict(X_tfidf)
    print(f"\n=== {name} ===")
    print(f"Accuracy: {accuracy_score(y, preds):.4f}")
    print(classification_report(y, preds))
    joblib.dump(model, f"models/{name}.joblib")

joblib.dump(vectorizer, "models/vectorizer.joblib")
