import warnings, pickle
warnings.filterwarnings("ignore")
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("data/processed/train.csv")
X = df["text"].fillna("").astype(str)
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 3), stop_words="english")
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

models = {
    "logistic_regression": LogisticRegression(max_iter=2000, random_state=42),
    "random_forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "multinomial_nb": MultinomialNB()
}

for name, model in models.items():
    model.fit(X_train_tfidf, y_train)
    preds = model.predict(X_test_tfidf)

    print(f"\n=== {name.upper()} (Train) ===")
    print(f"Accuracy: {accuracy_score(y_test, preds):.4f}")
    print(classification_report(y_test, preds, digits=2, zero_division=0))

    model_path = f"models/{name}.pickle"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

with open("models/tfidf_vectorizer.pickle", "wb") as f:
    pickle.dump(vectorizer, f)