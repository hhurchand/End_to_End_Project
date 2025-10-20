import pandas as pd
import re, string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

def preprocess():
    df = pd.read_csv("data/raw/spam_Emails_data.csv").dropna(subset=["text", "label"]).drop_duplicates()

    label_map = {"Spam": 1, "Ham": 0}
    df["label"] = df["label"].map(label_map)

    def clean_text(t):
        t = t.lower()
        t = re.sub(r"http\S+", " ", t)
        t = re.sub(r"\d+", " ", t)
        t = t.translate(str.maketrans("", "", string.punctuation))
        t = re.sub(r"\s+", " ", t).strip()
        return t

    df["text"] = df["text"].astype(str).apply(clean_text)

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    vectorizer = TfidfVectorizer(max_features=3000,ngram_range=(1, 3),
        sublinear_tf=True,min_df=3,max_df=0.85,stop_words="english")
    
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    train_df = pd.DataFrame(X_train.toarray(), columns=vectorizer.get_feature_names_out())
    train_df["label"] = y_train.values

    test_df = pd.DataFrame(X_test.toarray(), columns=vectorizer.get_feature_names_out())
    test_df["label"] = y_test.values

    train_df.to_csv("data/processed/train.csv", index=False)
    test_df.to_csv("data/processed/test.csv", index=False)

if __name__ == "__main__":
    preprocess()
