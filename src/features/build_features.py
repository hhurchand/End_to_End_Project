import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

def load_split_vectorize():
    """
    Load CLEAN_EMAILS.csv, split into train/test,
    and apply CountVectorizer to convert text into features.
    Returns: X_train, X_test, y_train, y_test, Xtr, Xte, vectorizer
    """
    #Loading cleaned data
    df = pd.read_csv("data/processed/CLEAN_EMAILS.csv")

    # Separate features/labels
    X = df["email"].astype(str).values
    y = df["label"].astype(int).values

    # Split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Vectorizer
    vectorizer = CountVectorizer()
    Xtr = vectorizer.fit_transform(X_train)
    Xte = vectorizer.transform(X_test)

    # Return all
    return X_train, X_test, y_train, y_test, Xtr, Xte, vectorizer
