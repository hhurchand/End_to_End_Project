import pandas as pd

def load_clean():
    df = pd.read_csv("data/processed/CLEAN_EMAILS.csv")
    X = df["email"].astype(str).values
    y = df["label"].astype(int).values
    return X, y
