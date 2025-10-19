import pandas as pd

def clean_and_save():
    """Load raw EMAILS.csv, clean it, and save CLEAN_EMAILS.csv."""
    df = pd.read_csv("data/raw/EMAILS.csv")

    # simple clean
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    df = df.drop_duplicates()

    text_cols = df.select_dtypes(include="object").columns
    for c in text_cols:
        df[c] = df[c].str.strip()
    df[text_cols] = df[text_cols].replace(["", "na", "n/a", "null", "none", "-", "--"], pd.NA)

    num_cols = df.select_dtypes(include="number").columns
    df[text_cols] = df[text_cols].fillna("")
    df[num_cols] = df[num_cols].fillna(0)

    df = df[df["label"].isin([0, 1])].copy()
    df["label"] = df["label"].astype(int)

    df.to_csv("data/processed/CLEAN_EMAILS.csv", index=False)
    print("✅ Saved: data/processed/CLEAN_EMAILS.csv")
