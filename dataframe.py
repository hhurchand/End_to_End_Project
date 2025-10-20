
"""
LOADS AND DISPLAYS THE DATAFRAME FOR EXPLORATION OF
THE CLEAN DATASET.
"""
import pandas as pd

def show_dataframe():
    # CLEAN DATA
    df = pd.read_csv("data/processed/CLEAN_EMAILS.csv")

    print("\nCLEAN DATAFRAME SAMPLE\n")
    print(df.head(10))

    # VERIFY COLUMN NAMES
    print("\nCOLUMNS:", list(df.columns))

    # INFORMATION
    print("\nINFORMATION:")
    print(df.info())

    # SHAPE
    print("\nSHAPE:", df.shape)

    # VERIFY IF ANY NULLS EXIST
    print("\nMISSING VALUES:")
    print(df.isnull().sum())

    # UNIQUE CHARACTERS
    text_col = "email"
    sample_text = " ".join(df[text_col].head(50).astype(str))
    specials = set([ch for ch in sample_text if not ch.isalnum() and not ch.isspace()])
    print("\nSPECIAL CHARACTERS:", specials)

if __name__ == "__main__":
    show_dataframe()
