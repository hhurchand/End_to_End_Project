import pandas as pd
from src.utils.logger import logger


def the_clean(df: pd.DataFrame, parameter: dict) -> pd.DataFrame:
    """
    CLEAN EMAIL TEXT BASED ON PARAMS

    STEPS:
    1) KEEP TEXT/LABEL COLUMNS
    2) DROP DUPLICATES
    3) PLACEHOLDERS FOR NAN
    4) STRIP WHITESPACE
    5) UPPERCASE
    6) REMOVE UNDERSCORE SEPARATORS
    7) DROP ROWS WITH EMPTY TEXT
    8) FILL REMAINING NAN WITH EMPTY STRING

    ARGS:
        df: RAW DATAFRAME
        parameter: PARAMS DICTIONARY

    RETURNS:
        CLEAN DATAFRAME WITH TEXT AND LABEL
    """

    # GET COLUMNS FROM PARAMS
    text_column = parameter["text"]
    label_column = parameter["label"]

    # STANDARDIZE LABELS TO NUMERIC | HAM = 0, SPAM = 1
    if df[label_column].dtype == "object":
        df[label_column] = (
            df[label_column]
            .astype(str)
            .str.lower()
            .map({"ham": 0, "spam": 1})
            .fillna(df[label_column]))


    # KEEP ONLY NEEDED COLUMNS
    df = df[[text_column, label_column]].copy()


    # REMOVE DUPLICATES
    if parameter["clean"]["remove_duplicates"]:
        df = df.drop_duplicates()


    # PLACEHOLDERS TO CONVERT TO NaN
    placeholders = {str(x).lower() for x in parameter["clean"]["placeholders"]}
    df[text_column] = df[text_column].apply(lambda x: None if isinstance(x, str) and x.strip().lower() in placeholders else x)

    # REMOVE WHITESPACES
    if parameter["clean"]["remove_whitespaces"]:
        df[text_column] = (df[text_column].astype("string").str.strip().str.replace(r"\s+", " ", regex=True))


    # UPPERCASE
    if parameter["clean"]["uppercase"]:
        df[text_column] = df[text_column].astype("string").str.upper()

    # REMOVE UNDERSCORES
    df[text_column] = df[text_column].str.replace(r"_+", " ", regex=True)

    # DROP ROWS WHERE TEXT IS EMPTY
    df = df.dropna(subset=[text_column])
    df = df[df[text_column].str.strip() != ""]

    # FILL ANY NAN WITH EMPTY STRING
    df[text_column] = df[text_column].fillna("")

    return df
