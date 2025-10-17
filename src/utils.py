import re
import string
from typing import List

PUNCTUATION = str.maketrans("", "", string.punctuation)

URL_RE = re.compile(r"https?://\S+|www\.\S+")
EMAIL_RE = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b")
NUM_RE = re.compile(r"\b\d+\b")

"""
This file has text cleaning and column detection helpers.
It helps clean spam/ham messages and find correct columns from CSV files.
"""

def clean_text(
    text: str,
    lowercase: bool = True,
    remove_urls: bool = True,
    remove_emails: bool = True,
    remove_numbers: bool = True,
    remove_punctuation: bool = True,
    strip_whitespace: bool = True,
) -> str:
    """
    Cleans the input text by removing unwanted parts like URLs or numbers.
    Also lowercases and trims spaces to make the text model-friendly.
    """
    if not isinstance(text, str):
        text = str(text)

    if lowercase:
        text = text.lower()
    if remove_urls:
        text = URL_RE.sub(" ", text)
    if remove_emails:
        text = EMAIL_RE.sub(" ", text)
    if remove_numbers:
        text = NUM_RE.sub(" ", text)
    if remove_punctuation:
        text = text.translate(PUNCTUATION)
    if strip_whitespace:
        text = " ".join(text.split())
    return text

def find_first_existing(cols: List[str], candidates: List[str]):
    """
    Looks for a column from a candidate list that exists in dataframe columns.
    Returns the first match or None if no match is found.
    """
    lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None