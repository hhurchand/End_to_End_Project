from pathlib import Path
from typing import Tuple
import pandas as pd
import yaml
from .utils import clean_text, find_first_existing

"""
This file handles dataset loading and configuration reading.
It reads params.yaml, loads CSV data, detects columns, and cleans text.
"""

class ConfigError(Exception):
    """
    Custom error for missing or invalid configuration.
    Used when dataset paths or columns are not found.
    """
    pass

def load_config(path: Path) -> dict:
    """
    Reads a YAML configuration file and returns it as a Python dictionary.
    The config file usually contains dataset and preprocess settings.
    """
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_dataset(cfg: dict, project_root: Path) -> Tuple[pd.DataFrame, str, str]:
    """
    Loads a CSV dataset and applies cleaning and label mapping.
    Detects text and label columns automatically and preprocesses them.
    """
    ds = cfg["dataset"]
    csv_path = project_root / ds["path"]
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {csv_path}")

    df = pd.read_csv(csv_path, encoding="utf-8", engine="python")

    text_col = find_first_existing(df.columns.tolist(), ds["text_columns"])
    label_col = find_first_existing(df.columns.tolist(), ds["label_columns"])

    if text_col is None:
        obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
        if obj_cols:
            text_col = obj_cols[0]
        else:
            raise ConfigError("Could not detect text column in dataset.")

    if label_col is None:
        if "label" not in df.columns:
            raise ConfigError("Could not detect label column; expected one of label/spam/target/class.")
        label_col = "label"

    pos = set(map(str, ds["positive_labels"]))
    neg = set(map(str, ds["negative_labels"]))

    def map_label(x):
        """
        Converts text labels into 1 (spam) and 0 (ham).
        Checks both config-defined and common label values.
        """
        xs = str(x)
        if xs in pos:
            return 1
        if xs in neg:
            return 0
        if xs.lower().strip() in {"spam", "1", "true", "yes"}:
            return 1
        return 0

    df = df[[text_col, label_col]].dropna()
    df[label_col] = df[label_col].map(map_label)

    pp = cfg["preprocess"]
    df[text_col] = df[text_col].astype(str).apply(
        lambda t: clean_text(
            t,
            lowercase=pp.get("lowercase", True),
            remove_urls=pp.get("remove_urls", True),
            remove_emails=pp.get("remove_emails", True),
            remove_numbers=pp.get("remove_numbers", True),
            remove_punctuation=pp.get("remove_punctuation", True),
            strip_whitespace=pp.get("strip_whitespace", True),
        )
    )

    return df, text_col, label_col