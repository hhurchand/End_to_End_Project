from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, List
import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit  # teacher-like split
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

"""
This file contains model-related functions and classes.
It handles data splitting, model training, and result evaluation.
"""

@dataclass
class Split:
    """
    A small data class to hold split data for training and testing.
    Stores both text and label sets for easy model handling.
    """
    X_train_text: List[str]
    X_test_text: List[str]
    y_train: np.ndarray
    y_test: np.ndarray
    groups_train: np.ndarray
    groups_test: np.ndarray

def _label_to_int(y) -> np.ndarray:
    """
    Converts string labels like 'spam' and 'ham' to integers.
    Spam becomes 1 and ham becomes 0.
    """
    if hasattr(y, "to_numpy"):
        y = y.to_numpy()
    y = np.asarray(y)
    if y.dtype.kind in {"i", "u"}:
        return y.astype(int)
    y = np.asarray([str(v).lower().strip() for v in y])
    return np.where(y == "spam", 1, 0).astype(int)

def _metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculates accuracy, precision, recall, and f1-score.
    Returns a dictionary of metric results for each model.
    """
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return {"acc": float(acc), "p": float(p), "r": float(r), "f1": float(f1)}

def split_and_vectorize(
    df,
    text_col: str,
    label_col: str,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Splits the dataset into training and testing sets using stratified sampling.
    Also creates placeholders for diagnostic values like vocab size.
    """
    X_text = df[text_col].astype(str).tolist()
    y = _label_to_int(df[label_col])

    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    (tr_idx, te_idx), = sss.split(np.arange(len(y)), y)

    Xtr_text = [X_text[i] for i in tr_idx]
    Xte_text = [X_text[i] for i in te_idx]
    y_tr, y_te = y[tr_idx], y[te_idx]

    diags = {
        "vocab_size": 0.0,
        "max_sim_min": 0.0,
        "max_sim_median": 0.0,
        "max_sim_mean": 0.0,
        "max_sim_max": 0.0,
        "share_test_ge_0.99": 0.0,
    }

    split = Split(
        X_train_text=Xtr_text,
        X_test_text=Xte_text,
        y_train=y_tr,
        y_test=y_te,
        groups_train=np.zeros_like(y_tr),
        groups_test=np.zeros_like(y_te),
    )
    return split, diags 

def train_supervised(
    X_train_text: List[str],
    y_train: np.ndarray,
    X_test_text: List[str],
    y_test: np.ndarray,
    verbose: bool = True,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, np.ndarray]]:
    """
    Trains Logistic Regression, Naive Bayes, and Random Forest models.
    Evaluates each model and prints performance results.
    """
    results: Dict[str, Dict[str, float]] = {}
    preds: Dict[str, np.ndarray] = {}

    pipe_lr = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 1), max_df=0.9, min_df=5, strip_accents="unicode"),
        LogisticRegression(max_iter=1000, class_weight="balanced", C=0.5, random_state=123),
    )
    pipe_nb = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), max_df=0.98, min_df=5, strip_accents="unicode", sublinear_tf=True),
        MultinomialNB(alpha=1.0),
    )
    pipe_rf = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), max_df=0.95, min_df=5, strip_accents="unicode"),
        RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    )
    for name, pipe in [("logreg", pipe_lr), ("nb", pipe_nb), ("rf", pipe_rf)]:
        pipe.fit(X_train_text, y_train)
        yhat = pipe.predict(X_test_text)
        results[name] = _metric_dict(y_test, yhat)
        preds[name] = yhat
        if verbose:
            print(f"[{name.upper()}] acc={results[name]['acc']:.4f} f1={results[name]['f1']:.4f}")

    return results, preds