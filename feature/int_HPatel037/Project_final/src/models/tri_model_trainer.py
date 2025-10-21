from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple, Optional, Union
import re
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
import mlflow
import mlflow.sklearn

@dataclass
class SplitData:
    X_train: Any
    X_test: Any
    y_train: np.ndarray
    y_test: np.ndarray
    vectorizer: Any

def _read_input_csv(csv_path: Union[Path, str]) -> pd.DataFrame:
    return pd.read_csv(csv_path)

def prepare_data(cfg: Dict[str, Any], root: Path) -> SplitData:
    text_col, label_col = cfg["data"]["text_col"], cfg["data"]["label_col"]
    df = _read_input_csv(cfg["data"]["raw_Dataset"])
    df = df.dropna(subset=[text_col, label_col])

    df[text_col] = df[text_col].astype(str).str.lower()
    y = df[label_col].astype(str).str.lower().apply(lambda v: 1 if "spam" in v or v == "1" else 0).to_numpy()

    tfidf = TfidfVectorizer(max_df=0.95, min_df=2, ngram_range=(1, 2))
    X = tfidf.fit_transform(df[text_col])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    return SplitData(X_train, X_test, y_train, y_test, tfidf)

def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1}

def grid_for(name: str):
    if name == "logreg":
        model = LogisticRegression(max_iter=2000)
        grid = {"C": [0.1, 1.0], "class_weight": [None, "balanced"]}
    elif name == "linearsvc":
        model = LinearSVC(max_iter=5000, dual="auto")
        grid = {"C": [0.5, 1.0], "class_weight": [None, "balanced"]}
    elif name == "rf":
        model = RandomForestClassifier(random_state=42)
        grid = {"n_estimators": [300, 500], "max_depth": [None, 30]}
    else:
        raise ValueError(f"Unknown model: {name}")
    gcv = GridSearchCV(model, grid, cv=3, n_jobs=-1, scoring="f1")
    return gcv, grid

def train_and_log_all(cfg, split):
    results = {}
    for name in ["logreg", "linearsvc", "rf"]:
        with mlflow.start_run(run_name=name, nested=True):
            gcv, grid = grid_for(name)
            gcv.fit(split.X_train, split.y_train)
            y_pred = gcv.predict(split.X_test)
            mets = compute_metrics(split.y_test, y_pred)
            mlflow.log_metrics(mets)
            mlflow.log_params(gcv.best_params_)
            mlflow.sklearn.log_model(gcv.best_estimator_, f"{name}_model")
            mlflow.log_text(classification_report(split.y_test, y_pred, digits=4), "classification_report.txt")
            results[name] = {"model": gcv.best_estimator_, **mets}
    return results
