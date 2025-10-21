from pathlib import Path
import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "model"

def load_model_and_vectorizer(model_name="rf"):
    vec = joblib.load(MODEL_DIR / "vectorizer.joblib")
    model = joblib.load(MODEL_DIR / f"{model_name}_model.joblib")
    return vec, model

def predict_label(text, model_name="rf"):
    vec, model = load_model_and_vectorizer(model_name)
    X = vec.transform([text])
    y = model.predict(X)[0]
    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0, 1]
    elif hasattr(model, "decision_function"):
        score = model.decision_function(X)[0]
        proba = 1 / (1 + np.exp(-score))
    label = "Spam" if int(y) == 1 else "Ham"
    return label, float(proba) if proba is not None else None
