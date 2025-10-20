import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def the_metrics(y_true, y_predict) -> pd.DataFrame:
    """
    CALCULATE & RETURN BASIC METRICS AS A DATAFRAME
    """
    accuracy = accuracy_score(y_true, y_predict)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_predict, average="binary", pos_label=1, zero_division=0)

    return pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1"],
        "Value":  [round(accuracy, 4), round(precision, 4), round(recall, 4), round(f1, 4)]})
