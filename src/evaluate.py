from typing import Dict
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, classification_report

"""
This file handles evaluation parts of the project.
It includes confusion matrix, summary results, and report printing.
"""

def save_confusion_matrix(y_true, y_pred, out_path: str, title: str) -> None:
    """
    Saves a confusion matrix image for visualizing predictions.
    It helps check how well the model classified spam and ham.
    """
    plt.figure()
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
    plt.title(title)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

def summarize_results(results: Dict[str, Dict[str, float]]) -> str:
    """
    Converts all model results into a CSV text format.
    Each row includes model name and accuracy, precision, recall, and f1.
    """
    lines = ["model,accuracy,precision,recall,f1"]
    for name, m in results.items():
        lines.append(f"{name},{m['acc']:.4f},{m['p']:.4f},{m['r']:.4f},{m['f1']:.4f}")
    return "\n".join(lines)

def print_hashes(name_to_preds: Dict[str, np.ndarray]) -> None:
    """
    Prints hash values of each model's predictions.
    Used for checking if predictions match without showing full arrays.
    """
    import hashlib
    for name, yhat in name_to_preds.items():
        arr = np.asarray(yhat)
        h = hashlib.md5(np.ascontiguousarray(arr)).hexdigest()
        print(f"[PRED_HASH] {name}: {h}")

def show_classification_report(y_true, y_pred, target_names=("Ham", "Spam"), digits: int = 4) -> None:
    """
    Displays a text-based classification report.
    It prints precision, recall, f1, and support for both classes.
    """
    print("\nCLASSIFICATION REPORT (Ham=0, Spam=1):\n")
    print(classification_report(y_true, y_pred, target_names=target_names, digits=digits))