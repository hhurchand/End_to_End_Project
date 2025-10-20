import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix,roc_curve,auc,precision_recall_curve,average_precision_score)

# MY THEME
COLOR_HAM = "royalblue"
COLOR_SPAM = "aquamarine"
LINE_BASE = {"color": "black", "linestyle": "dotted"}


def save_the_graphs(file_name, outputs_dir):
    """
    SAVE THE GRAPH
    """
    path = os.path.join(outputs_dir, file_name)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_ham_vs_spam(y_true, y_predict, outputs_dir):
    """
    PLOT HAM 0 VS SPAM 1 COUNTS
    """
    counts = [(y_true == 0).sum(), (y_true == 1).sum()]
    labels = ["HAM", "SPAM"]
    colors = [COLOR_HAM, COLOR_SPAM]

    plt.figure(figsize=(6, 4))
    bars = plt.bar(labels, counts, color=colors, edgecolor="white")

    for bar, val in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width() / 2, val + 1, str(val),ha="center", va="bottom")

    plt.title("HAM VS SPAM")
    plt.xlabel("CATEGORY")
    plt.ylabel("COUNT")
    save_the_graphs("ham_vs_spam.png", outputs_dir)


def plot_confusion_matrix(y_true, y_predict, outputs_dir):
    """
    PLOT CONFUSION MATRIX
    """
    cm = confusion_matrix(y_true, y_predict)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"])
    plt.title("CONFUSION MATRIX")
    plt.xlabel("PREDICTED")
    plt.ylabel("TRUE")
    save_the_graphs("confusion_matrix.png", outputs_dir)


def plot_roc_curve(y_true, y_probability, outputs_dir):
    """
    PLOT ROC CURVE
    """
    fpr, tpr, _ = roc_curve(y_true, y_probability)
    auc_val = auc(fpr, tpr)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, color=COLOR_SPAM, label=f"AUC = {auc_val:.3f}")
    plt.plot([0, 1], [0, 1], **LINE_BASE)
    plt.xlabel("FALSE POSITIVE RATE")
    plt.ylabel("TRUE POSITIVE RATE")
    plt.title("ROC CURVE")
    plt.legend(loc="lower right")
    save_the_graphs("roc_curve.png", outputs_dir)


def plot_precision_recall_curve(y_true, y_probability, outputs_dir):
    """
    PLOT PRECISION-RECALL CURVE
    """
    precision, recall, _ = precision_recall_curve(y_true, y_probability)
    ap = average_precision_score(y_true, y_probability)
    plt.figure(figsize=(6, 4))
    plt.plot(recall, precision, color=COLOR_HAM, label=f"AP = {ap:.3f}")
    plt.plot([0, 1], [1, 0], **LINE_BASE)
    plt.xlabel("RECALL")
    plt.ylabel("PRECISION")
    plt.title("PRECISION-RECALL CURVE")
    plt.legend(loc="lower left")
    save_the_graphs("precision_recall_curve.png", outputs_dir)
