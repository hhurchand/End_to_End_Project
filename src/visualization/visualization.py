import matplotlib.pyplot as plt
import numpy as np

def plot_ham_spam(y_train):
    """Bar chart for ham vs spam."""
    ham = int((y_train == 0).sum())
    spam = int((y_train == 1).sum())
    plt.figure()
    plt.bar(["Ham", "Spam"], [ham, spam])
    plt.title("Ham vs Spam")
    plt.xlabel("Class"); plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

def plot_cm(cm, class_names):
    """Display confusion matrix."""
    plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.colorbar()
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names)
    plt.yticks(ticks, class_names)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, int(cm[i, j]), ha="center", va="center")
    plt.ylabel("True"); plt.xlabel("Predicted")
    plt.tight_layout()
    plt.show()
