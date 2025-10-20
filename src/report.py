import os
from sklearn.metrics import classification_report
from src.metrics import the_metrics
from src.plots import (plot_ham_vs_spam,plot_confusion_matrix,plot_roc_curve,plot_precision_recall_curve)

def save_classification_report(y_true, y_predict, out_path_txt: str):
    """
    SAVE CLASSIFICATION REPORT AS .TXT FILE
    """
    report = classification_report(y_true, y_predict, digits=4, zero_division=0)
    with open(out_path_txt, "w") as f:
        f.write(report)
    return out_path_txt


def the_report(y_test, y_predict, y_probability, parameter: dict):
    """
    CREATE REPORT: SAVE PLOTS + METRICS
    """
    outputs_dir = parameter["data"]["outputs"]
    os.makedirs(outputs_dir, exist_ok=True)

    print(f"\n THE REPORT : {outputs_dir}")

    # PLOTS
    plot_ham_vs_spam(y_test, y_predict, outputs_dir)
    plot_confusion_matrix(y_test, y_predict, outputs_dir)
    if y_probability is not None:
        plot_roc_curve(y_test, y_probability, outputs_dir)
        plot_precision_recall_curve(y_test, y_probability, outputs_dir)

    # METRICS 
    metrics_df = the_metrics(y_test, y_predict)
    save_classification_report(y_test, y_predict,
                                    os.path.join(outputs_dir, "classification_report.txt"))

    print("REPORT COMPLETE. FILES SAVED.")
    return metrics_df
