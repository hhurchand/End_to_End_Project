from pathlib import Path
import argparse
import pandas as pd

from src.data import load_config, load_dataset
from src.models import split_and_vectorize, train_supervised
from src.evaluate import (
    save_confusion_matrix,
    summarize_results,
    print_hashes,
    show_classification_report,
)

"""
This main file runs the training and evaluation of all models.
It loads the dataset, prepares data, trains models, and saves reports.
The goal is to compare ML models and show results neatly.
"""

PROJECT_ROOT = Path(__file__).resolve().parent

def _print_clean_summary(results: dict) -> str:
    """
    Makes a readable summary table for all models.
    It sorts them by F1 score and prints accuracy, precision, recall, and F1.
    """
    ordered = sorted(results.items(), key=lambda kv: kv[1]["f1"], reverse=True)
    lines = []
    lines.append("\nMODEL PERFORMANCE SUMMARY\n")
    lines.append(f"{'Model':<10} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1-Score':>10}")
    lines.append("-" * 55)
    for name, m in ordered:
        lines.append(
            f"{name.upper():<10} "
            f"{m['acc']*100:>8.2f}% "
            f"{m['p']*100:>9.2f}% "
            f"{m['r']*100:>7.2f}% "
            f"{m['f1']*100:>9.2f}%"
        )
    return "\n".join(lines)

def main() -> None:
    """
    This is the main workflow for model training.
    It loads data, runs the models, and prints summaries and reports.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true", help="show diagnostics/logs")
    args = parser.parse_args()
    VERBOSE = args.verbose

    cfg = load_config(PROJECT_ROOT / "params.yaml")
    df, text_col, label_col = load_dataset(cfg, PROJECT_ROOT)

    before = len(df)
    df = df.drop_duplicates(subset=[text_col]).reset_index(drop=True)
    if VERBOSE:
        print(f"[DEDUP] Removed {before - len(df)} duplicate messages before splitting.")

    split, diags = split_and_vectorize(df, text_col, label_col, test_size=0.2, random_state=42)

    if VERBOSE:
        print(f"[DIAG] rows_after_dedup={len(df)}")
        overall = (
            pd.Series(df[label_col]).map({"ham": 0, "spam": 1, 0: 0, 1: 1}).astype(int).value_counts().to_dict()
        )
        print(f"[DIAG] label_counts_overall: {overall}")
        print(f"[DIAG] label_counts_train : {pd.Series(split.y_train).value_counts().to_dict()}")
        print(f"[DIAG] label_counts_test  : {pd.Series(split.y_test).value_counts().to_dict()}")
        print(f"[DIAG] vocab_size: {int(diags['vocab_size'])}")
        print(
            "[DIAG] test→train max cosine similarity: "
            f"min/median/mean/max = {diags['max_sim_min']} {diags['max_sim_median']} "
            f"{diags['max_sim_mean']} {diags['max_sim_max']}"
        )
        print(f"[DIAG] share of test with max_sim >= 0.99: {diags['share_test_ge_0.99']}")

    results, preds = train_supervised(
        split.X_train_text, split.y_train, split.X_test_text, split.y_test,
        verbose=VERBOSE
    )

    if VERBOSE:
        print_hashes(preds)

    out_dir = PROJECT_ROOT / "reports" / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    best = max(results, key=lambda k: results[k]["f1"])
    save_confusion_matrix(split.y_test, preds[best], out_dir / f"cm_{best}.png", f"Confusion: {best}")

    show_classification_report(split.y_test, preds[best], target_names=("Ham", "Spam"), digits=4)

    pretty = _print_clean_summary(results)
    print(pretty)
    (out_dir / "summary_pretty.txt").write_text(pretty + "\n", encoding="utf-8")

    summary_csv = summarize_results(results)
    (out_dir / "summary.csv").write_text(summary_csv, encoding="utf-8")

    print(f"\n Best Model by F1-Score: {best.upper()}")
    print(f" Plots & results saved in: {out_dir}")

if __name__ == "__main__":
    main()
