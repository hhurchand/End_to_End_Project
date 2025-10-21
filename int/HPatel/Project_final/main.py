from pathlib import Path
from typing import Dict, Any
from yaml import safe_load
import mlflow
from joblib import dump as jobdump
import pandas as pd
import sys

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))

from src.models.tri_model_trainer import prepare_data, train_and_log_all

PARAMS = ROOT / "params.yaml"
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "model"

def load_config(path: Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return safe_load(f)

def merge_datasets(data_dir: Path) -> Path:
    df1 = pd.read_csv(data_dir / "combined_dataset.csv")
    df2 = pd.read_csv(data_dir / "enron_spam_data.csv")
    df = pd.concat([df1, df2], ignore_index=True).drop_duplicates()
    merged = data_dir / "merged_spam.csv"
    df.to_csv(merged, index=False)
    print(f"Merged dataset saved: {merged} | Shape={df.shape}")
    return merged

def main():
    cfg = load_config(PARAMS)
    merged_path = merge_datasets(DATA_DIR)
    cfg["data"]["raw_Dataset"] = str(merged_path)

    with mlflow.start_run(run_name="EmailSpam_TriModel_Pipeline"):
        split = prepare_data(cfg, ROOT)
        results = train_and_log_all(cfg, split)
        MODEL_DIR.mkdir(exist_ok=True)
        for name, r in results.items():
            jobdump(r["model"], MODEL_DIR / f"{name}_model.joblib")
        jobdump(split.vectorizer, MODEL_DIR / "vectorizer.joblib")

        print("\n=== Model Accuracy Summary ===")
        for name, m in results.items():
            print(f"{name:10s} | Acc={m['accuracy']:.4f} | F1={m['f1']:.4f}")
        best = max(results, key=lambda k: results[k]["f1"])
        print(f"[BEST MODEL] {best.upper()} with F1={results[best]['f1']:.4f}")

if __name__ == "__main__":
    main()
