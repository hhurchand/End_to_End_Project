from __future__ import annotations

from pathlib import Path

import mlflow
import pandas as pd

from src.utils.input import CSVLoader, YAMLLoader
from src.transform import TransformerX


PROJECT_ROOT = Path(__file__).parent
CONFIG_FILE = PROJECT_ROOT / "params.yaml"


def resolve_data_path(cfg: dict) -> Path:
    """
    Use cfg['data']['raw_data'] if present.
    Supports absolute or relative paths (relative to repo root).
    Falls back to data/raw/airlines_flights_data.csv.
    """
    try:
        raw_path = cfg["data"]["raw_data"]
        p = Path(raw_path)
        if not p.is_absolute():
            p = PROJECT_ROOT / raw_path
        return p
    except Exception:
        return PROJECT_ROOT / "data" / "raw" / "airlines_flights_data.csv"


def run_pipeline() -> pd.DataFrame:
    print("Loading configuration...")
    settings = YAMLLoader().load_file(str(CONFIG_FILE))

    data_path = resolve_data_path(settings)
    print(f"Using data file: {data_path}")

    print("Loading raw CSV...")
    raw_df: pd.DataFrame = CSVLoader().load_file(file_path=str(data_path))
    print(f"Loaded dataset with shape {raw_df.shape}")
    print(f"Columns: {list(raw_df.columns)}")

    print("Building transformer...")
    tfm = TransformerX(raw_df, settings)

    print("Running full pipeline...")
    processed_df = tfm.full_pipeline()
    print(f"Transformed shape: {processed_df.shape}")

  
    out_dir = PROJECT_ROOT / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "airlines_flights_data_processed.csv"
    processed_df.to_csv(out_file, index=False)
    print(f"Saved processed data to: {out_file}")

    
    print("Logging params to MLflow...")
    mlflow.set_experiment("flights_analysis_experiment")
    with mlflow.start_run():
        if "model" in settings and "n_estimators" in settings["model"]:
            mlflow.log_param("n_estimators", settings["model"]["n_estimators"])
        mlflow.log_param("n_rows_raw", int(raw_df.shape[0]))
        mlflow.log_param("n_cols_raw", int(raw_df.shape[1]))
        mlflow.log_param("n_rows_processed", int(processed_df.shape[0]))
        mlflow.log_param("n_cols_processed", int(processed_df.shape[1]))
        drops = settings["features"].get("drop_features", [])
        mlflow.log_param("dropped_columns", ",".join(drops) if drops else "none")
        mlflow.log_param(
            "non_categorical_features",
            ",".join(settings["features"]["non_categorical_features"]),
        )
        mlflow.log_param(
            "numerical_features",
            ",".join(settings["features"]["numerical_features"]),
        )

    print("MLflow logging complete.")
    print("Pipeline finished.")
    return processed_df


if __name__ == "__main__":
    run_pipeline()         