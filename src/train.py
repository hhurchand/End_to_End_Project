from __future__ import annotations

from src.model import Model
from src.utils.input import YAMLLoader
from main import run_pipeline


if __name__ == "__main__":
    df_processed = run_pipeline()

    settings = YAMLLoader().load_file("params.yaml")

    model_wrap = Model(df_processed, settings)
    split = model_wrap.train_test_split()
    rf = model_wrap.train_model(split)
    metrics = model_wrap.evaluate(rf, split)
    print("Final metrics:", metrics)

    path = model_wrap.save(rf, "models/random_forest.pkl")
    print("Saved model to:", path)