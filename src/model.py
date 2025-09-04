from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split as sk_train_test_split


@dataclass
class SplitResult:
    X_train: Any
    X_test: Any
    y_train: pd.Series
    y_test: pd.Series


class Model:
    """
    Wraps: feature/target split, train/test split, RF training, and evaluation.
    Works on the *processed* dataframe returned by the transform pipeline.
    """

    def __init__(self, processed_df: pd.DataFrame, config: Dict):
        self.df = processed_df
        self.cfg = config

        target_cfg = self.cfg.get("model", {}).get("target", [])
        if isinstance(target_cfg, list) and len(target_cfg) == 1:
            self.target_name = target_cfg[0]
        elif isinstance(target_cfg, str):
            self.target_name = target_cfg
        else:
            raise ValueError(
                "Config error: model.target must be a single column name (e.g., ['price'])."
            )

        if self.target_name not in self.df.columns:
            raise KeyError(
                f"Target column '{self.target_name}' not found in processed dataframe."
            )

    
    def _xy(self) -> Tuple[np.ndarray, pd.Series]:
        """Return X (np.ndarray) and y (Series) from processed df."""
        y = pd.to_numeric(self.df[self.target_name], errors="coerce")
        X = self.df.drop(columns=[self.target_name])

        
        try:
            X_mat = X.to_numpy()
        except Exception:
            X_mat = np.asarray(X)

    
        if y.isna().any():
            keep = ~y.isna()
            y = y.loc[keep]
            X_mat = X_mat[keep.values, :]

        return X_mat, y


    def train_test_split(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
        shuffle: bool = True,
    ) -> SplitResult:
        X, y = self._xy()
        X_train, X_test, y_train, y_test = sk_train_test_split(
            X, y, test_size=test_size, random_state=random_state, shuffle=shuffle
        )
        return SplitResult(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

    def train_model(
        self,
        split: SplitResult,
        n_estimators: int | None = None,
        random_state: int = 42,
        n_jobs: int = -1,
    ) -> RandomForestRegressor:
        if n_estimators is None:
            n_estimators = int(self.cfg.get("model", {}).get("n_estimators", 100))

        model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        model.fit(split.X_train, split.y_train)

    
        try:
            if mlflow.active_run() is not None:
                mlflow.log_param("rf_n_estimators", n_estimators)
        except Exception:
            pass

        return model

    def evaluate(
        self,
        model: RandomForestRegressor,
        split: SplitResult,
        log_to_mlflow: bool = True,
    ) -> Dict[str, float]:
        y_pred = model.predict(split.X_test)
        r2 = r2_score(split.y_test, y_pred)
        mae = mean_absolute_error(split.y_test, y_pred)
        mse = mean_squared_error(split.y_test, y_pred)
        rmse = float(np.sqrt(mse))

        metrics = {"r2": float(r2), "mae": float(mae), "mse": float(mse), "rmse": rmse}

        if log_to_mlflow:
            try:
                if mlflow.active_run() is not None:
                    mlflow.log_metrics(metrics)
            except Exception:
                pass

        return metrics

    def save(self, model: RandomForestRegressor, path: str | Path = "models/rf_model.pkl") -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, p)
        return p