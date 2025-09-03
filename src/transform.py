# src/transform.py
from __future__ import annotations
from typing import Tuple, Dict, Any, List
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

class DataTransformation:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        feats = config.get("features", {})
        self.target_col: str = feats.get("target")
        self.num_cols: List[str] = feats.get("numeric", [])
        self.cat_cols: List[str] = feats.get("categorical", [])

        # Build the transformer now
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", "passthrough", self.num_cols),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), self.cat_cols),
            ],
            remainder="drop",
        )

        self.pipeline = Pipeline(steps=[("pre", self.preprocessor)])

    def fit_transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        self._check_columns(df)
        X = df[self.num_cols + self.cat_cols]
        y = df[self.target_col] if self.target_col in df.columns else None

        X_trans = self.pipeline.fit_transform(X)

        # Build feature names (optional but nice)
        cat_feature_names = []
        if self.cat_cols:
            ohe = self.preprocessor.named_transformers_["cat"]
            # scikit-learn >=1.0
            cat_feature_names = list(ohe.get_feature_names_out(self.cat_cols))
        feature_names = self.num_cols + cat_feature_names

        X_df = pd.DataFrame(X_trans, columns=feature_names, index=df.index)
        return X_df, y

    def transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        self._check_columns(df)
        X = df[self.num_cols + self.cat_cols]
        y = df[self.target_col] if self.target_col in df.columns else None
        X_trans = self.pipeline.transform(X)

        cat_feature_names = []
        if self.cat_cols:
            ohe = self.preprocessor.named_transformers_["cat"]
            cat_feature_names = list(ohe.get_feature_names_out(self.cat_cols))
        feature_names = self.num_cols + cat_feature_names

        X_df = pd.DataFrame(X_trans, columns=feature_names, index=df.index)
        return X_df, y

    def _check_columns(self, df: pd.DataFrame) -> None:
        required = set(self.num_cols + self.cat_cols + ([self.target_col] if self.target_col else []))
        missing = [c for c in required if c and c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns in dataframe: {missing}")
