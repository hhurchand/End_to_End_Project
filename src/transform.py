from __future__ import annotations

from typing import Any, List

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


class TransformerX:
    """
    End-to-end preprocessing for the flights dataset:
      - Drop columns from config.features.drop_features
      - Convert 'stops' text -> numeric codes
      - One-hot encode categoricals; passthrough all others
      - Clean up column names for readability
    """

    def __init__(self, data_frame: pd.DataFrame, config: dict):
        self.df_main = data_frame.copy()
        self.cfg = config

    
    def _non_categorical_from_cfg(self) -> List[str]:
        return list(self.cfg["features"]["non_categorical_features"])

    def extract_cats(self) -> List[str]:
        non_cat = set(self._non_categorical_from_cfg())
        return [c for c in self.df_main.columns if c not in non_cat]

    def extract_nums(self) -> List[str]:
        return list(self.cfg["features"]["numerical_features"])

    def set_target(self) -> str:
        t = self.cfg["model"]["target"]
        return t[0] if isinstance(t, list) else t

    
    def strip_columns(self, to_remove: List[str]) -> pd.DataFrame:
        existing = [c for c in to_remove if c in self.df_main.columns]
        if existing:
            self.df_main = self.df_main.drop(columns=existing)
        return self.df_main

    def words_to_nums(self) -> pd.DataFrame:
        """Map 'stops' textual values to {zero:0, one:1, two_or_more:2} if present."""
        lookup = {"zero": 0, "one": 1, "two_or_more": 2}
        if "stops" in self.df_main.columns:
            self.df_main["stops"] = (
                self.df_main["stops"].map(lookup).astype("Int64")
            )
        return self.df_main

    def ohe_transform(self) -> pd.DataFrame:
        """One-hot encode categoricals; passthrough the rest (including numeric + target)."""
        cats = self.extract_cats()

        
        try:
            encoder = OneHotEncoder(sparse_output=True, handle_unknown="ignore")
        except TypeError:
            encoder = OneHotEncoder(sparse=True, handle_unknown="ignore")

        ct = ColumnTransformer(
            transformers=[("encoder", encoder, cats)],
            remainder="passthrough",
        )

        matrix = ct.fit_transform(self.df_main)  # often sparse
        col_names = ct.get_feature_names_out()

        self.df_main = pd.DataFrame.sparse.from_spmatrix(
            matrix, index=self.df_main.index, columns=col_names
        )
        return self.df_main

    def tidy_names(self) -> pd.DataFrame:
        """Drop 'encoder__' and 'remainder__' prefixes for readability."""
        self.df_main.columns = (
            pd.Index(self.df_main.columns)
            .str.replace(r"^encoder__", "", regex=True)
            .str.replace(r"^remainder__", "", regex=True)
        )
        return self.df_main

    
    def full_pipeline(self) -> pd.DataFrame:
        # 1) Drop configured columns
        drops = list(self.cfg["features"].get("drop_features", []))
        self.strip_columns(drops)

        
        self.words_to_nums()

        
        self.ohe_transform()

    
        self.tidy_names()

        return self.df_main