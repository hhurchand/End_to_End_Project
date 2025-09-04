# src/transform_data.py
from typing import List, Tuple, Optional, Union
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

class DataTransformation:
    """
    Works with params like:
    data:
      raw_data: "data/raw/airlines_flights_data.csv"
    model:
      target: ['price']  # or "price"
    features:
      non_categorical_features: ['duration','days_left','price','stops']
      numerical_features: ['duration','days_left','stops']
      drop_features: ['flight','index']
    """

    def __init__(self, df: pd.DataFrame, config: dict):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")
        self.df = df.copy()
        self.config = config

    # ---------- config helpers ----------
    def _get_target_name(self) -> Optional[str]:
        tgt = (self.config.get("model", {}) or {}).get("target")
        # support string or list like ['price']
        if isinstance(tgt, list) and tgt:
            return tgt[0]
        if isinstance(tgt, str) and tgt.strip():
            return tgt
        # also support features.target fallback if ever used
        return (self.config.get("features", {}) or {}).get("target")

    def _get_drop_list(self) -> List[str]:
        feats = self.config.get("features", {}) or {}
        drops = feats.get("drop_features") or feats.get("drop_cols") or []
        return [c for c in drops if c in self.df.columns]

    def _get_numeric_list(self) -> List[str]:
        feats = self.config.get("features", {}) or {}
        nums = feats.get("numerical_features")
        if nums:
            return [c for c in nums if c in self.df.columns]
        # fallback: infer
        return self.df.select_dtypes(exclude=["object", "category"]).columns.tolist()

    def _get_non_cat_list(self) -> List[str]:
        feats = self.config.get("features", {}) or {}
        non_cat = feats.get("non_categorical_features", []) or []
        return [c for c in non_cat if c in self.df.columns]

    # ---------- data helpers ----------
    def drop_features(self) -> None:
        drops = self._get_drop_list()
        if drops:
            self.df.drop(columns=drops, inplace=True, errors="ignore")

    def map_word_number(self) -> None:
        # Only if 'stops' is textual like 'zero','one','two_or_more'
        if "stops" in self.df.columns and self.df["stops"].dtype == object:
            self.df["stops"] = self.df["stops"].map({"zero": 0, "one": 1, "two_or_more": 2})

    def get_categorical_features(self) -> List[str]:
        # Start from all columns minus declared non-categorical and target
        cols = list(self.df.columns)
        for c in self._get_non_cat_list():
            if c in cols:
                cols.remove(c)
        tgt = self._get_target_name()
        if tgt and tgt in cols:
            cols.remove(tgt)
        # Keep only object/category among the remainder
        return self.df[cols].select_dtypes(include=["object", "category"]).columns.tolist()

    # ---------- pipeline ----------
    def _build_preprocessor(self) -> ColumnTransformer:
        cat_cols = self.get_categorical_features()
        num_cols = self._get_numeric_list()

        num_tf = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ])
        cat_tf = Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])

        return ColumnTransformer(
            transformers=[
                ("cat", cat_tf, cat_cols),
                ("num", num_tf, num_cols),
            ],
            remainder="drop"
        )

    def one_hot_encode(self) -> pd.DataFrame:
        pre = self._build_preprocessor()
        arr = pre.fit_transform(self.df)
        cols = pre.get_feature_names_out()
        out = pd.DataFrame(arr, columns=cols, index=self.df.index)
        # tidy names
        out.columns = (out.columns
                       .str.replace(r"^(cat|num)__", "", regex=True)
                       .str.replace(r"^remainder__", "", regex=True))
        return out

    def transformation_pipeline(self) -> Tuple[pd.DataFrame, Optional[pd.Series], ColumnTransformer]:
        # 1) basic cleanup
        self.drop_features()
        self.map_word_number()

        # 2) transform
        df_proc = self.one_hot_encode()

        # 3) split X, y if target available post-transform
        tgt = self._get_target_name()
        y = None
        if tgt and tgt in df_proc.columns:
            y = df_proc[tgt]
            X = df_proc.drop(columns=[tgt])
        else:
            X = df_proc

        # return fitted preprocessor if you want to persist it
        pre = self._build_preprocessor()
        pre.fit(self.df)  # fit once more so caller gets a fitted object
        return X, y, pre
