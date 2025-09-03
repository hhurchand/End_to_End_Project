import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

class DataTransformation:
    def __init__(self, config: dict):
        feats = config["features"]
        self.cat_cols = feats["non_categorical_features"]
        self.num_cols = feats["numerical_features"]
        self.target  = feats["target"]

        self.prep = ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), self.cat_cols),
            ("num", StandardScaler(), self.num_cols),
        ])

    def fit_transform(self, df: pd.DataFrame):
        X = df[self.cat_cols + self.num_cols]
        y = df[self.target]
        X_t = self.prep.fit_transform(X)
        return X_t, y

    def transform(self, df: pd.DataFrame):
        X = df[self.cat_cols + self.num_cols]
        y = df[self.target]
        X_t = self.prep.transform(X)
        return X_t, y
