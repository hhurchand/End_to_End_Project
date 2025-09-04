from typing import List
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd

class DataTransformation:
    def __init__(self, df: pd.DataFrame, config: dict):
        self.df = df.copy()
        self.config = config

    def _drop_features(self, cols: List[str]) -> pd.DataFrame:
        self.df.drop(columns=cols, axis=1, inplace=True)
        return self.df

    def get_categorical_features(self) -> List[str]:
        all_features = list(self.df.columns)
        non_categorical = self.config['features']['non_categorical_features']
        return [f for f in all_features if f not in non_categorical]

    def numerical_features(self) -> List[str]:
        return self.config['features']['numerical_features']

    def get_target(self) -> str:
        return self.config['target']

    def map_word_number(self) -> pd.DataFrame:
        map_dict = {"one": 1, "zero": 0, "two_or_more": 2}
        if "stops" in self.df.columns:
            self.df["stops"] = self.df["stops"].map(map_dict)
        return self.df

    def one_hot_encode(self) -> pd.DataFrame:
        ohe = OneHotEncoder(sparse_output=True, handle_unknown='ignore')
        categorical_features = self.get_categorical_features()
        transformer = ColumnTransformer(
            transformers=[('onehot', ohe, categorical_features)],
            remainder='passthrough'
        )
        encoded_sparse = transformer.fit_transform(self.df)
        feature_names = transformer.get_feature_names_out()
        self.df = pd.DataFrame.sparse.from_spmatrix(encoded_sparse, columns=feature_names)
        return self.df

    def rename_dataframe_columns(self) -> pd.DataFrame:
        self.df.columns = self.df.columns.str.replace("remainder__", "")
        self.df.columns = self.df.columns.str.replace("onehot__", "")
        return self.df

    def transformation_pipeline(self) -> pd.DataFrame:
        self.map_word_number()
        self.one_hot_encode()
        self.rename_dataframe_columns()
        return self.df
