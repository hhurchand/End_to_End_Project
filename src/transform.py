import pandas as pd
from typing import List
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


class DataTransformation:

    # SAVE CONFIG + DATAFRAME

    def __init__(self, df: pd.DataFrame, config):

        self.config = config
        self.df = df.copy()



    # DROP FEATURES

    def drop_features(self) -> pd.DataFrame:

        drop_cols = self.config["features"]["drop_features"]
        self.df = self.df.drop(columns=drop_cols, errors="ignore")

        return self.df
    


    # GET CATEGORICAL COLUMNS

    def get_categorical(self) -> List[str]:

        all_cols = list(self.df.columns)
        non_cat = self.config["features"]["non_categorical_features"]
        for x in non_cat:
            if x in all_cols:
                all_cols.remove(x)

        return all_cols
    


    # ENCODE CATEGORICAL

    def encode(self) -> pd.DataFrame:

        ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        transform = ColumnTransformer(
            [("onehot", ohe, self.get_categorical())], remainder="passthrough")

        encoded = transform.fit_transform(self.df)
        features = transform.get_feature_names_out()
        self.df = pd.DataFrame(encoded, columns=features)

        return self.df
    


    # RENAME COLUMNS

    def rename(self) -> pd.DataFrame:

        self.df.columns = self.df.columns.str.replace("remainder__", "", regex=False)
        self.df.columns = self.df.columns.str.replace("onehot__", "", regex=False)

        return self.df
    


    # MAP STOPS TO NUMBERS

    def map_stops(self) -> pd.DataFrame:

        mapping = {"zero": 0, "one": 1, "two_or_more": 2}
        if "stops" in self.df.columns:
            self.df["stops"] = self.df["stops"].map(mapping).fillna(self.df["stops"])
        
        return self.df
    


    # PIPELINE

    def pipeline(self) -> pd.DataFrame:

        # DROP - ENCODE - RENAME - MAP
        self.df = self.drop_features()
        self.df = self.encode()
        self.df = self.rename()
        self.df = self.map_stops()

        return self.df
