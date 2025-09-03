import pandas as pd
from typing import List
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


class DataTransformation:

    def __init__(self, df: pd.DataFrame, config):
        self.config = config
        self.df = df.copy()

        self.target = config["target"]
        self.non_categorical_features = config["features"]["non_categorical_features"]
        self.numerical_features = config["features"]["numerical_features"]

    def encode(self):