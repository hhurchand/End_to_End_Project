import pandas as pd
from sklearn.preprocessing import OneHotEncoder

class Transform:

    def __init__(self, df, config):
        self.df = df.copy()
        self.config = config

    def drop_features(self):
        cols_to_drop = self.config["features"]["drop_features"]
        self.df = self.df.drop(columns=cols_to_drop, errors="ignore")
        return self.df

    def get_categorical(self):
        all_cols = list(self.df.columns)
        non_cat = self.config["features"]["non_categorical_features"]
        return [col for col in all_cols if col not in non_cat]

    def encode(self):
        cat_cols = self.get_categorical()
        ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        encoded = ohe.fit_transform(self.df[cat_cols])
        
        new_cols = ohe.get_feature_names_out(cat_cols)
        encoded_df = pd.DataFrame(encoded, columns=new_cols)

        num_df = self.df.drop(columns=cat_cols)

        self.df = pd.concat([num_df.reset_index(drop=True),
                             encoded_df.reset_index(drop=True)], axis=1)
        return self.df

    def rename(self):
        self.df.columns = self.df.columns.str.replace("remainder__", "", regex=False)
        self.df.columns = self.df.columns.str.replace("onehot__", "", regex=False)
        return self.df


    def map_stops(self):
        if "stops" in self.df.columns:
            mapping = {"zero": 0, "one": 1, "two_or_more": 2}
            self.df["stops"] = self.df["stops"].map(mapping).fillna(self.df["stops"])
        return self.df

    def pipeline(self):
        self.drop_features()
        self.encode()
        self.rename()
        self.map_stops()
        return self.df
