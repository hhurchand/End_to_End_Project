from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from typing import List
import pandas as pd


class DataTransformation:
    def __init__(self,df,config):
        self.df = df.copy()
        self.config = config
        self.transform_data = None
    

    def ohencoder(self):
        features = self.config["features"]["categorical_features"]     
        ohe = OneHotEncoder(sparse_output=False,handle_unknown='ignore')
        transform_data = ColumnTransformer(transformers=[('onehot',ohe,features)],
                                        remainder='passthrough')
        return transform_data

    def fit_transform_obj(self):
        if self.transform_data == None:
            self.transform_data = self.ohencoder()
            data = self.transform_data.fit_transform(self.df)
        else:
            data = self.transform_data.transform(self.df)
        return data

    
    def features_cleaning(self):
        data = self.fit_transform_obj()
        features = self.transform_data.get_feature_names_out()
        df_encoded = pd.DataFrame(data, columns=features)
        df_encoded.columns = df_encoded.columns.str.replace("onehot__", "")
        df_encoded.columns = df_encoded.columns.str.replace("remainder__", "")
        return df_encoded

    
    def mapping(self):
        df_encoded = self.features_cleaning()
        df_encoded.drop(["flight"], axis=1, inplace=True)
        mapping = {"zero": 0, "one": 1, "two_or_more": 2}
        df_encoded['stops'] = df_encoded['stops'].map(mapping)
        return df_encoded
        