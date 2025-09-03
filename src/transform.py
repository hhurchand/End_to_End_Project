from typing import List
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd

class DataTransformation:

    def __init__(self,df:pd.DataFrame,config:dict):
        self.config = config
        self.df = df.copy()


    def get_categorical_features(self) -> List:

        numerical_features = self.config["features"]["numerical_features"]
        discard_columns = self.config["features"]["drop_column"]
        target_feature = self.config["model"]["target"]
        exclude_feature = numerical_features + target_feature

        categorical_features = [
            column for column in self.df.columns
            if column not in exclude_feature
        ]

        filter_features = [col for col in categorical_features
                           if col not in discard_columns]

        return filter_features
    

    def data_transform(self,filter_features:List) -> pd.DataFrame:

        discard_columns = self.config["features"]["drop_column"]
        self.df = self.df.drop(columns=discard_columns)

        ohe = OneHotEncoder(sparse_output=False,handle_unknown="ignore")
        transform = ColumnTransformer(transformers=[("onehot",ohe,filter_features)],remainder="passthrough")

        target_feature = self.config["model"]["target"]
        df_to_transform = self.df.drop(columns=target_feature)

        transform_data = transform.fit_transform(df_to_transform)  
        feature_names = transform.get_feature_names_out(df_to_transform.columns)

        df_transformed = pd.DataFrame(transform_data,columns=feature_names)
        df_transformed[target_feature[0]] = self.df[target_feature[0]].values

        return df_transformed
    
 
    def data_encode(self,df:pd.DataFrame) -> pd.DataFrame:

        df_encode = df.copy()
        df_encode.columns= df_encode.columns.str.replace("onehot__","")
        df_encode.columns = df_encode.columns.str.replace("remainder__","")
        df_encode["stops"] = df_encode["stops"].map({"zero":0,"one":1,"two_or_more":2})
 
        return df_encode

