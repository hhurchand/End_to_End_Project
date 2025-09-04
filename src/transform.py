from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd

class DataTransformation:
    def __init__(self,df, config):
        self.config = config
        self.df = df.copy()
        
    def check_input(self):
        print(self.config)

    def one_hot_encode(self):
        categorical_features = self.config["features"]["categorical_features"]
        # One hot encode our categorical variables
        #sparse_output = False allows us to get a dataFrame output
        #handle_unknown = "ignore" is important so that new data are considered during production
        ohe = OneHotEncoder(sparse_output=False,handle_unknown="ignore")
        self.transform_data = ColumnTransformer(transformers=[("onehot",ohe,categorical_features)],remainder="passthrough")
        self.data = self.transform_data.fit_transform(self.df)

        #passthrough indicates that there will be other features which will not be one hot encoded

    def encode_data(self):
        features = self.transform_data.get_feature_names_out()
        self.df_encoded = pd.DataFrame(self.data, columns=features)

    def cleanup_encoded_data_columns(self):
        self.df_encoded.columns = self.df_encoded.columns.str.replace("onehot__","")
        self.df_encoded.columns = self.df_encoded.columns.str.replace("remainder__","")
        self.df_encoded.drop("flight",axis=1, inplace=True)
        
    def map_stops_to_numerical(self):
        stops_mapping = {
            "zero": 0,
            "one": 1,
            "two_or_more": 2
        }

        if "stops" in self.df_encoded.columns:
            self.df_encoded["stops"] = self.df["stops"].map(stops_mapping)


    def transform_data_pipeline(self):
        self.one_hot_encode()
        self.encode_data()
        self.cleanup_encoded_data_columns()
        self.map_stops_to_numerical()
        

    def get_target(self):
        return self.config["model"]["target"]

    def get_numerical_features(self):
        return self.config["features"]["numerical_features"]


