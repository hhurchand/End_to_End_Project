from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
class DataTransformation:
 
    def __init__(self, config,df):
        self.config = config
        self.df = df
 
    def check_input(self):
        print("This is from transform", self.config)
        
    def transform_data(self):
        categorical_features = self.config['features']['categorical_features']
        ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        transform_data = ColumnTransformer(transformers=[('onehot', ohe, categorical_features)],
                                   remainder='passthrough')
        