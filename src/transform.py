from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


class DataTransformation:
    def __init__(self,df,config):
        self.df = df.copy()
        self.config = config
        

    def ohencoder(self,features,config):
        self.features = config["features"]["categorical_features"]     
        ohe = OneHotEncoder(sparse_output=False,handle_unknown='ignore')
        return ColumnTransformer(transformers=[('onehot',ohe,self.features)],
                                        remainder='passthrough')

    def check_input(self):
        print("this is from transform",self.config)
