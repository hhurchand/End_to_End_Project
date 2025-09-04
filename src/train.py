from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
from pathlib import Path
from typing import Union,List, Dict,Tuple


class MLDataPrep():
        
    def __init__(self,df:pd.DataFrame,config:dict):
        self.df = df.copy()
        self.config = config



    def split_dataframe(self,X,y) -> Tuple:
        random_state = self.config["prep"]["random_state"]
        test_size = self.config["prep"]["test_size"]
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=test_size,random_state= random_state)
        return X_train,X_test,y_train,y_test
    
    def train_model(self,X_train,y_train):
        model_name = self.config["prep"]["ml_models_1"]
        model_params = self.config["model"].get(model_name,{})
        model_classes = {"RandomForestRegressor": RandomForestRegressor }
        model_class = model_classes.get(model_name)
        model = model_class(**model_params)
        model.fit(X_train,y_train)
        return model
    

    def train_pipeline(self):

        target_col = self.config["prep"]["target_col"]
        X = self.df.drop(columns=[target_col])
        y = self.df[target_col]
        X_train, X_test, y_train, y_test = self.split_dataframe(X=X, y=y)
        trained_model = self.train_model(X_train, y_train)
        return trained_model, X_test, y_test


        
    




