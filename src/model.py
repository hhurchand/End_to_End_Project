from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
from typing import Tuple


class Model():
        
    def __init__(self,df:pd.DataFrame):
        self.df = df.copy()
        
    def train_test_split(self) -> Tuple:
        X = self.df.drop(["price"],axis=1)
        y = self.df["price"]
        X_train,X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)
        return X_train, X_test, y_train, y_test
    
    def train_model(self,X_train,y_train):
        model = RandomForestRegressor(random_state=42)
        model.fit(X_train,y_train)
        return model



