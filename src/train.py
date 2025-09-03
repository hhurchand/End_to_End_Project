from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
from pathlib import Path
from typing import Union,List, Dict,Any


class MLDataPreparation():
        
    def __init__(self,df:pd.DataFrame):
        self.df = df.copy()


    def split_dataframe(self,test_size:float,X,y):
            x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=test_size,random_state =45)
            return x_train,x_test,y_train,y_test
    
    



