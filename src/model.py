import pandas as pd
from typing import Dict, Tuple

from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


import pandas as pd
from typing import Dict
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

class Model:

    def __init__(self, df: pd.DataFrame, config: Dict):

        self.df = df
        self.config = config

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None



    def train_test_split(self, test_size: float = 0.2, random_state: int = 42):

        target = self.config["target"][0]
        X = self.df.drop(columns=[target])
        y = self.df[target]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)



    def train_model(self):
        
        n = self.config["model"]["n_estimators"]
        self.model = RandomForestRegressor(n_estimators=n, random_state=42)
        self.model.fit(self.X_train, self.y_train)
