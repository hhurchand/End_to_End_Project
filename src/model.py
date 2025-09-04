import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

class Model:
    def __init__(self, df: pd.DataFrame, target_column: str):
        self.df = df
        self.target = target_column

    def train_test_split(self, test_size=0.2, random_state=42):
        X = self.df.drop(columns=[self.target])
        y = self.df[self.target]
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

    def train_model(self, X_train, X_test, y_train, y_test, n_estimators=100):
        model = RandomForestRegressor(n_estimators=n_estimators)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return mean_squared_error(y_test, y_pred)

