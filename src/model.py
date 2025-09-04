import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


class Model:
    def __init__(self, df_encoded: pd.DataFrame):
        features =list(df_encoded.columns)
        features.remove("price")
        self.X, self.y = df_encoded[features], df_encoded["price"]
        self.model = RandomForestRegressor(random_state=42)

    def train_test_split(self):
        return train_test_split(self.X, self.y, test_size=0.3, random_state=42)

    def train_model(self, X_train, X_test, y_train, y_test):
        self.model.fit(X_train, y_train)
        preds = self.model.predict(X_test)
        mse = mean_squared_error(y_test, preds)
        print("Model trained successfully!")
        print("Mean Squared Error (MSE):", mse)
        return preds, mse
