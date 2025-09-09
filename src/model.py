import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

class Model:
    def __init__(self, df: pd.DataFrame, target: str):
        self.df = df
        self.target = target

    def train_test_split(self, test_size: float = 0.2, random_state: int = 42):
        X = self.df.drop(columns=[self.target])
        y = self.df[self.target]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

    def train_model(self):
        self.model = RandomForestRegressor(random_state=42)
        self.model.fit(self.X_train, self.y_train)
