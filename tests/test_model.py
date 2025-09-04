import numpy as np
import pandas as pd
from src.model import Model

def make_df(n=100, seed=0):
    rng = np.random.default_rng(seed)
    # Fake encoded feature columns + target "price"
    df = pd.DataFrame({
        "duration": rng.normal(120, 30, n),
        "days_left": rng.integers(0, 60, n),
        "airline_AA": rng.integers(0, 2, n),
        "class_Business": rng.integers(0, 2, n),
    })
    # Target roughly depends on features + noise
    price = 200 + 0.8*df["duration"] - 1.5*df["days_left"] + 50*df["class_Business"] + rng.normal(0, 10, n)
    df["price"] = price
    return df

def test_train_test_split_shapes():
    df = make_df(n=120)
    mdl = Model(df)
    X_train, X_test, y_train, y_test = mdl.train_test_split()
    assert len(X_train) + len(X_test) == len(df)
    assert len(y_train) + len(y_test) == len(df)

def test_train_and_mse_is_number():
    df = make_df(n=120)
    mdl = Model(df)
    X_train, X_test, y_train, y_test = mdl.train_test_split()
    preds, mse = mdl.train_model(X_train, X_test, y_train, y_test)
    assert len(preds) == len(y_test)
    assert isinstance(mse, (int, float, np.floating))
