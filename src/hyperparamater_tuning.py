import mlflow
import mlflow.sklearn
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from src.constants import TRAIN_DATA_PATH, TEST_DATA_PATH
import pandas as pd
from sklearn.metrics import accuracy_score

train_df = pd.read_csv(TRAIN_DATA_PATH)
test_df = pd.read_csv(TEST_DATA_PATH)
X_train, y_train = train_df.drop("label", axis=1), train_df["label"]
X_test, y_test = test_df.drop("label", axis=1), test_df["label"]

param_grid = {"n_estimators": [50, 100, 150], "max_depth": [None, 10, 20]}

with mlflow.start_run(run_name="RandomForest_Tuning"):
    grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3)
    grid.fit(X_train, y_train)
    best = grid.best_estimator_
    acc = accuracy_score(y_test, best.predict(X_test))

    mlflow.log_params(grid.best_params_)
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(best, "Best_RandomForest")

    print("Best Parameters:", grid.best_params_)
    print("Accuracy:", acc)
