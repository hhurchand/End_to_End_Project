import mlflow

with mlflow.start_run():
    n_estimators = 100
    mlflow.log_param("n_estimators",n_estimators)