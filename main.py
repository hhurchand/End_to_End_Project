from src.utils.input import CSVLoader, YAMLLoader

import mlflow

with mlflow.start_run():
    n_estimators = 100
    mlflow.log_param("n_estimators", n_estimators)

df = CSVLoader().load_file("C:/Users/Owner/OneDrive/Desktop/GitProject/End_to_End_Project/data/raw/airlines_flights_data.csv")

config = YAMLLoader().load_file("params.yaml")
print(config)
