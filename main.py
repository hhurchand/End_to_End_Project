import mlflow
from src.utils.input import csvLoader

with mlflow.start_run():
    n_estimators = 100
    mlflow.log_param("n_estimators",n_estimators)

csvLoader = csvLoader()
df = csvLoader.load_file("data/raw/airlines_flights_data.csv")
print(df)