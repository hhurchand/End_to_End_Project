import mlflow
from src.utils.input import CSVLoader, YAMLLoader


config = YAMLLoader().load_file("params.yaml")

n_estimators = config["model"]["n_estimators"]
input_filepath = config["data"]["raw_data"]

with mlflow.start_run():

    mlflow.log_param("n_estimators",n_estimators)

df = CSVLoader().load_file(csv_file=input_filepath)


