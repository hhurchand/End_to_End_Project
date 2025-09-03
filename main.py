import mlflow
from src.utils.input import csvLoader, yamlLoader

config = yamlLoader().load_file("params.yaml")

n_estimators = config["model"]["n_estimators"]
csv_path = config["data"]["raw_data"]

with mlflow.start_run():
    mlflow.log_param("n_estimators",n_estimators)

csvLoader = csvLoader()
df = csvLoader.load_file(csv_path)

