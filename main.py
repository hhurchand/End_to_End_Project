from src.utils.input import CSVLoader,YAMLLoader

import mlflow

config = YAMLLoader().load_file("params.yaml")

n_estimators = config["model"]["n_estimators"]
input_filepath = config["data"]["raw_data"]



with mlflow.start_run():

    mlflow.log_param("n_estimators",n_estimators)



df = CSVLoader().load_file(file_path=input_filepath)


