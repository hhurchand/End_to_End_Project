from src.utils.input import CSVLoader, YAMLLoader
import mlflow

config = YAMLLoader().load_file("params.yaml")

n_estimate = config["model"]["n_estimators"]

with mlflow.start_run():
    mlflow.log_param("n_estimators",n_estimate)


input_path = config["data"]["raw_data"]
df = CSVLoader().load_file(input_path)



 # -------------------Testing -------------------------------------------------------- #
# print("n_estimators = ",n_estimate)
# print("path = ",input_path)
# print("df",df)

