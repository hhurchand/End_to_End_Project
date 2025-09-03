import mlflow
from src.utils.input import CSVLoader, YAMLLoader
from src.transform import DataTransformation

if __name__ == "__main__":
    config = YAMLLoader().load_file("params.yaml")
    input_filepath = config["data"]["raw_data"]
    df = CSVLoader().load_file(csv_file=input_filepath)
    data_transform_obj = DataTransformation(df,config)
    data_transform_obj.check_input()

# n_estimators = config["model"]["n_estimators"]


# with mlflow.start_run():

    # mlflow.log_param("n_estimators",n_estimators)










