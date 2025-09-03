import mlflow
from src.utils.input import csvLoader, yamlLoader
from src.transform import DataTransformation

# n_estimators = config["model"]["n_estimators"]
# with mlflow.start_run():
#     mlflow.log_param("n_estimators",n_estimators)

if __name__ == "__main__":
    config = yamlLoader().load_file("params.yaml")

    csv_path = config["data"]["raw_data"]

    csvLoader = csvLoader()
    df = csvLoader.load_file(csv_path)

    dataTransformObject = DataTransformation(df,config)
    dataTransformObject.check_input()