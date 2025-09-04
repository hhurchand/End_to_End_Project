from src.utils.input import CSVLoader, YAMLLoader
from src.transform import DataTransformation
import mlflow
import pandas as pd


if __name__ == "__main__":

    # LOAD CONFIG
    the_config = YAMLLoader().load_file("params.yaml")
    the_input_path = the_config["data"]["raw_data"]

    # LOAD THE RAW DATA
    the_dataframe = CSVLoader().load_file(file_path=the_input_path)

    # SAMPLE DATA TO TEST
    the_dataframe = the_dataframe.sample(1000, random_state=42)

    # RUN THE TRANSFORM PIPELINE
    the_transform_object = DataTransformation(the_dataframe, the_config)
    the_encoded_dataframe = the_transform_object.pipeline()


    # THE RESULTS
    print(the_encoded_dataframe.head(10))
    print(the_encoded_dataframe.shape)


    # n_estimators = the_config["model"]["n_estimators"]
    # print("n_estimators:", n_estimators)
    # print("path:", the_input_path)

    # with mlflow.start_run():
    #     mlflow.log_param("n_estimators", n_estimators)
    #     mlflow.log_param("raw_data", the_input_path)

    # print(the_dataframe.head())
    # print("df shape:", the_dataframe.shape)
