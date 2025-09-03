from src.utils.input import CSVLoader, YAMLLoader
from src.transform import DataTransformation
import mlflow



# n_estimators = config["model"]["n_estimators"]


# print("n_estimators:", n_estimators)
# print("path:", input_filepath)

# with mlflow.start_run():

#    mlflow.log_param("n_estimators", n_estimators)
#    mlflow.log_param("raw_data", input_filepath)


# print(df.head())
# print("df shape:", df.shape)



if __name__ == "__main__":

    config = YAMLLoader().load_file("params.yaml")
    input_filepath = config["data"]["raw_data"]

    df = CSVLoader().load_file(file_path=input_filepath)

    the_transform_object = DataTransformation(df,config)


 

