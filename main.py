from src.utils.input import CSVLoader, YAMLLoader
from src.transform import DataTransformation
import mlflow

config = YAMLLoader().load_file("params.yaml")

n_estimate = config["model"]["n_estimators"]

with mlflow.start_run():
    mlflow.log_param("n_estimators",n_estimate)


input_path = config["data"]["raw_data"]
df = CSVLoader().load_file(input_path)

transform = DataTransformation(df,config)
categorical_feature = transform.get_categorical_features()
#print(categorical_feature)

df_update = transform.data_transform(categorical_feature)
df_encoded = transform.data_encode(df_update)
print (df_encoded)



 # -------------------Testing -------------------------------------------------------- #
# print("n_estimators = ",n_estimate)
# print("path = ",input_path)
# print("df",df)


