from src.utils.input import CSVLoader, YAMLLoader
from src.transform import DataTransformation

import mlflow

config = YAMLLoader().load_file("params.yaml")

n_estimators = config["model"]["n_estimators"]
input_filepath = config["data"]["raw_data"]

with mlflow.start_run():
    mlflow.log_param("n_estimators", n_estimators)

# Load raw data
df = CSVLoader().load_file(file_path=input_filepath)

# Initialize transformation
data_transform_obj = DataTransformation(config)

# Apply transformation
X, y = data_transform_obj.fit_transform(df)

print("Transformed features shape:", X.shape)
print("Target shape:", y.shape)
