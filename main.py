from src.utils.input import CSVLoader, YAMLLoader
from src.transform import DataTransformation
from src.model import Model

if __name__ == "__main__":
    config = YAMLLoader().load_file("params.yaml")
    input_filepath = config["data"]["raw_data"]
    df = CSVLoader().load_file(csv_file=input_filepath)
    data_transform_obj = DataTransformation(df,config)
    transform_data = data_transform_obj.ohencoder()
    transform_data = data_transform_obj.fit_transform_obj()
    df_encoded = data_transform_obj.features_cleaning()
    df_encoded = data_transform_obj.mapping()
    df_train = Model(df_encoded,config=config)
    df_split = df_train.split_data(X = df_encoded.drop(config["model"]["target"],axis=1),y = df_encoded[config["model"]["target"]],test_size=0.3,random_state=42)
    df_model = df_train.modeling(random_state=42)
    df_predict = df_train.prediction()


# n_estimators = config["model"]["n_estimators"]


# with mlflow.start_run():

    # mlflow.log_param("n_estimators",n_estimators)










