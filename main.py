from src.utils.input import CSVLoader,YAMLLoader
from src.transform import DataTransformation

import mlflow
if __name__ == "__main__": 
    config = YAMLLoader().load_file("params.yaml")

    n_estimators = config['model']['n_estimators']
    input_filepath = config['data']['raw_data']



    with mlflow.start_run():
        
        mlflow.log_param("n_estimators",n_estimators)
        
        
        
    df = CSVLoader().load_file(file_path=input_filepath)


    data_transform_obj = DataTransformation(config,df)

    data_transform_obj.transform_data()

                        