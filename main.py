from src.utils.input import csvLoader, yamlLoader
from src.transform import DataTransformation
from src.cleaned_loader import CleanedLoader
from src.model import Model

if __name__ == "__main__":
    config = yamlLoader().load_file("params.yaml")

    
    clean_data = config["settings"]["clean_data"]

    csvLoaderObject = csvLoader()
    if clean_data:
        csv_path = config["data"]["raw_data"]

        df = csvLoaderObject.load_file(csv_path)

        dataTransformObject = DataTransformation(df,config)
        X, y = dataTransformObject.transform_data_pipeline()

        
    else:
        cleanedLoader = CleanedLoader(config,csvLoaderObject)

        X, y = cleanedLoader.load_cleaned_data()

    model = Model(config, X, y)
    model.train_test_split()
    model.train_logistic_regression()
    model.predict()
    model.evaluate()


    

