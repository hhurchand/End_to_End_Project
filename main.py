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



    train_logistic_regression = config["settings"]["train_logistic_regression"]
    train_random_forest = config["settings"]["train_random_forest"]
    train_naive_bayes = config["settings"]["train_naive_bayes"]

    save_pickle_file = config["settings"]["save_pickle_file"]

    model = Model(config, X, y)
    model.train_test_split()

    if train_logistic_regression:
        model.train_logistic_regression()
        model.predict()
        model.evaluate()


    if train_random_forest:
        model.train_random_forest()
        model.predict()
        model.evaluate()


    if train_naive_bayes:
        model.train_naive_bayes()
        model.predict()
        model.evaluate()
    

    if save_pickle_file:
        model.save_pickle_file()

    

