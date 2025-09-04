from pathlib import Path
from src.utils.input import CSVLoader, YAMLLoader
from src.transform import DataTransformation
from src.model import Model

def main():
    config = YAMLLoader().load_file("params.yaml")
    df = CSVLoader().load_file("data/airlines_flights_data.csv")
    transformer = DataTransformation(df, config)
    df_transformed = transformer.transformation_pipeline()
    target_column = config['target']['target_column'][0]
    n_estimators = config['model']['n_estimators']
    model = Model(df_transformed, target_column)
    X_train, X_test, y_train, y_test = model.train_test_split()
    mse = model.train_model(X_train, X_test, y_train, y_test, n_estimators=n_estimators)
    print("Test MSE:", mse)

if __name__ == "__main__":
    main()
