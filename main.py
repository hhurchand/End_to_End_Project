from src.utils.input import CSVLoader, YAMLLoader
from src.transform_data import DataTransformation  # <-- make sure this file exists

if __name__ == "__main__":
    # 1) Load config
    config = YAMLLoader().load_file("params.yaml")

    # 2) Load CSV
    input_filepath = config["data"]["raw_data"]
    df = CSVLoader().load_file(file_path=input_filepath)

    # 3) Run transformation pipeline
    transformer = DataTransformation(df, config)
    X, y, pipeline = transformer.transformation_pipeline()

    print("X shape:", X.shape)
    print("y shape:", y.shape if y is not None else None)
    print("Done.")
