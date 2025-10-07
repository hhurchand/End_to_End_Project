from src.utils.input import csvLoader, yamlLoader

if __name__ == "__main__":
    config = yamlLoader().load_file("params.yaml")