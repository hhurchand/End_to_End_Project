import pandas as pd
from scipy import sparse

class CleanedLoader:
    def __init__(self, config, csvLoader):
        self.config = config
        self.csvLoader = csvLoader

    def load_X_Sparse(self):
        print("Load x sparse")
        X_sparse_path = self.config["data"]["X_sparse"]
        self.X = sparse.load_npz(X_sparse_path)

    def load_cleaned_y(self):
        
        print("Load cleaned y")      
        cleaned_y_path = self.config["data"]["cleaned_y"]
        self.y = self.csvLoader.load_file(cleaned_y_path)

    def load_cleaned_data(self):
        print("Load cleaned Data")
        self.load_X_Sparse()
        self.load_cleaned_y()

        return self.X, self.y