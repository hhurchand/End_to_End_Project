import pandas as pd
from scipy import sparse

class CleanedLoader:
    """
    Initializes DataTransformation Object

    Parameters
    -----------
    config: A dictionary of modifiable variables to be referenced
    csvLoader: A util built to facilitate reading CSV files
    """
    def __init__(self, config, csvLoader):
        self.config = config
        self.csvLoader = csvLoader

    """
    Loads the email messages that have been stemmed and tokenized
    """
    def load_X_Sparse(self):
        print("Load x sparse")
        X_sparse_path = self.config["data"]["X_sparse"]
        self.X = sparse.load_npz(X_sparse_path)

    """
    Loads the Spam/Ham field that has been stored as boolean values
    """
    def load_cleaned_y(self):
        
        print("Load cleaned y")      
        cleaned_y_path = self.config["data"]["cleaned_y"]
        self.y = self.csvLoader.load_file(cleaned_y_path)

    """
    Load X Sparse and Cleaned y
    """
    def load_cleaned_data(self):
        print("Load cleaned Data")
        self.load_X_Sparse()
        self.load_cleaned_y()

        return self.X, self.y