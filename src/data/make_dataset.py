import pandas as pd
from pathlib import Path

class DataLoader:

    """
    Data loader class for loading file

    It takes a filepath as input and through a method it will load the file
    """


    def __init__(self,path:str):
        """
        Initialize the Dataloader class

        Args:
            path (str): file location of the dataset in the cookiecutter template
        """
        self.path = path


    def load_csv(self):
        """
        load the csv file into a dataframe

        Returns:
            pandas.DataFrame: The loaded csv file
        """
        return pd.read_csv(self.path)
    
    def load_xls(self):
        """
        load the xls file into a dataframe

        Returns:
            pandas.DataFrame: The loaded xls file
        """
        return pd.read_excel(self.path)
