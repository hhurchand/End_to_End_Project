import pandas as pd
import os


class CSVLoader:
    def __init__(self):
        pass

    def supported_formats(self):
        """Override this in subclasses if you want more formats."""
        return ["csv"]

    def load_file(self, path: str):
        """Loads a CSV file into a Pandas DataFrame."""

        # check extension
        ext = os.path.splitext(path)[1].lower().replace(".", "")
        if ext not in self.supported_formats():
            raise ValueError(f"Unsupported file format: {ext}. Supported: {self.supported_formats()}")

        # check file existence
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

        # load CSV with pandas
        df = pd.read_csv(path)

        # always return DataFrame
        return df
