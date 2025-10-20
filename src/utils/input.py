from pathlib import Path
from typing import Dict, Union
import pandas as pd

from yaml import safe_load
from src.utils.logger import logger


class CSV:
    def load(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """
        LOAD CSV FILE INTO DATAFRAME
        """
        file_path = Path(file_path)
        try:
            return pd.read_csv(file_path)
        except FileNotFoundError as e:
            logger.error(f"{e}: {file_path}")


    def save(self, df: pd.DataFrame, file_path: Union[str, Path]) -> None:
        """
        SAVE DATAFRAME TO CSV
        """
        file_path = Path(file_path)
        try:
            df.to_csv(file_path, index=False)
        except Exception as e:
            logger.error(f"{e}: {file_path}")


class YAML:
    def load(self, file_path: Union[str, Path]) -> Dict:
        """
        LOAD YAML FILE INTO DICTIONARY
        """
        file_path = Path(file_path)
        try:
            with open(file_path, "r") as yaml_file:
                return safe_load(yaml_file)
        except FileNotFoundError as e:
            logger.error(f"{e}: {file_path}")

