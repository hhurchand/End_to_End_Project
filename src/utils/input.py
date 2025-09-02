from abc import ABC,abstractmethod
from pathlib import Path
from typing import Union,List, Dict,Any
from src.utils.logger_file import logger
import pandas as pd
import json
from yaml import safe_load

class FileLoader(ABC):
    @abstractmethod
    def load_file(self,file_path:Union[str,Path])->Any:
        """Load a file and return its contents.

        Args:
            file_path (Union[str,Path]): Path to the file. 

        Returns:
            Any: The loaded file content

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file format is not supported
        
        """
        pass

    @abstractmethod
    def supported_formats(self) -> List[str]:
        """
        Return list of file formats supported by this loader.
        
        Returns:
            List of supported file extensions (e.g., ['.txt', '.csv'])
        """
        pass

class JSONLoader(FileLoader):
    """Loads json file

        Args:
            file_path (Union[str,Path]): Path to the file. 

        Returns:
            Any: The loaded file content

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file format is not supported
        
        """
    def load_file(self, file_path: Union[str,Path]) -> Dict:
        file_path = Path(file_path)
        try:
            with open(file_path,'r') as json_file:
                return json.load(json_file)
        
        except FileNotFoundError as e:
            logger.error(f"{e} : {file_path}")

    def supported_formats(self) -> List[str]:

        return ['.json']

class CSVLoader(FileLoader):
    """Loads a CSV file. Read supported_formats method to get supported formats
        
        Args:
            file_path (Union[str,Path]): Path to the file. 

        Returns:
            Any: The loaded file content

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file format is not supported
        
        """
    def load_file(self, csv_file: Union[str,Path]) -> pd.DataFrame:
        file_path = Path(csv_file)
        try:
            with open(csv_file) as csv_file:
                return pd.read_csv(csv_file)
        
        except FileNotFoundError as e:
            logger.error(f"{e} : {csv_file}")

    def supported_formats(self) -> List[str]:

        return ['.csv']
    
class YAMLLoader(FileLoader):
    def load_file(self,file_path:Union[str,Path])->Dict:
        """Load a file and return its contents.

        Args:
            file_path (Union[str,Path]): Path to the file. 

        Returns:
            Any: The loaded file content

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file format is not supported
        
        """
        try:
            with open(file_path) as yamal_file:
                return safe_load(yamal_file)
        except FileNotFoundError as e:
            logger.error(f"{e} : {file_path}")

    def supported_formats(self) -> List[str]:

        return ['.yaml','yml']