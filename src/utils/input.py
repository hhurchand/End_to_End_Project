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
    """Loads json file"""
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
    """
    Load a csv file.  Read supported formats to get supported formats.

        Returns:
            Any: The loaded file content as a DataFrame

        Raises:
            FileNotFoundError: If the file doesn't exist
       
    """

    def load_file(self, file_path) -> pd.DataFrame:
        try:
            with open (file_path) as csv_file:
                return pd.read_csv(csv_file)
            
        except FileNotFoundError as e:
            logger.error(f"{e} : {file_path}")
 
    def supported_formats(self) -> List[str]:

        return ['.csv']
    

class YAMLLoader(FileLoader):

    def load_file(self,file_path:Union[str,Path])->Dict:
        """Load a file and return its contents.

        Args:
            file_path (Union[str,Path]): Path to the file. 

        Returns:
            Any: The loaded file content

        Raises:l
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file format is not supported
        
        """
        try:
            file_path = Path(file_path)
            with open (file_path) as yaml_file:
                return safe_load(yaml_file)
                
        except FileNotFoundError as e:
            logger.error(f"{e} : {file_path}")                

    def supported_formats(self) -> List[str]:

        return ['.yaml','.yml']
