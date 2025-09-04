from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union, List, Dict, Any
from src.utils.logger_file import logger
import pandas as pd
import json
from yaml import safe_load

class FileLoader(ABC):
    @abstractmethod
    def load_file(self, file_path: Union[str, Path]) -> Any:
        """Load a file and return its contents."""
        pass

    @abstractmethod
    def supported_formats(self) -> List[str]:
        """Return list of file formats supported by this loader."""
        pass

    def _resolve_path(self, file_path: Union[str, Path]) -> Path:
        """Resolve the path and check if it exists."""
        path = Path(file_path).resolve()
        if not path.exists():
            msg = f"File not found: {path}"
            logger.error(msg)
            raise FileNotFoundError(msg)
        return path

class JSONLoader(FileLoader):
    def load_file(self, file_path: Union[str, Path]) -> Dict:
        path = self._resolve_path(file_path)
        with open(path, 'r') as f:
            return json.load(f)

    def supported_formats(self) -> List[str]:
        return ['.json']

class CSVLoader(FileLoader):
    def load_file(self, file_path: Union[str, Path]) -> pd.DataFrame:
        path = self._resolve_path(file_path)
        return pd.read_csv(path)

    def supported_formats(self) -> List[str]:
        return ['.csv']

class YAMLLoader(FileLoader):
    def load_file(self, file_path: Union[str, Path]) -> Dict:
        path = self._resolve_path(file_path)
        with open(path) as f:
            return safe_load(f)

    def supported_formats(self) -> List[str]:
        return ['.yaml', '.yml']
