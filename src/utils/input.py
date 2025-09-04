from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Union

import json
import pandas as pd
import yaml


class FileLoader(ABC):
    @abstractmethod
    def load_file(self, file_path: Union[str, Path]) -> Any:
        """Read a file and return its content."""
        raise NotImplementedError

    @abstractmethod
    def supported_formats(self) -> List[str]:
        """List of supported extensions (e.g., ['.csv'])."""
        raise NotImplementedError


class CSVLoader(FileLoader):
    """Read CSV into a pandas DataFrame."""

    def load_file(self, file_path: Union[str, Path]) -> pd.DataFrame:
        p = Path(file_path)
        if not p.exists():
            raise FileNotFoundError(f"CSV not found: {p}")
        return pd.read_csv(p)

    def supported_formats(self) -> List[str]:
        return [".csv"]


class JSONLoader(FileLoader):
    """Read JSON into a dict."""

    def load_file(self, file_path: Union[str, Path]) -> Dict:
        p = Path(file_path)
        if not p.exists():
            raise FileNotFoundError(f"JSON not found: {p}")
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)

    def supported_formats(self) -> List[str]:
        return [".json"]


class YAMLLoader(FileLoader):
    """Read YAML into a dict."""

    def load_file(self, file_path: Union[str, Path]) -> Dict:
        p = Path(file_path)
        if not p.exists():
            raise FileNotFoundError(f"YAML not found: {p}")
        with open(p, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def supported_formats(self) -> List[str]:
        return [".yaml", ".yml"]