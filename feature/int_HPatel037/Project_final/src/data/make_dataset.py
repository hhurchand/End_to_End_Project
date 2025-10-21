# src/data/make_dataset.py
from pathlib import Path
import pandas as pd

class DataLoader:
    """Minimal CSV loader used by tri_model_trainer.py"""
    def __init__(self, path: Path):
        self.path = Path(path)

    def load_csv(self):
        return pd.read_csv(self.path)
