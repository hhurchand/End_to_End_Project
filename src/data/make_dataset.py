
import pandas as pd

class DataLoader:
  """Methods to load files
  """
  def __init__(self,path:str):
    self.path = path

  def load_csv(self):
    return pd.read_csv(self.path)

  def load_json(self):
    return pd.read_json(self.path)

  def load_excel(self):
    return pd.read_excel(self.path)

  def repr(self):
    return f"DataLoader(path={self.path}"