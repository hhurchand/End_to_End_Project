from src.data.make_dataset import DataLoader
from src.data.Specific_dataset_transform import Mod_dataset
from src.features.build_features import DataCleaning
from src.features.build_features import Features
from src.models.train_model import MLModelTraining
from src.models.predict_model import MLModelEvaluation
from src.visualization.visualize import Visualization
from yaml import safe_load
from constants import PARAMS
from pathlib import Path
import pandas as pd


# Load config
with open(PARAMS,encoding="UTF-8") as f:
    config = safe_load(f)

# Load data
df_load = DataLoader(path=Path(config["data"]["Spam_dataset"])).load_xls()
#df_load = pd.read_csv(r"data\processed\cleaned_dataset.csv")


# Special Modified Dataset
mod = Mod_dataset(df_load)
df_processed = mod.process()

#visulization: 
visualizer = Visualization(df=df_processed,label_col="label",text_col="Message")
visualizer.visualize()

# Transformation
# A: lowercase all the data in Message column
df_lower = DataCleaning(df_processed).lower_case([config["transform"]["features"]])

# B: Remove all punctuation not just in one column but in all columns
df_punc = DataCleaning(df_lower).remove_punc([config["transform"]["features"],config["transform"]["encode"]])

# C: Remove all space
df_space = DataCleaning(df_punc).remove_space([config["transform"]["features"]])

# D: Text normalization include: create tokens, remove stopwords, lemmatize tokens, and remove single letter tokens
df_normalize = Features(df_space).text_normalization([config["transform"]["features"]])

# E: Encoding the last columns "label"
df_encode = Features(df_normalize).encoding([config["transform"]["encode"]])


# F: Apply tf-idf to extract text
X, tfidf_vectorize = Features(df_encode).transform_text_num(columns=[config["transform"]["features"]])
y = df_encode[(config["transform"]["encode"])]

# G: Train_Test_Split
prep = MLModelTraining()
X_train,X_test,y_train,y_test = prep.split_data(X,y)

# H: Hyperparameter tuning and model trained
fitted_models = prep.train_hypertuning_cv(X_train, y_train)
prep.save_models(fitted_models,tfidf_vectorize)

# I: Evaluate model
assess = MLModelEvaluation(fitted_models,X_test,y_test)
df_results = assess.predict()
print(df_results)

