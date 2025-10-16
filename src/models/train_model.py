from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import pickle
from pathlib import Path
import mlflow
import mlflow.sklearn

from yaml import safe_load
from constants import PARAMS
from pathlib import Path

with open(PARAMS,encoding="UTF-8") as f:   
    """
    Load configuration parameters from a YAML file specified by PARAMS.

    The configuration is loaded once and stored in `config`.
    """
    config = safe_load(f)


class MLModelTraining:
    """
    A class to handle training and hyperparameter tuning of multiple machine learning models.

    This class includes methods to:
    - Split a dataset into training and testing sets
    - Perform hyperparameter tuning using GridSearchCV
    - Train and return the best models for each selected algorithm

    """

    def __init__(self):
        """
        Initialized the MLDataprep class

        """
        pass
  
    def split_data(self,X,y):
        """
        Split the input features and labels into training and testing sets.

        The split is controlled by parameters defined in the external config file:
        - test_size: proportion of the dataset to include in the test split
        - random_state: random seed for reproducibility

        Parameters:
            X (scipy.sparse.csr_matrix): Input features.
            y (pandas.series): Target labels.

        Returns:
            tuple: X_train, X_test, y_train, y_test
        """
        test_size = config["model"]["test_size"]
        random_state = config["model"]["random_state"]
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=test_size,random_state=random_state)
        return X_train,X_test,y_train,y_test
    
    def train_hypertuning_cv(self,X_train,y_train,):
        """
        Perform hyperparameter tuning and training using GridSearchCV
        for each model defined in the config.

        For each model listed in the config, the method:
        - Loads the corresponding model instance
        - Applies GridSearchCV using the specified parameter grid and 5-fold CV
        - Trains the model on the training data
        - Selects and stores the best performing model
        - Logs parameters, metrics and trained model to mlflow for tracking

        Parameters:
            X_train (scipy.sparse.csr_matrix): Training feature data.
            y_train (pandas.series): Training target labels.

        Returns:
            dict: A dictionary where keys are model names (str), and values are
                  the best trained model instances.
        """
        models = config["model"]["models"]
        param_grids = config["param_grids"]
        fitted_models = {}

        model_map = {
            "RandomForestClassifier": RandomForestClassifier(random_state=42),
            "MultinomialNB": MultinomialNB(),
            "LogisticRegression": LogisticRegression(max_iter=1000)}

        for name,model in model_map.items():
            if name in models:
                with mlflow.start_run(run_name=f"{name}_training") as run:
                    grid = GridSearchCV(model, param_grids[name], cv=5)
                    grid.fit(X_train, y_train)

                    best_params = grid.best_params_
                    best_score = grid.best_score_

                    mlflow.log_params(best_params)
                    mlflow.log_metric("best_cv_score", best_score)
                    mlflow.set_tag("model", name)
                    mlflow.set_tag("Coder", "AA")

                    best_model = model.__class__(**best_params)
                    best_model.fit(X_train,y_train)

                    fitted_models[name] = best_model
                    
                    mlflow.sklearn.log_model(best_model, artifact_path=name)

        return fitted_models

    def save_models(self,fitted_models,tfidf_vectorize):

        """
        Saves each trained model in the fitted_models dictionary to a pickle file using joblib.

        Args:
            fitted_models (dict): Dictionary with model names as keys and trained model instances as values.
            vectorizer
        """
        save_path = Path(r"D:\Users\desk\Documents\Artificial Intelligence Specialist - LEA.E3\420-PIA-1D - Integration project\integrated_project_spam\models")
        
        for model_name, model in fitted_models.items():
            filename = save_path.joinpath(f"{model_name}.pkl")
            with open(filename, "wb") as f:
                pickle.dump(model,f)

        vectorize_file = save_path.joinpath("tfidf_vectorize.pkl")
        with open(vectorize_file, "wb") as f:
            pickle.dump(tfidf_vectorize, f)
   

