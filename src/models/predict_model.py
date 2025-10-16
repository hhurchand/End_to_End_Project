import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
import mlflow


class MLModelEvaluation:
    """
    A class to evaluate trained machine learning models on a test dataset.

    This class takes multiple fitted models and performs prediction and evaluation
    using accuracy and classification metrics.
    """

    def __init__(self,fitted_models,X_test,y_test):
        """
        Initialize the MLModelEvaluation class.

        Args:
            fitted_models (dict): A dictionary of trained model instances,
                                  where keys are model names and values are the models.
            X_test (scipy.sparse.csr_matrix): Test feature data.
            y_test (pandas.series): True labels for the test data.
        """
        self.fitted_models = fitted_models
        self.X_test = X_test
        self.y_test = y_test


    def predict(self):
        """
        Evaluate each trained model by making predictions on the test data,
        and compute evaluation metrics including Accuracy, Precision, Recall, and F1-score.

        Returns:
            list[dict]: A list of dictionaries, each containing the evaluation metrics
                        for all model.
        """
        results = []

        for name, model in self.fitted_models.items():
            y_pred = model.predict(self.X_test)
            accuracy = accuracy_score(self.y_test, y_pred)
            report = classification_report(self.y_test, y_pred, output_dict=True)
            results.append({
                "Model": name,
                "Accuracy": accuracy,
                "Precision": report["weighted avg"]["precision"],
                "Recall": report["weighted avg"]["recall"],
                "F1-Score": report["weighted avg"]["f1-score"]
            })

            with mlflow.start_run(run_name=f"{name}_evaluation", nested=True):
                mlflow.log_metric("Evaluate_accuracy", accuracy)
                mlflow.log_metric("Evaluate_precision", report["weighted avg"]["precision"])
                mlflow.log_metric("Evaluate_recall", report["weighted avg"]["recall"])
                mlflow.log_metric("Evaluate_f1_score", report["weighted avg"]["f1-score"])
                mlflow.set_tag("model", name)
                mlflow.set_tag("phase", "evaluation")

        df_results = pd.DataFrame(results)

        return df_results


        