
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB

import mlflow
import mlflow.sklearn

import pickle

class Model:

    """
    Initializes Model Object

    Parameters
    -----------
    config: A dictionary of modifiable variables to be referenced
    X: The stemmed, tokenized email contents
    y: The Spam/Ham field converted to a boolean value (1/0)
    """
    def __init__(self, config, X, y):
            self.config = config
            self.X = X
            self.y = y

    
    def train_test_split(self):
        """
        Split the X and y values to be used as training and testing data for the model
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=42)

    
    def train_logistic_regression(self):
        """
        Use the training and testing data to train a logistic regression model
        """
        print("Train logistic regression")
        # Start an MLflow run
        with mlflow.start_run():
           # Train a logistic regression classifier
            self.classifier = LogisticRegression()
            self.classifier.fit(self.X_train, self.y_train)

            mlflow.sklearn.log_model(
                sk_model=self.classifier,
                name="logistic_regression_model", # Path within the run artifacts
                registered_model_name="spam-ham-logistic-regression-model", # Name for the Model Registry
            )

    
    def train_random_forest(self):
        """
        Use the training and testing data to train a Random Forest model
        """
        print("Train random forest")
        with mlflow.start_run():
            self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
            self.classifier.fit(self.X_train, self.y_train)

            mlflow.sklearn.log_model(
                sk_model=self.classifier,
                name="random_forest_model", # Path within the run artifacts
                registered_model_name="spam-ham-random-forest-model", # Name for the Model Registry
            )

    
    def train_naive_bayes(self):
        """
        Use the training and testing data to train a Naive Bayes model
        """
        print("Train Naive Bayes")
        with mlflow.start_run():
            self.classifier = MultinomialNB()
            self.classifier.fit(self.X_train, self.y_train)

            mlflow.sklearn.log_model(
                sk_model=self.classifier,
                name="naive_bayes_model", # Path within the run artifacts
                registered_model_name="spam-ham-naive-bayes-model", # Name for the Model Registry
            )

    
    def predict(self):
        """
        Test the trained model against the X_test testing data and store the result in y_pred
        """
        self.y_pred = self.classifier.predict(self.X_test)

    
    def evaluate(self):
        """
        Evaluate the results of the y_pred predictions against the actual y_test data
        """
        with mlflow.start_run():
            
            # Evaluate the accuracy of the classifier
            accuracy = accuracy_score(self.y_test, self.y_pred)
            print("Accuracy:", accuracy)
            mlflow.log_metric("accuracy", str(accuracy))

            class_report = classification_report(self.y_test, self.y_pred, output_dict=True)
            print(class_report)
            mlflow.log_metric("precision", class_report['0']['precision'])
            mlflow.log_metric("recall", class_report['0']['recall'])
            mlflow.log_metric("f1-score", class_report['0']['f1-score'])
            

            conf_matrix = confusion_matrix(self.y_test, self.y_pred)
            print(conf_matrix)

    
    def save_pickle_file(self):
        """
        Save the trained model to a pickle file to be used in the streamlit application
        """
        print("Save model to pickle file")

        filename = self.config["data"]["pickle_file"]
        with open(filename, 'wb') as file:
            # Use pickle.dump() to serialize the object and write it to the file
            pickle.dump(self.classifier, file)

        print(f"Model saved to {filename}")


    