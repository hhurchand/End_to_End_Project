
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

class Model:

    def __init__(self, config, X, y):
            self.config = config
            self.X = X
            self.y = y

    def train_test_split(self):
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=42)

    def train_logistic_regression(self):
           # Train a logistic regression classifier
        self.classifier = LogisticRegression()
        self.classifier.fit(self.X_train, self.y_train)

    def predict(self):
        self.y_pred = self.classifier.predict(self.X_test)

    def evaluate(self):
        # Evaluate the accuracy of the classifier
        accuracy = accuracy_score(self.y_test, self.y_pred)
        print("Accuracy:", accuracy)

        class_report = classification_report(self.y_test, self.y_pred)
        print(class_report)

        conf_matrix = confusion_matrix(self.y_test, self.y_pred)
        print(conf_matrix)
