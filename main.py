import pandas as pd
import pickle
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from src.constants import TRAIN_DATA_PATH, TEST_DATA_PATH

def load_data():
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    test_df = pd.read_csv(TEST_DATA_PATH)
    X_train, y_train = train_df.drop("label", axis=1), train_df["label"]
    X_test, y_test = test_df.drop("label", axis=1), test_df["label"]
    return X_train, X_test, y_train, y_test

def train_and_log_model(name, model, X_train, y_train, X_test, y_test):
    with mlflow.start_run(run_name=name):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        mlflow.log_param("model", name)
        mlflow.log_metric("accuracy", acc)

        print(f"=== {name} ===")
        print(f"Accuracy: {acc:.4f}")
        print(classification_report(y_test, y_pred))

        pickle.dump(model, f"models/{name}.pickle")
        mlflow.sklearn.log_model(model, name)

def main():
    X_train, X_test, y_train, y_test = load_data()

    models = {
        "Logistic_Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random_Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Multinomial_NB": MultinomialNB(alpha=1.0)
    }

    for name, model in models.items():
        train_and_log_model(name, model, X_train, y_train, X_test, y_test)

if __name__ == "__main__":
    main()
