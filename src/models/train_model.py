import mlflow
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression



def tune_with_mlflow(model_name, estimator, params, Xtr, y_train, Xte, y_test):
    """Run small GridSearch with MLflow logging."""
    with mlflow.start_run(run_name=f"tune_{model_name}"):
        gs = GridSearchCV(estimator, params, cv=3)
        gs.fit(Xtr, y_train)
        best_model = gs.best_estimator_
        acc = best_model.score(Xte, y_test)
        mlflow.log_params(gs.best_params_)
        mlflow.log_metric("accuracy", acc)
        print(f"🔧 Tuned {model_name} best params: {gs.best_params_}")
        return best_model


def tune_models(Xtr, Xte, y_train, y_test):
    """Tune three models using MLflow."""
    mlflow.set_experiment("spam_classifier")

    tuned = {}
    tuned["LinearSVC"] = tune_with_mlflow(
        "LinearSVC", LinearSVC(), {"C": [0.01, 0.1, 1]}, Xtr, y_train, Xte, y_test
    )
    tuned["MultinomialNB"] = tune_with_mlflow(
        "MultinomialNB", MultinomialNB(), {"alpha": [0.1, 1, 10]}, Xtr, y_train, Xte, y_test
    )
    tuned["LogReg"] = tune_with_mlflow(
        "LogReg", LogisticRegression(max_iter=1000), {"C": [0.1, 1, 10], "solver": ["liblinear"]},
        Xtr, y_train, Xte, y_test
    )

    print("✅ MLflow tuning complete. Check UI for results.")
    return tuned


def def_lin(Xtr, Xte, y_train, y_test):
    clf = LinearSVC()
    clf.fit(Xtr, y_train)
    preds = clf.predict(Xte)
    acc = accuracy_score(y_test, preds)
    return "LinearSVC", acc, preds, clf

def def_nb(Xtr, Xte, y_train, y_test):
    clf = MultinomialNB()
    clf.fit(Xtr, y_train)
    preds = clf.predict(Xte)
    acc = accuracy_score(y_test, preds)
    return "MultinomialNB", acc, preds, clf

def def_log(Xtr, Xte, y_train, y_test):
    clf = LogisticRegression(max_iter=1000)
    clf.fit(Xtr, y_train)
    preds = clf.predict(Xte)
    acc = accuracy_score(y_test, preds)
    return "LogReg", acc, preds, clf
