from sklearn.linear_model import LogisticRegression

def the_model(parameter=None):
    """
    BUILD LOGISTIC REGRESSION MODEL
    """
    # YAML HYPERPARAMETERS
    configure = parameter.get("models", {}).get("logistic", {}) if parameter else {}
    return LogisticRegression(**configure)

def fit_and_predict(model, X_train_transform, y_train, X_test_transform):
    """
    FIT MODEL THEN RETURN PREDICTIONS AND PROBABILITIES
    RETURNS: logistic_model, y_predict, y_probability
    """
    model.fit(X_train_transform, y_train)
    y_predict = model.predict(X_test_transform)
    y_probability = model.predict_proba(X_test_transform)[:, 1]
    
    return model, y_predict, y_probability
