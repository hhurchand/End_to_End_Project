from sklearn.linear_model import SGDClassifier

def the_model(parameter=None):
    """
    BUILD LINEAR MODEL
    """
    # DEFAULTS; YAML CAN OVERRIDE ANY
    configure = {"loss": "hinge", "random_state": 42, "max_iter": 1000, "tol": 1e-3}
    if parameter:
        configure.update(parameter.get("models", {}).get("linear", {}))
    return SGDClassifier(**configure)

def fit_and_predict(model, X_train_transform, y_train, X_test_transform):
    """
    FIT MODEL THEN RETURN PREDICTIONS AND PROBABILITIES
    RETURNS: linear_model, y_predict, y_probability
    """
    model.fit(X_train_transform, y_train)
    y_predict = model.predict(X_test_transform)

    # DECISION SCORES
    y_probability = model.decision_function(X_test_transform)
    
    return model, y_predict, y_probability
