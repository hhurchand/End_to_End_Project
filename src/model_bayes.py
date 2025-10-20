from sklearn.naive_bayes import MultinomialNB

def the_model(parameter=None):
    """
    BUILD MULTINOMIAL NAIVE BAYES
    """
    # YAML HYPERPARAMETERS
    configure = parameter.get("models", {}).get("bayes", {}) if parameter else {}
    return MultinomialNB(**configure)

def fit_and_predict(bayes_model, X_train_transform, y_train, X_test_transform):
    """
    TRAIN THE MODEL AND MAKE PREDICTIONS
    A. FIT THE MODEL ON TRAINING DATA
    B. PREDICT LABELS ON TEST DATA
    C. RETURNS: bayes_model, y_predict, y_probability
    """
    bayes_model.fit(X_train_transform, y_train)
    y_predict = bayes_model.predict(X_test_transform)

    # CLASS PROBABILITY
    y_probability = bayes_model.predict_proba(X_test_transform)[:, 1]
    
    return bayes_model, y_predict, y_probability
