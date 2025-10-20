from sklearn.feature_extraction.text import TfidfVectorizer

def vectorizer(parameter: dict) -> TfidfVectorizer:
    """
    BUILD TF-IDF VECTORIZER FROM PARAMS
    UNIGRAM_MIN | nA, UNIGRAM_MAX | nB, MIN_DF, MAX_DF
    """
    nA = parameter["tfidf"]["unigram_min"]
    nB = parameter["tfidf"]["unigram_max"]
    min_df = parameter["tfidf"]["min_df"]
    max_df = parameter["tfidf"]["max_df"]
    return TfidfVectorizer(ngram_range=(nA, nB), min_df=min_df, max_df=max_df)

def the_vectorize_fit(X_train, X_test, the_vec: TfidfVectorizer):
    """
    FIT ON TRAIN THEN TRANSFORM TRAIN / TEST
    RETURNS: X_TRAIN_T, X_TEST_T
    """
    X_train_t = the_vec.fit_transform(X_train)
    X_test_t  = the_vec.transform(X_test)
    return X_train_t, X_test_t
