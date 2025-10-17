import pandas as pd, string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from constants import *

df = pd.read_csv(RAW_DATA_PATH).dropna().drop_duplicates()
df["label"] = df["label"].map(LABEL_MAP)

df["text"] = df["text"].str.lower() \
    .str.replace(r"\d+", "", regex=True) \
    .str.translate(str.maketrans("", "", string.punctuation)) \
    .str.split().apply(lambda x: " ".join(x))

X_train_text, X_test_text, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df["label"])

vectorizer = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES, ngram_range=TFIDF_NGRAM_RANGE, stop_words=TFIDF_STOP_WORDS)
X_train, X_test = vectorizer.fit_transform(X_train_text), vectorizer.transform(X_test_text)

pd.DataFrame(X_train.toarray(), columns=vectorizer.get_feature_names_out()).assign(label=y_train.values).to_csv(TRAIN_DATA_PATH, index=False)
pd.DataFrame(X_test.toarray(), columns=vectorizer.get_feature_names_out()).assign(label=y_test.values).to_csv(TEST_DATA_PATH, index=False)
