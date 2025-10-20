import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.input import YAML, CSV
from src.vectorize import vectorizer, the_vectorize_fit


def sample_vector(X_train_transform, y_train, n_features=30, n_rows=10):
    """
    PREVIEW OF THE VECTORIZED DATAFRAME | SAMPLE
    SHOWS FIRST n_rows AND FIRST n_features AS v1, v2, v3...
    """
    # DENSE SAMPLE
    dense_sample = X_train_transform[:n_rows].toarray()[:, :n_features]

    # COLUMN NAMES
    feature_names = [f"v{i+1}" for i in range(dense_sample.shape[1])]

    # BUILD THE DATAFRAME
    df_preview = pd.DataFrame(dense_sample, columns=feature_names)
    df_preview.insert(0, "label", y_train[:n_rows])

    print("\nVECTOR DATAFRAME SAMPLE")
    print(df_preview)
    print("\nSHAPE (SAMPLE):", df_preview.shape)
    print(f"1st {n_rows} ROWS × 1st {n_features} FEATURES.")
    return df_preview


if __name__ == "__main__":
    
    parameter = YAML().load("params.yaml")

    # LOAD CLEANED DATA
    df_clean = CSV().load(parameter["data"]["clean"])

    # COLUMNS
    text_column = parameter["text"]
    label_column = parameter["label"]

    # FEATURES & TARGET
    X_text = df_clean[text_column].values
    y_label = df_clean[label_column].values

    # TRAIN - TEST SPLIT
    X_train, X_test, y_train, y_test = train_test_split(X_text,y_label,
    test_size=parameter["split"]["test_size"],
    random_state=parameter["split"]["random_state"],
    stratify=y_label if parameter["split"]["stratify"] else None)

    # TF-IDF
    the_vectorizer = vectorizer(parameter)
    X_train_transform, X_test_transform = the_vectorize_fit(X_train, X_test, the_vectorizer)

    # SAMPLE SIZE
    sample_vector(X_train_transform, y_train, n_features=20, n_rows=5)
