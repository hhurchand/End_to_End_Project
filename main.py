import os
import pickle
from collections import Counter
from sklearn.model_selection import train_test_split

from src.utils.input import CSV, YAML
from src.transform import the_clean
from src.vectorize import vectorizer, the_vectorize_fit
from src.report import the_report
from flow import run

# MODELS
from src.model_bayes import the_model as build_bayes, fit_and_predict as run_bayes
from src.model_logistic import the_model as build_logistic, fit_and_predict as run_logistic
from src.model_linear import the_model as build_linear, fit_and_predict as run_linear



# MAIN PIPELINE : CLEAN | SPLIT | TF-IDF | TRAIN | REPORT

def save_the_model(the_model, model_name, outputs_dir):
    """
    SAVE TRAINED MODEL AS PICKLE FILE
    """
    path = os.path.join(outputs_dir, f"{model_name}_model.pkl")
    with open(path, "wb") as f:
        pickle.dump(the_model, f)
    print(f"MODEL SAVED: {path}")


def the_counts(arr):
    """
    FOR CLEANER READABLE COUNTS
    """
    c = Counter(arr.tolist())
    return {int(k): int(v) for k, v in sorted(c.items())}


def the_main():
    """
    FULL PIPELINE TO TRAIN AND SAVE ALL MODELS
    """
    # LOAD PARAMS
    parameter = YAML().load("params.yaml")

    # LOAD THE CLEAN DATA
    clean_path = parameter["data"]["clean"]
    df = CSV().load(clean_path)

    print(f"\nDATA LOADED: {clean_path}")
    print("SHAPE       :", df.shape)

    # CLEAN DATA | UNCOMMENT IF TRANSFORMATION NEEDED
    # df = the_clean(df, parameter)

    # THE COLUMNS
    text_column = parameter["text"]
    label_column = parameter["label"]

    X_text = df[text_column].values
    y_label = df[label_column].values

    # TRAIN - TEST SPLIT
    stratify_labels = y_label if parameter["split"]["stratify"] else None
    X_train, X_test, y_train, y_test = train_test_split(
        X_text,
        y_label,
        test_size=parameter["split"]["test_size"],
        random_state=parameter["split"]["random_state"],
        stratify=stratify_labels,)

    # CLEAN COUNTS PRINT
    train_counts = the_counts(y_train)
    test_counts = the_counts(y_test)

    print("\nSPLIT SUMMARY")
    print("TRAIN LABELS :", train_counts)
    print("TEST  LABELS :", test_counts)

    # TF-IDF
    the_vector = vectorizer(parameter)
    X_train_tf, X_test_tf = the_vectorize_fit(X_train, X_test, the_vector)

    # SAVE THE FITTED VECTORIZER
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/vectorizer.pkl", "wb") as f:
        pickle.dump(the_vector, f)
    print("\n")
    print("VECTORIZER SAVED: outputs/vectorizer.pkl")



    print("\nTF-IDF COMPLETE")
    print("TRAIN MATRIX :", X_train_tf.shape)
    print("TEST  MATRIX :", X_test_tf.shape)

    # MODEL MAP
    MODELS = {
        "bayes": (build_bayes, run_bayes),
        "logistic": (build_logistic, run_logistic),
        "linear": (build_linear, run_linear),}

    # TRAIN EACH MODEL
    for name, (build_fn, run_fn) in MODELS.items():
        print(f"\n TRAINING MODEL: {name.upper()}")

        outputs_dir = f"outputs/{name}"
        parameter["data"]["outputs"] = outputs_dir
        os.makedirs(outputs_dir, exist_ok=True)

        model = build_fn(parameter)
        model, y_predict, y_probability = run_fn(model, X_train_tf, y_train, X_test_tf)

        save_the_model(model, name, outputs_dir)

        # ONE REPORT CALL RETURNS METRICS_DF
        metrics_df = the_report(y_test, y_predict, y_probability, parameter)

        # MLFLOW LOG
        run(name, parameter, metrics_df, outputs_dir)

    print("\nPIPELINE COMPLETE\n")


if __name__ == "__main__":
    the_main()
