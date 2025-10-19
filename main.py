# main.py — COMPLETE PIPELINE (CLEAN → TRAIN → TUNE → SAVE)

import pickle
from sklearn.metrics import classification_report, confusion_matrix

from src.data.make_dataset import clean_and_save
from src.features.build_features import load_split_vectorize
from src.models.train_model import def_lin, def_nb, def_log, tune_models
from src.visualization.visualization import plot_ham_spam, plot_cm


def main():
    # STEP 1 — CLEAN DATA
    print("STEP 1: Cleaning raw data...")
    clean_and_save()

    # STEP 2 — SPLIT + VECTORIZE
    print("\nSTEP 2: Splitting + Vectorizing...")
    X_train, X_test, y_train, y_test, Xtr, Xte, vectorizer = load_split_vectorize()
    plot_ham_spam(y_train)

    # STEP 3 — MLflow HYPERPARAMETER TUNING
    print("\nSTEP 3: Running MLflow tuning...")
    tuned = tune_models(Xtr, Xte, y_train, y_test)

    # STEP 4 — TRAIN MODELS (using defaults)
    print("\nSTEP 4: Training models...")
    r1 = def_lin(Xtr, Xte, y_train, y_test)
    r2 = def_nb(Xtr, Xte, y_train, y_test)
    r3 = def_log(Xtr, Xte, y_train, y_test)

    print("\nRESULTS (accuracy)")
    for name, acc, _, _ in [r1, r2, r3]:
        print(f"{name:<12} {acc:.4f}")

    # STEP 5 — PICK BEST MODEL
    best = r1
    for r in [r2, r3]:
        if r[1] > best[1] or (r[1] == best[1] and r[0] == "LinearSVC"):
            best = r
    best_name, best_acc, best_preds, best_clf = best
    print(f"\nBEST MODEL: {best_name} (acc={best_acc:.4f})")

    # STEP 6 — REPORT + CONFUSION MATRIX
    print("\nCLASSIFICATION REPORT (Ham=0, Spam=1):")
    print(classification_report(y_test, best_preds, target_names=["Ham", "Spam"], digits=4))
    cm = confusion_matrix(y_test, best_preds)
    plot_cm(cm, ["Ham", "Spam"])

    # STEP 7 — SAVE BEST MODEL + VECTORIZER
    print("\nSTEP 7: Saving best model and vectorizer (pickle)...")
    with open("final_model.pkl", "wb") as f:
        pickle.dump(best_clf, f)
    with open("vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    print("✅ Saved: final_model.pkl , vectorizer.pkl")

    print("\nPIPELINE COMPLETE ✅")


if __name__ == "__main__":
    main()
