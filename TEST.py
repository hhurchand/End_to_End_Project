"""
ANALYZE AND TEST THE CONTENTS OF THE CSV FILE AFTER
TRANSFORMATION / CLEANING IS COMPLETED.
"""

from src.utils.input import CSV, YAML

def run_test():
    # LOAD PARAMS
    params = YAML().load("params.yaml")
    raw_path = params["data"]["clean"]

    # LOAD CSV
    df = CSV().load(raw_path)

    # VERIFICATION
    print("\nSHAPE")
    print(df.shape)

    print("\nCOLUMNS")
    print(list(df.columns))

    # LABEL COUNTS
    print("\nLABEL COUNTS (EXPECT 0=HAM, 1=SPAM)")
    print(df["label"].value_counts())

    # EXAMPLES OF EACH CLASS
    print("\nEXAMPLES: SPAM (label==1)")
    print(df[df["label"] == 1]["email"].head(5))

    print("\nEXAMPLES: HAM (label==0)")
    print(df[df["label"] == 0]["email"].head(5))

if __name__ == "__main__":
    run_test()
