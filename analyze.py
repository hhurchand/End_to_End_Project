from src.utils.input import CSV, YAML

"""
VIEW AND DESCRIBE THE DATASET :

THIS SCRIPT CAN LOAD ANY DATASET DEFINED IN params.yaml AND
DISPLAYS ITS STRUCTURE , SAMPLE RECORDS AND SUMMARY STATISTICS.

Usage: python analyze.py

Purpose:
    - VERIFY THAT THE DATASET PATH AND COLUMNS ARE CORRECT
    - CHECK SHAPE, MISSING VALUES AND DATA TYPES
    - QUICKLY PREVIEW SAMPLES OF THE CONTENT OF THE DATASET."""



def analyze():
    # LOAD PARAMS
    params = YAML().load("params.yaml")
    the_raw_path = params["data"]["raw"]

    # LOAD CSV
    df = CSV().load(the_raw_path)

    # SHAPE & COLUMNS
    print("\nSHAPE")
    print(df.shape)

    print("\nCOLUMNS")
    print(list(df.columns))

    # INFORMATION
    print("\nINFORMATION")
    df.info()

    # FIRST ROWS
    print("\nFIRST COUPLE OF ROWS")
    print(df.head(30))

    # SAMPLE DATA
    print("\nSAMPLE DATA")
    n = 30 if len(df) >= 30 else len(df)
    print(df.sample(n, random_state=42))

    # DESCRIBE
    print("\nDESCRIPTION")
    print(df.describe(include="all").transpose())

if __name__ == "__main__":
    analyze()
