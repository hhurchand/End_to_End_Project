from src.data.make_dataset import DataLoader
from src.features.transform import Transform,Tokenized_sentence,Label_column,Tfidf
from src.models.train_model import Train_Split_data, Modeling, ML_Split_data, ML_Modeling, ML_Evaluation
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from pathlib import Path
from constants import PARAMS
from yaml import safe_load
from datetime import datetime
import mlflow


with mlflow.start_run():

    # Load configs
    with open(PARAMS,encoding="UTF-8") as f:
        config = safe_load(f)
    
    # Log model and compiling hyperparameters to MLflow
    mlflow.log_params(config["model"])
    mlflow.log_params(config["compiling"]) 

    # Load data
    data = DataLoader(path=Path(config["data"]["raw_Dataset"])).load_csv()
    print("Step 1 {DataLoader} Compeleted",datetime.now().strftime("%H:%M:%S"))

    # Transform data
    transform = Transform()
    data = transform.dropnull(data)
    tokens = transform.transform_text(data)
    print("Step 2 {Transform} Compeleted",datetime.now().strftime("%H:%M:%S"))

    # Tokenize
    tokenize = Tokenized_sentence()
    data[config["transform"]["features"]] = data[config["transform"]["features"]].apply(tokenize.tokenize_text)
    print("Step 3 {Tokenize} Compeleted",datetime.now().strftime("%H:%M:%S"))

    # Label_Encode
    label = Label_column(data)
    data = label.label()
    print("Step 4 {Label_Encode} Compeleted",datetime.now().strftime("%H:%M:%S"))

    # Vectorize
    tfidf = Tfidf(data)
    X, y = tfidf.tfidvec()
    print("Step 5 {Vectorize} Compeleted",datetime.now().strftime("%H:%M:%S"))
    print(tfidf)
    print(X.shape,y.shape)

    # ------ DL-Modeling -----#
    # Split data
    splitter = Train_Split_data(
        X, 
        y, 
        test_size=0.2,
        val_size=0.1
    )
    X_train, X_val, X_test, y_train, y_val, y_test = splitter.train_split()
    print("Step 6 {Split_data} Completed", datetime.now().strftime('%H:%M:%S'))

    # Build model
    input_shape = (X_train.shape[1],)
    model = Modeling.functional_api(input_shape,
                                    dense1_no=config["model"]["dense1"],
                                    dense2_no=config["model"]["dense2"],
                                    dropout=config["model"]["dropout"],
                                    activation=config["model"]["activation"],
                                    output_activation=config["model"]["output_activation"])
    print("Step 7 {Build_Model} Completed", datetime.now().strftime('%H:%M:%S'))

    # Train model
    model, history = Modeling.compiling(
        model,
        X_train, y_train,
        X_val, y_val,
        epochs=config["compiling"]["epochs"],
        batch_size=config["compiling"]["batch_size"]
    )
    print("Step 8 {Train_Model} Completed", datetime.now().strftime('%H:%M:%S'))

    # Evaluate model
    acc, report, y_pred = Modeling.evaluate(model, X_test, y_test)
    print("Step 9 {Evaluate_Model} Completed", datetime.now().strftime('%H:%M:%S'))

    # Save DL model
    model.save("models/dl_model.keras")
    print("Step 10 {Save_DL_Model} Completed", datetime.now().strftime('%H:%M:%S'))

    # ------ ML-Modeling -----#
    # Split Data for ML
    split = ML_Split_data(X, y)
    X_train, X_test, y_train, y_test = split.xy_train()
    print("Step ML-Step 1 {Split_Data_for_ML} Completed", datetime.now().strftime('%H:%M:%S'))

    # ML modeling
    ml_model1 = ML_Modeling(X_train,X_test,y_train,y_test)
    ml_model1.model_selection(DecisionTreeClassifier())
    ml_model2 = ML_Modeling(X_train,X_test,y_train,y_test)
    ml_model2.model_selection(RandomForestClassifier())
    ml_model3 = ML_Modeling(X_train,X_test,y_train,y_test)
    ml_model3.model_selection(LogisticRegression())
    print("Step ML-Step 2 {ML_Modeling} Completed", datetime.now().strftime('%H:%M:%S'))

    # ML evaluation
    eva1 = ML_Evaluation(X_train,X_test,y_train,y_test,DecisionTreeClassifier())
    eva1.model_validation()
    eva2 = ML_Evaluation(X_train,X_test,y_train,y_test,RandomForestClassifier())
    eva2.model_validation()
    eva3 = ML_Evaluation(X_train,X_test,y_train,y_test,LogisticRegression())
    eva3.model_validation()
    print("Step ML-Step 3 {ML_evalutaion} Completed", datetime.now().strftime('%H:%M:%S'))