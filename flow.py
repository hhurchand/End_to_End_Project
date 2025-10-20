import mlflow
import mlflow


mlflow.set_experiment("SPAM_VS_HAM")

# MLFLOW LOGGER
def the_mlflow(run_name, hyperparameters, metrics, outputs_dir):
    """
    START A RUN | LOG PARAMS | LOG METRICS | LOG ARTIFACTS
    """
    with mlflow.start_run(run_name=run_name):
        if hyperparameters:
            mlflow.log_params(hyperparameters)
        if metrics:
            mlflow.log_metrics(metrics)
        mlflow.log_artifacts(outputs_dir)

def run(name, parameter, metrics_df, outputs_dir):
    """
    LOG ONE MODEL RUN IN MLFLOW
    """
    # METRICS TO DICTIONARY
    metrics = metrics_df.set_index("Metric")["Value"].astype(float).to_dict()

    # MODEL HYPERPARAMETERS FROM YAML
    hyper = parameter["models"].get(name, {})

    # LOG
    the_mlflow(run_name=name, hyperparameters=hyper, metrics=metrics, outputs_dir=outputs_dir)
