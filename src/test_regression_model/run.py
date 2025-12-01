import argparse
import logging
import wandb
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.metrics import mean_absolute_error

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

def go(args):
    run = wandb.init(job_type="test_regression_model")

    logger.info("Downloading artifacts")
    # Download the model
    model_local_path = run.use_artifact(args.mlflow_model).download()
    
    # Download the test dataset
    test_dataset_path = run.use_artifact(args.test_dataset).file()
    df = pd.read_csv(test_dataset_path)

    # Separate features and target
    y_test = df['price']
    X_test = df.drop(['price'], axis=1)

    logger.info("Loading model and performing inference on test set")
    sk_pipe = mlflow.sklearn.load_model(model_local_path)
    y_pred = sk_pipe.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    logger.info(f"Test MAE: {mae}")

    run.summary["test_mae"] = mae
    run.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the trained model against the test dataset")

    parser.add_argument(
        "--mlflow_model", 
        type=str, 
        help="Input MLflow model",
        required=True
    )
    
    parser.add_argument(
        "--test_dataset", 
        type=str, 
        help="Input test dataset",
        required=True
    )

    args = parser.parse_args()
    go(args)