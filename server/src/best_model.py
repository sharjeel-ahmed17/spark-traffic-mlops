from dotenv import load_dotenv
load_dotenv()
from server.src.logger import logging
from mlflow.tracking import MlflowClient
import mlflow
import os

def best_model():
    try:
        logging.info("Starting best model selection process...")

        tracking_uri    = os.getenv("MLFLOW_TRACKING_URI")
        experiment_name = "Traffic_Vehicles_Regression_Experiment"
        model_name      = "Traffic_Vehicle_Prediction"

        mlflow.set_tracking_uri(tracking_uri)
        client = MlflowClient()
        logging.info("MLflow tracking URI : %s", tracking_uri)
        logging.info("Experiment          : %s", experiment_name)

        experiment = client.get_experiment_by_name(experiment_name)

        if experiment is None:
            raise Exception(
                f"Experiment '{experiment_name}' not found. "
                "Run training.py first to create it."
            )

        experiment_id = experiment.experiment_id
        logging.info("Experiment ID : %s", experiment_id)

        runs = client.search_runs(
            experiment_ids=[experiment_id],
            filter_string="metrics.RMSE > 0",   # exclude incomplete/failed runs
            order_by=["metrics.RMSE ASC"],       # best = lowest RMSE
        )

        if not runs:
            raise Exception(
                "No completed runs found in the experiment. "
                "Run training.py first."
            )

        best_run = runs[0]

        logging.info("===================================")
        logging.info("BEST RUN FOUND")
        logging.info("  Run ID     : %s", best_run.info.run_id)
        logging.info("  Model name : %s", best_run.data.params.get("model", "N/A"))
        logging.info("  RMSE       : %.4f", best_run.data.metrics["RMSE"])
        logging.info("  R2         : %.4f", best_run.data.metrics.get("R2",  0.0))
        logging.info("  MAE        : %.4f", best_run.data.metrics.get("MAE", 0.0))
        logging.info("===================================")

        model_uri = f"runs:/{best_run.info.run_id}/model"

        logging.info("Registering model from URI : %s", model_uri)

        model_version = mlflow.register_model(
            model_uri=model_uri,
            name=model_name,
        )

        logging.info("===================================")
        logging.info("MODEL REGISTERED")
        logging.info("  Name    : %s", model_name)
        logging.info("  Version : %s", model_version.version)
        logging.info("===================================")

        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Production",
            archive_existing_versions=True,  
        )

        logging.info("Model v%s transitioned to stage: Production", model_version.version)
        logging.info("===================================")
        logging.info("Best model selection process completed successfully!")
        logging.info("===================================")

    except Exception as e:
        logging.error("Error during best model selection: %s", str(e))
        raise

if __name__ == "__main__":
    try:
        best_model()
    except Exception as e:
        logging.error("Error during best model selection: %s", str(e))