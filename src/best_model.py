from dotenv import load_dotenv
load_dotenv()
from logger import logging
from mlflow.tracking import MlflowClient
import mlflow
import os

def best_model():
    try:
        logging.info("Starting best model selection process...")

        tracking_uri    = os.getenv("MLFLOW_TRACKING_URI")
        experiment_name = "Traffic_Vehicles_Regression_Experiment"
        model_name      = "Traffic_Vehicle_Prediction"
        pipeline_name   = "Traffic_Transformation_Pipeline"

        mlflow.set_tracking_uri(tracking_uri)
        client = MlflowClient()

        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            raise Exception(f"Experiment '{experiment_name}' not found.")

        experiment_id = experiment.experiment_id

        runs = client.search_runs(
            experiment_ids=[experiment_id],
            filter_string="metrics.RMSE > 0",
            order_by=["metrics.RMSE ASC"],
        )

        if not runs:
            raise Exception("No completed runs found.")

        best_run = runs[0]

        logging.info("===================================")
        logging.info("BEST RUN FOUND")
        logging.info("  Run ID : %s", best_run.info.run_id)
        logging.info("  RMSE   : %.4f", best_run.data.metrics["RMSE"])
        logging.info("===================================")

        # Prediction model register karo
        model_uri = f"runs:/{best_run.info.run_id}/model"
        model_version = mlflow.register_model(
            model_uri=model_uri,
            name=model_name,
        )
        logging.info("Prediction model registered — version %s", model_version.version)

        # Transformation pipeline register karo
        pipeline_uri = f"runs:/{best_run.info.run_id}/transformation_pipeline"
        pipeline_version = mlflow.register_model(
            model_uri=pipeline_uri,
            name=pipeline_name,
        )
        logging.info("Transformation pipeline registered — version %s", pipeline_version.version)

        # Dono ko Production mein transition karo
        for name, version in [
            (model_name,    model_version.version),
            (pipeline_name, pipeline_version.version),
        ]:
            client.transition_model_version_stage(
                name=name,
                version=version,
                stage="Production",
                archive_existing_versions=True,
            )
            logging.info("%s v%s → Production", name, version)

        logging.info("===================================")
        logging.info("Best model selection completed!")
        logging.info("===================================")

    except Exception as e:
        logging.error("Error: %s", str(e))
        raise

if __name__ == "__main__":
    try:
        best_model()
    except Exception as e:
        logging.error("Error: %s", str(e))