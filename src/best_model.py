from dotenv import load_dotenv
from tomlkit import datetime
load_dotenv()
from logger import logging
from mlflow.tracking import MlflowClient
import mlflow
import os
import datetime


def best_model():
    try:
        logging.info("Starting best model selection process...")

        tracking_uri  = os.getenv("MLFLOW_TRACKING_URI")
        model_name    = "Traffic_Vehicle_Prediction"
        pipeline_name = "Traffic_Transformation_Pipeline"
        run_id_path   = "logs/best_run_id.txt"

        mlflow.set_tracking_uri(tracking_uri)
        client = MlflowClient()

        if not os.path.exists(run_id_path):
            raise Exception("logs/best_run_id.txt nahi mili — pehle train.py chalao")

        with open(run_id_path, "r") as f:
            best_run_id = f.read().strip()

        logging.info("===================================")
        logging.info("BEST RUN ID : %s", best_run_id)
        logging.info("===================================")

        # Prediction model register karo
        model_version = mlflow.register_model(
            model_uri=f"runs:/{best_run_id}/model",
            name=model_name,
        )
        logging.info("Prediction model registered — version %s", model_version.version)

        # Transformation pipeline — same run se
        pipeline_version = mlflow.register_model(
            model_uri=f"runs:/{best_run_id}/transformation_pipeline",
            name=pipeline_name,
        )
        logging.info("Transformation pipeline registered — version %s", pipeline_version.version)

        # Dono Production mein
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


        os.makedirs("logs", exist_ok=True)
        with open("logs/best_model.log", "w") as f:
            f.write(f"Best model selection completed at {datetime.datetime.now()}\n")
            f.write(f"Model: Traffic_Vehicle_Prediction\n")
            f.write(f"Pipeline: Traffic_Transformation_Pipeline\n")
            logging.info("Log file saved: logs/best_model.log")

    except Exception as e:
        logging.error("Error: %s", str(e))
        raise

if __name__ == "__main__":
    try:
        best_model()
    except Exception as e:
        logging.error("Error: %s", str(e))