from dotenv import load_dotenv
load_dotenv()
from server.src.config_utils import load_config
from server.src.logger import logging
from pyspark.sql import SparkSession
from pyspark.ml.regression import GeneralizedLinearRegression,DecisionTreeRegressor,RandomForestRegressor,GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
import mlflow
import mlflow.spark
import json
import os
from datetime import datetime


def train_models():
    try:
        logging.info("Starting model training process...")
        config, params = load_config()

        spark = SparkSession.builder.appName("TrafficTraining").getOrCreate()

        train_df = spark.read.parquet(config["data"]["train_data"])
        test_df  = spark.read.parquet(config["data"]["test_data"])

        logging.info("===================================")
        logging.info("Loaded Transformed Dataset")
        logging.info("  Train rows : %d", train_df.count())
        logging.info("  Test  rows : %d", test_df.count())
        logging.info("===================================")

        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
        mlflow.set_experiment(config["project"]["experiment_name"])
        logging.info("MLflow experiment set: Traffic_Vehicles_Regression_Experiment")

        target_col = params["training"]["target_col"]
        
        evaluator_rmse = RegressionEvaluator(
            labelCol=target_col,
            predictionCol="prediction",
            metricName="rmse",
        )
        evaluator_r2 = RegressionEvaluator(
            labelCol=target_col,
            predictionCol="prediction",
            metricName="r2",
        )
        evaluator_mae = RegressionEvaluator(
            labelCol=target_col,
            predictionCol="prediction",
            metricName="mae",
        )

        models_config = params["training"]["models"]
        models = {
            "PoissonGLM": {
                "model": GeneralizedLinearRegression(
                    labelCol=target_col,
                    featuresCol="features",
                    family="poisson",
                    link="log",
                ),
                "params": models_config["PoissonGLM"],
            },
            "DecisionTree": {
                "model": DecisionTreeRegressor(
                    labelCol=target_col,
                    featuresCol="features",
                ),
                "params": models_config["DecisionTree"],
            },
            "RandomForest": {
                "model": RandomForestRegressor(
                    labelCol=target_col,
                    featuresCol="features",
                ),
                "params": models_config["RandomForest"],
            },
            "GradientBoosting": {
                "model": GBTRegressor(
                    labelCol=target_col,
                    featuresCol="features",
                ),
                "params": models_config["GradientBoosting"],
            },
        }

        best_model_name = None
        best_rmse       = float("inf")
        best_model      = None

        results = {}

        for name, m_config in models.items():

            logging.info("===================================")
            logging.info("Training : %s", name)
            logging.info("===================================")

            with mlflow.start_run(run_name=name):

                model         = m_config["model"]
                param_builder = ParamGridBuilder()

                for param_name, values in m_config["params"].items():
                    param_builder = param_builder.addGrid(
                        getattr(model, param_name), values
                    )

                param_grid = param_builder.build()
                logging.info("  ParamGrid size : %d combinations", len(param_grid))

                cv = CrossValidator(
                    estimator=model,
                    estimatorParamMaps=param_grid,
                    evaluator=evaluator_rmse,
                    numFolds=params["training"]["cv_folds"],
                    seed=params["training"]["seed"],
                )

                cv_model    = cv.fit(train_df)
                predictions = cv_model.transform(test_df)

                rmse = evaluator_rmse.evaluate(predictions)
                r2   = evaluator_r2.evaluate(predictions)
                mae  = evaluator_mae.evaluate(predictions)

                logging.info("  RMSE : %.4f", rmse)
                logging.info("  R2   : %.4f", r2)
                logging.info("  MAE  : %.4f", mae)

                results[name] = {
                    "RMSE" : round(rmse, 4),
                    "R2"   : round(r2,   4),
                    "MAE"  : round(mae,  4),
                }

                mlflow.log_param("model",     name)
                mlflow.log_param("num_folds", 3)
                mlflow.log_metric("RMSE", rmse)
                mlflow.log_metric("R2",   r2)
                mlflow.log_metric("MAE",  mae)
                mlflow.spark.log_model(cv_model.bestModel, "model")

                if rmse < best_rmse:
                    best_rmse       = rmse
                    best_model      = cv_model.bestModel
                    best_model_name = name
                    logging.info(
                        "  *** New best model: %s (RMSE=%.4f) ***", name, rmse
                    )

        logging.info("===================================")
        logging.info("BEST MODEL : %s", best_model_name)
        logging.info("BEST RMSE  : %.4f", best_rmse)
        logging.info("===================================")

        with mlflow.start_run(run_name="Best_Model_" + best_model_name):
            mlflow.log_param("best_model", best_model_name)
            mlflow.log_metric("best_RMSE", best_rmse)
            mlflow.spark.log_model(best_model, "best_model")

        logging.info("Best model logged to MLflow under 'best_model'.")

        scores_path = config["artifacts"]["scores_file"]

        run_record = {
            "run_timestamp" : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "best_model"    : best_model_name,
            "best_rmse"     : round(best_rmse, 4),
            "models"        : results,
        }

        if os.path.exists(scores_path):
            with open(scores_path, "r") as f:
                all_scores = json.load(f)
            logging.info("Existing scores.json found — appending new run.")
        else:
            all_scores = []
            logging.info("No scores.json found — creating new file.")

        all_scores.append(run_record)

        with open(scores_path, "w") as f:
            json.dump(all_scores, f, indent=4)

        logging.info("===================================")
        logging.info("Scores saved to : %s", os.path.abspath(scores_path))
        logging.info("Total runs in file : %d", len(all_scores))
        logging.info("===================================")
        logging.info("Training Completed Successfully")
        logging.info("===================================")

    except Exception as e:
        logging.error("Error during model training: %s", str(e))
        raise

if __name__ == "__main__":
    try:
        train_models()
    except Exception as e:
        logging.error("Error during model training: %s", str(e))