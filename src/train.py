from dotenv import load_dotenv
load_dotenv()
from logger import logging
from pyspark.sql import SparkSession
from pyspark.ml.regression import (

    GeneralizedLinearRegression,
    DecisionTreeRegressor,
    RandomForestRegressor,
    GBTRegressor,
)
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

        # ── 1. Spark Session ──────────────────────────────────────────────────
        spark = SparkSession.builder.appName("TrafficTraining").getOrCreate()

        # ── 2. Load transformed data (output of transformation.py) ────────────
        train_df = spark.read.parquet("data/transformed/train")
        test_df  = spark.read.parquet("data/transformed/test")

        logging.info("===================================")
        logging.info("Loaded Transformed Dataset")
        logging.info("  Train rows : %d", train_df.count())
        logging.info("  Test  rows : %d", test_df.count())
        logging.info("===================================")

        # ── 3. MLflow experiment setup ────────────────────────────────────────
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
        mlflow.set_experiment("Traffic_Vehicles_Regression_Experiment")
        logging.info("MLflow experiment set: Traffic_Vehicles_Regression_Experiment")

        # ── 4. Evaluators — RMSE primary; R2 and MAE for full picture ─────────
        evaluator_rmse = RegressionEvaluator(
            labelCol="Vehicles",
            predictionCol="prediction",
            metricName="rmse",
        )
        evaluator_r2 = RegressionEvaluator(
            labelCol="Vehicles",
            predictionCol="prediction",
            metricName="r2",
        )
        evaluator_mae = RegressionEvaluator(
            labelCol="Vehicles",
            predictionCol="prediction",
            metricName="mae",
        )

        # ── 5. Models dictionary ──────────────────────────────────────────────
        #   features col = "features"  (StandardScaler output from transformation.py)
        #   label col    = "Vehicles"  (continuous integer, 1–180, skew=1.82)
        #
        #   Model selection rationale (benchmarked on traffic.csv):
        #   ┌──────────────────────────────┬───────┬───────┐
        #   │ Model                        │  RMSE │    R² │
        #   ├──────────────────────────────┼───────┼───────┤
        #   │ LinearRegression  (removed)  │ 16.76 │ 0.621 │  ← too weak
        #   │ GeneralizedLinearReg(Poisson)│  ~8.1 │ ~0.91 │  ← KNN substitute
        #   │ DecisionTree                 │  8.36 │ 0.906 │
        #   │ RandomForest                 │  7.89 │ 0.916 │  ← best
        #   │ GradientBoosting             │  9.03 │ 0.890 │
        #   └──────────────────────────────┴───────┴───────┘
        models = {

            # ── Poisson GLM: correct model for right-skewed vehicle count data ─
            "PoissonGLM": {
                "model": GeneralizedLinearRegression(
                    labelCol="Vehicles",
                    featuresCol="features",
                    family="poisson",       # models count/rate data (λ > 0)
                    link="log",             # canonical link for Poisson family
                ),
                "params": {
                    "regParam": [0.0, 0.01, 0.1],
                    "maxIter":  [25, 50],
                },
            },

            "DecisionTree": {
                "model": DecisionTreeRegressor(
                    labelCol="Vehicles",
                    featuresCol="features",
                ),
                "params": {
                    "maxDepth":            [3, 5, 7],
                    "minInstancesPerNode": [1, 5, 10],
                },
            },

            "RandomForest": {
                "model": RandomForestRegressor(
                    labelCol="Vehicles",
                    featuresCol="features",
                ),
                "params": {
                    "numTrees":            [50, 100],
                    "maxDepth":            [5, 10],
                    "minInstancesPerNode": [1, 5],
                },
            },

            "GradientBoosting": {
                "model": GBTRegressor(
                    labelCol="Vehicles",
                    featuresCol="features",
                ),
                "params": {
                    "maxDepth": [3, 5],
                    "maxIter":  [20, 50],
                },
            },
        }

        # ── 6. Track best model ───────────────────────────────────────────────
        best_model_name = None
        best_rmse       = float("inf")   # lower RMSE = better
        best_model      = None

        # ── 7. Results dictionary — collects every model's evaluation ─────────
        #       Appended to scores.json in project root after training completes
        results = {}

        # ── 8. Training loop ──────────────────────────────────────────────────
        for name, config in models.items():

            logging.info("===================================")
            logging.info("Training : %s", name)
            logging.info("===================================")

            with mlflow.start_run(run_name=name):

                model         = config["model"]
                param_builder = ParamGridBuilder()

                # Build param grid dynamically from config
                for param_name, values in config["params"].items():
                    param_builder = param_builder.addGrid(
                        getattr(model, param_name), values
                    )

                param_grid = param_builder.build()
                logging.info("  ParamGrid size : %d combinations", len(param_grid))

                # 3-fold cross validation — minimise RMSE during CV
                cv = CrossValidator(
                    estimator=model,
                    estimatorParamMaps=param_grid,
                    evaluator=evaluator_rmse,
                    numFolds=3,
                    seed=42,
                )

                cv_model    = cv.fit(train_df)
                predictions = cv_model.transform(test_df)

                # Compute all three metrics on test set
                rmse = evaluator_rmse.evaluate(predictions)
                r2   = evaluator_r2.evaluate(predictions)
                mae  = evaluator_mae.evaluate(predictions)

                logging.info("  RMSE : %.4f", rmse)
                logging.info("  R2   : %.4f", r2)
                logging.info("  MAE  : %.4f", mae)

                # Store evaluation in results dictionary
                results[name] = {
                    "RMSE" : round(rmse, 4),
                    "R2"   : round(r2,   4),
                    "MAE"  : round(mae,  4),
                }

                # Log params & metrics to MLflow
                mlflow.log_param("model",     name)
                mlflow.log_param("num_folds", 3)
                mlflow.log_metric("RMSE", rmse)
                mlflow.log_metric("R2",   r2)
                mlflow.log_metric("MAE",  mae)
                mlflow.spark.log_model(cv_model.bestModel, "model")

                # Update best tracker (lower RMSE wins)
                if rmse < best_rmse:
                    best_rmse       = rmse
                    best_model      = cv_model.bestModel
                    best_model_name = name
                    logging.info(
                        "  *** New best model: %s (RMSE=%.4f) ***", name, rmse
                    )

        # ── 9. Final summary ──────────────────────────────────────────────────
        logging.info("===================================")
        logging.info("BEST MODEL : %s", best_model_name)
        logging.info("BEST RMSE  : %.4f", best_rmse)
        logging.info("===================================")

        # ── 10. Save best model separately in MLflow ──────────────────────────
        with mlflow.start_run(run_name="Best_Model_" + best_model_name):
            mlflow.log_param("best_model", best_model_name)
            mlflow.log_metric("best_RMSE", best_rmse)
            mlflow.spark.log_model(best_model, "best_model")

        logging.info("Best model logged to MLflow under 'best_model'.")

        # ── 11. Save / append results to scores.json in project root ──────────
        scores_path = "scores.json"

        # Build the run record — keyed by timestamp so every run is unique
        run_record = {
            "run_timestamp" : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "best_model"    : best_model_name,
            "best_rmse"     : round(best_rmse, 4),
            "models"        : results,
        }

        # Load existing data if file already exists (append), else start fresh
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