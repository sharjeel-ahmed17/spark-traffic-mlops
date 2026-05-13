from fastapi import FastAPI
from dotenv import load_dotenv

import os
import sys
import mlflow
import mlflow.spark

from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors
from pyspark.ml import PipelineModel

# ─────────────────────────────────────────────────────────────
# FIX PYSPARK WINDOWS ISSUE
# ─────────────────────────────────────────────────────────────
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

# ─────────────────────────────────────────────────────────────
# LOAD ENV VARIABLES
# ─────────────────────────────────────────────────────────────
load_dotenv()

# ─────────────────────────────────────────────────────────────
# SET MLFLOW TRACKING URI
# ─────────────────────────────────────────────────────────────
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

# ─────────────────────────────────────────────────────────────
# FASTAPI APP
# ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="Traffic Vehicle Prediction API",
    version="1.0.0",
    description="Traffic Vehicle Prediction using Spark MLlib + MLflow"
)

# ─────────────────────────────────────────────────────────────
# CREATE SPARK SESSION
# ─────────────────────────────────────────────────────────────
spark = SparkSession.builder \
    .appName("TrafficPredictionAPI") \
    .getOrCreate()

# ─────────────────────────────────────────────────────────────
# LOAD MODEL AND PIPELINE
# ─────────────────────────────────────────────────────────────
MODEL_NAME = "Traffic_Vehicle_Prediction"

# Load the transformation pipeline
pipeline_model = PipelineModel.load("models/pipeline")

# Load the trained model from MLflow
model = mlflow.spark.load_model(
    model_uri=f"models:/{MODEL_NAME}/Production"
)

# ─────────────────────────────────────────────────────────────
# IMPORTANT:
# YOUR TRAINED MODEL EXPECTS MORE THAN 5 FEATURES
#
# UPDATE THE FEATURE COUNT BELOW ACCORDING TO:
#
# train_df.select("features").first()
#
# Example:
# DenseVector([1,2,3,4,5,6,7,8,9])
#
# Then create same number of inputs here.
# ─────────────────────────────────────────────────────────────

@app.get("/")
def home():
    return {
        "message": "Traffic Vehicle Prediction API Running"
    }

# ─────────────────────────────────────────────────────────────
# PREDICTION ROUTE
# ─────────────────────────────────────────────────────────────
@app.post("/predict")
def predict(
    junction: float,
    day_of_week: float,
    hour: float,
    month: float,
    year: float
):
    try:
        # CREATE RAW INPUT DATAFRAME
        # The pipeline expects columns exactly as they were during training
        input_data = [(junction, day_of_week, hour, month, year)]
        columns = ["Junction", "DayOfWeek", "Hour", "Month", "Year"]

        input_df = spark.createDataFrame(input_data, columns)

        # PRE-PROCESSING: Cast categorical columns to double for OHE (as done in transformation.py)
        from pyspark.sql.functions import col
        input_df = (
            input_df.withColumn("Junction_idx",   col("Junction").cast("double"))
                   .withColumn("DayOfWeek_idx",  col("DayOfWeek").cast("double"))
                   .withColumn("Hour_idx",       col("Hour").cast("double"))
        )

        # 1. Transform raw features using the saved pipeline
        transformed_df = pipeline_model.transform(input_df)

        # 2. Use the trained model to predict based on the 'features' column
        prediction_df = model.transform(transformed_df)

        prediction = prediction_df.select("prediction").collect()[0][0]

        return {
            "predicted_vehicles": round(float(prediction), 2),
            "model_name": MODEL_NAME,
            "status": "success"
        }

    except Exception as e:
        return {
            "status": "failed",
            "error": str(e)
        }

# ─────────────────────────────────────────────────────────────
# RUN COMMAND
#
# uvicorn app:app --reload
#
# SWAGGER DOCS
#
# http://127.0.0.1:8000/docs
# ─────────────────────────────────────────────────────────────