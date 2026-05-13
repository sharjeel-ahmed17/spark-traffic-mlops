from fastapi import FastAPI
from dotenv import load_dotenv

import os
import sys
import mlflow
import mlflow.spark

from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors

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
# LOAD MODEL FROM MLFLOW MODEL REGISTRY
# ─────────────────────────────────────────────────────────────
MODEL_NAME = "Traffic_Vehicle_Prediction"

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
    feature_1: float,
    feature_2: float,
    feature_3: float,
    feature_4: float,
    feature_5: float,
    feature_6: float,
    feature_7: float,
    feature_8: float,
    feature_9: float
):
    try:

        # CREATE FEATURE VECTOR
        features = Vectors.dense([
            feature_1,
            feature_2,
            feature_3,
            feature_4,
            feature_5,
            feature_6,
            feature_7,
            feature_8,
            feature_9
        ])

        # CREATE SPARK DATAFRAME
        input_df = spark.createDataFrame(
            [(features,)],
            ["features"]
        )

        # PREDICT
        prediction_df = model.transform(input_df)

        prediction = prediction_df.select(
            "prediction"
        ).collect()[0][0]

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