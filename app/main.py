from fastapi import FastAPI
from dotenv import load_dotenv
import os
import sys
import mlflow
import mlflow.spark
from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors
from pyspark.ml import PipelineModel

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

load_dotenv()

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

app = FastAPI(
    title="Traffic Vehicle Prediction API",
    version="1.0.0",
    description="Traffic Vehicle Prediction using Spark MLlib + MLflow"
)

spark = SparkSession.builder \
    .appName("TrafficPredictionAPI") \
    .getOrCreate()

MODEL_NAME = "Traffic_Vehicle_Prediction"

pipeline_model = PipelineModel.load("models/pipeline")

model = mlflow.spark.load_model(
    model_uri=f"models:/{MODEL_NAME}/Production"
)

@app.get("/")
def home():
    return {
        "message": "Traffic Vehicle Prediction API Running"
    }

@app.post("/predict")
def predict(
    junction: float,
    day_of_week: float,
    hour: float,
    month: float,
    year: float
):
    try:
        input_data = [(junction, day_of_week, hour, month, year)]
        columns = ["Junction", "DayOfWeek", "Hour", "Month", "Year"]
        input_df = spark.createDataFrame(input_data, columns)
        from pyspark.sql.functions import col
        input_df = (
            input_df.withColumn("Junction_idx",   col("Junction").cast("double"))
                   .withColumn("DayOfWeek_idx",  col("DayOfWeek").cast("double"))
                   .withColumn("Hour_idx",       col("Hour").cast("double"))
        )

        transformed_df = pipeline_model.transform(input_df)
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