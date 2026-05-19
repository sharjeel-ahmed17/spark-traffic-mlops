from fastapi import FastAPI
from dotenv import load_dotenv
import os
import sys
import mlflow
import mlflow.spark
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from fastapi.middleware.cors import CORSMiddleware
import sentry_sdk

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

load_dotenv()

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
SENTRY_DSN = os.getenv("SENTRY_DSN")

sentry_sdk.init(
    dsn=SENTRY_DSN,
    send_default_pii=True,
)

app = FastAPI(
    title="Traffic Vehicle Prediction API",
    version="1.0.0",
    description="Traffic Vehicle Prediction using Spark MLlib + MLflow"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins="*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

spark = SparkSession.builder \
    .appName("TrafficPredictionAPI") \
    .getOrCreate()

MODEL_NAME = "Traffic_Vehicle_Prediction"

# Prediction model
model = mlflow.spark.load_model(
    model_uri=f"models:/{MODEL_NAME}/1"
)

# Transformation pipeline — same jo training mein use hui thi
PIPELINE_PATH = os.getenv("PIPELINE_PATH", "models/pipeline")
pipeline_model = PipelineModel.load(PIPELINE_PATH)

@app.get("/")
def home():
    return {"message": "Traffic Vehicle Prediction API Running"}

@app.get("/tester")
def tester():
    return {"message": "tester endpoint is working fine"}

@app.get("/sentry-debug")
async def trigger_error():
    division_by_zero = 1 / 0

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

        # Pehle same transformation pipeline apply karo
        input_df = input_df \
            .withColumn("Junction_idx", input_df["Junction"].cast("double")) \
            .withColumn("DayOfWeek_idx", input_df["DayOfWeek"].cast("double")) \
            .withColumn("Hour_idx", input_df["Hour"].cast("double"))

        transformed_df = pipeline_model.transform(input_df)

        # Phir prediction
        prediction_df = model.transform(transformed_df)
        prediction = prediction_df.select("prediction").collect()[0][0]

        return {
            "predicted_vehicles": round(float(prediction), 2),
            "model_name": MODEL_NAME,
            "status": "success"
        }

    except Exception as e:
        sentry_sdk.capture_exception(e)
        return {
            "status": "failed",
            "error": str(e)
        }