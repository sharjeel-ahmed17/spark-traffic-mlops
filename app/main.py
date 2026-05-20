from fastapi import FastAPI
from dotenv import load_dotenv
import os
import sys
import mlflow
import mlflow.spark
from pyspark.sql import SparkSession
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

MODEL_NAME     = "Traffic_Vehicle_Prediction"
PIPELINE_NAME  = "Traffic_Transformation_Pipeline"

# Prediction model — MLflow se
model = mlflow.spark.load_model(
    model_uri=f"models:/{MODEL_NAME}/4"
)

# Transformation pipeline — MLflow se (local path nahi)
pipeline_model = mlflow.spark.load_model(
    model_uri=f"models:/{PIPELINE_NAME}/1"
)

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

        # idx columns banao — pipeline ke liye
        input_df = (
            input_df
            .withColumn("Junction_idx",  input_df["Junction"].cast("double"))
            .withColumn("DayOfWeek_idx", input_df["DayOfWeek"].cast("double"))
            .withColumn("Hour_idx",      input_df["Hour"].cast("double"))
        )

        # Transformation pipeline apply karo
        transformed_df = pipeline_model.transform(input_df)

        # Prediction
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