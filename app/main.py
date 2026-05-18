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
    # Add data like request headers and IP for users,
    # see https://docs.sentry.io/platforms/python/data-management/data-collected/ for more info
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

model = mlflow.spark.load_model(
    model_uri=f"models:/{MODEL_NAME}/Production"
)

@app.get("/")
def home():
    return {
        "message": "Traffic Vehicle Prediction API Running"
    }

# for sentry testing, you can call this endpoint to trigger an error and see it in Sentry dashboard
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

        prediction_df = model.transform(input_df)
        prediction = prediction_df.select("prediction").collect()[0][0]

        return {
            "predicted_vehicles": round(float(prediction), 2),
            "model_name": MODEL_NAME,
            "status": "success"
        }

    except Exception as e:
        sentry_sdk.capture_exception(e) # Capture the exception in Sentry
        return {
            "status": "failed",
            "error": str(e)
        }