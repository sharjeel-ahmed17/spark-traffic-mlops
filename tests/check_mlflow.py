import mlflow
from dotenv import load_dotenv
import os

load_dotenv()
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

client = mlflow.tracking.MlflowClient()

# Set the alias
client.set_registered_model_alias("Traffic_Vehicle_Prediction", "champion", "1")
print("Alias set!")

# Verify
versions = client.search_model_versions("name='Traffic_Vehicle_Prediction'")
for v in versions:
    print(f"Version: {v.version}, Aliases: {v.aliases}, Status: {v.status}")