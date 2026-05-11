from logger import logging
from pyspark.sql import SparkSession
import os
def load_data():
    try:
        logging.info("Starting data loading process...")
        spark = SparkSession.builder \
            .appName("Traffic Vehicle Prediction") \
            .getOrCreate()
        print("Spark Session Created")
        file_path = os.path.join("data/raw", "traffic.csv")
        print("Reading file from:", file_path)
        df = spark.read.csv(file_path, header=True, inferSchema=True)
        df.show(5)
        df.printSchema()
        logging.info("Data loading  process completed successfully!")
    except Exception as e:
        logging.error("Error during data loading: %s", str(e))
if __name__ == "__main__":
    try:
        load_data()
    except Exception as e:
        logging.error("Error during data loading: %s", str(e))