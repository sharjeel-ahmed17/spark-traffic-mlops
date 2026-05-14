from server.src.logger import logging
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_timestamp, hour, dayofweek, month, year
from pyspark.ml.feature import VectorAssembler, StandardScaler, OneHotEncoder
from pyspark.ml import Pipeline
import os
import shutil
from server.src.config_utils import load_config

def transform_data():
    try:
        logging.info("Starting data transformation process...")
        config, params = load_config()

        spark = SparkSession.builder.appName("TrafficTransformation").getOrCreate()

        cleaned_data_path = config["data"]["processed_dir"]
        df = spark.read.csv(
            cleaned_data_path,
            header=True,
            inferSchema=True,
        )
        logging.info("===================================")
        logging.info("Loaded Cleaned Dataset")
        logging.info("  Rows    : %d", df.count())
        logging.info("  Columns : %s", df.columns)
        logging.info("===================================")

        logging.info("Parsing DateTime and extracting time features...")
        df = df.withColumn("DateTime", to_timestamp(col("DateTime"), "yyyy-MM-dd HH:mm:ss"))

        existing = df.columns
        if "Year" not in existing:
            df = df.withColumn("Year", year(col("DateTime")))
        if "Month" not in existing:
            df = df.withColumn("Month", month(col("DateTime")))
        if "Hour" not in existing:
            df = df.withColumn("Hour", hour(col("DateTime")))
        if "DayOfWeek" not in existing:
            df = df.withColumn("DayOfWeek", dayofweek(col("DateTime")))   # 1=Sun … 7=Sat

        logging.info("Time features confirmed: Year, Month, Hour, DayOfWeek")

        logging.info("Applying OneHotEncoder to Junction, DayOfWeek, Hour...")

        df = (
            df.withColumn("Junction_idx",   col("Junction").cast("double"))
              .withColumn("DayOfWeek_idx",  col("DayOfWeek").cast("double"))
              .withColumn("Hour_idx",       col("Hour").cast("double"))
        )

        ohe = OneHotEncoder(
            inputCols=["Junction_idx", "DayOfWeek_idx", "Hour_idx"],
            outputCols=["Junction_ohe", "DayOfWeek_ohe", "Hour_ohe"],
            dropLast=True,          # avoids dummy-variable trap
            handleInvalid="keep",
        )

        logging.info("Assembling feature vector...")

        categorical_features = [f"{col_name}_ohe" for col_name in params["transformation"]["categorical_cols"]]
        numeric_features = params["transformation"]["numeric_cols"]
        
        assembler_input_cols = numeric_features + categorical_features

        assembler = VectorAssembler(
            inputCols=assembler_input_cols,
            outputCol="features_raw",
            handleInvalid="skip",
        )

        logging.info("Adding StandardScaler...")

        scaler = StandardScaler(
            inputCol="features_raw",
            outputCol="features",
            withMean=False,   # False is safer for sparse OHE vectors
            withStd=True,
        )

        logging.info("Building and fitting Pipeline...")

        pipeline = Pipeline(stages=[ohe, assembler, scaler])
        pipeline_model = pipeline.fit(df)
        df_final = pipeline_model.transform(df)

        logging.info("Pipeline fitted successfully.")
        logging.info("===================================")
        logging.info("Transformed Dataset Schema")
        logging.info("===================================")
        df_final.printSchema()

        logging.info("===================================")
        logging.info("Transformed Dataset Preview (top 5 rows)")
        logging.info("===================================")
        df_final.select(
            "DateTime", "Junction", "Hour", "DayOfWeek",
            "Month", "Year", "Vehicles", "features"
        ).show(5, truncate=False)

        test_size = params["transformation"]["test_size"]
        random_split_seed = params["transformation"]["random_split_seed"]
        train_df, test_df = df_final.randomSplit([1.0 - test_size, test_size], seed=random_split_seed)
        logging.info("===================================")
        logging.info("Train/Test Split")
        logging.info("  Train rows : %d", train_df.count())
        logging.info("  Test  rows : %d", test_df.count())
        logging.info("===================================")

        output_dir = config["data"]["transformed_dir"]
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        train_path = config["data"]["train_data"]
        test_path  = config["data"]["test_data"]

        train_df.write.mode("overwrite").parquet(train_path)
        test_df.write.mode("overwrite").parquet(test_path)
        logging.info("Train data saved : %s", train_path)
        logging.info("Test  data saved : %s", test_path)

        models_dir = config["artifacts"]["pipeline_model"]
        if os.path.exists(models_dir):
            shutil.rmtree(models_dir)

        pipeline_model.save(models_dir)
        logging.info("Pipeline model saved : %s", models_dir)

        logging.info("===================================")
        logging.info("Transformation Complete")
        logging.info("  %s/", config["data"]["train_data"])
        logging.info("  %s/", config["data"]["test_data"])
        logging.info("  %s/", config["artifacts"]["pipeline_model"])
        logging.info("===================================")
        logging.info("Data transformation process completed successfully!")

    except Exception as e:
        logging.error("Error during data transformation: %s", str(e))
        raise

if __name__ == "__main__":
    try:
        transform_data()
    except Exception as e:
        logging.error("Error during data transformation: %s", str(e))