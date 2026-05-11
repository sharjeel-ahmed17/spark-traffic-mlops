from logger import logging
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, trim, to_timestamp, hour, dayofweek, month, year
import os


def cleaning():
    try:
        logging.info("Starting data cleaning process...")

        # ── 1. Spark Session ──────────────────────────────────────────────────
        spark = SparkSession.builder.appName("TrafficCleaning").getOrCreate()

        # ── 2. Load dataset ───────────────────────────────────────────────────
        df = spark.read.csv(
            "data/raw/traffic.csv",
            header=True,
            inferSchema=True,
        )
        logging.info("===================================")
        logging.info("Original Dataset Shape")
        logging.info("Rows    : %d", df.count())
        logging.info("Columns : %d", len(df.columns))
        logging.info("===================================")

        # ── 3. Null value audit ───────────────────────────────────────────────
        logging.info("Checking Null Values...")
        for column in df.columns:
            null_count = df.filter(col(column).isNull()).count()
            logging.info("  %-12s : %d null values", column, null_count)

        # ── 4. Remove duplicates ──────────────────────────────────────────────
        before_dup = df.count()
        df = df.dropDuplicates()
        after_dup = df.count()
        logging.info("===================================")
        logging.info("Duplicate Removal")
        logging.info("  Rows before      : %d", before_dup)
        logging.info("  Rows after       : %d", after_dup)
        logging.info("  Duplicates removed: %d", before_dup - after_dup)
        logging.info("===================================")

        # ── 5. Drop the surrogate ID column (not useful for analysis) ─────────
        logging.info("Removing unnecessary column: ID")
        df = df.drop("ID")
        logging.info("ID column removed successfully.")

        # ── 6. Cast DateTime string → proper TimestampType ───────────────────
        logging.info("Casting DateTime column to TimestampType...")
        df = df.withColumn("DateTime", to_timestamp(col("DateTime"), "yyyy-MM-dd HH:mm:ss"))
        logging.info("DateTime cast completed.")

        # ── 7. Validate Vehicles range (must be > 0) ──────────────────────────
        invalid_vehicles = df.filter(col("Vehicles") <= 0).count()
        logging.info("Vehicles <= 0 (invalid rows): %d", invalid_vehicles)
        if invalid_vehicles > 0:
            df = df.filter(col("Vehicles") > 0)
            logging.info("Invalid Vehicles rows removed.")

        # ── 8. Validate Junction values (must be 1-4) ──────────────────────────
        invalid_junctions = df.filter(~col("Junction").isin(1, 2, 3, 4)).count()
        logging.info("Invalid Junction values: %d", invalid_junctions)
        if invalid_junctions > 0:
            df = df.filter(col("Junction").isin(1, 2, 3, 4))
            logging.info("Invalid Junction rows removed.")

        # ── 9. Feature engineering – time components ──────────────────────────
        logging.info("Adding time-based feature columns...")
        df = (
            df.withColumn("Year",       year(col("DateTime")))
              .withColumn("Month",      month(col("DateTime")))
              .withColumn("Hour",       hour(col("DateTime")))
              .withColumn("DayOfWeek",  dayofweek(col("DateTime")))   # 1=Sun … 7=Sat
        )
        logging.info("Feature columns added: Year, Month, Hour, DayOfWeek")

        # ── 10. Final schema ──────────────────────────────────────────────────
        logging.info("===================================")
        logging.info("Final Dataset Schema")
        logging.info("===================================")
        df.printSchema()

        # ── 11. Preview cleaned data ──────────────────────────────────────────
        logging.info("===================================")
        logging.info("Cleaned Dataset Preview (top 5 rows)")
        logging.info("===================================")
        df.show(5, truncate=False)

        # ── 12. Summary statistics ─────────────────────────────────────────────
        logging.info("===================================")
        logging.info("Summary Statistics")
        logging.info("===================================")
        df.select("Junction", "Vehicles", "Hour", "Month", "Year").describe().show()

        # ── 13. Save cleaned dataset ───────────────────────────────────────────
        output_path = "data/processed/clean_traffic_data"
        logging.info("Saving cleaned dataset to: %s", output_path)
        df.write.mode("overwrite").option("header", True).csv(output_path)
        logging.info("===================================")
        logging.info("Cleaned data saved successfully")
        logging.info("Path: %s", output_path)
        logging.info("Final row count  : %d", df.count())
        logging.info("Final column count: %d", len(df.columns))
        logging.info("===================================")
        logging.info("Data cleaning process completed successfully!")

    except Exception as e:
        logging.error("Error during data cleaning: %s", str(e))
        raise


if __name__ == "__main__":
    try:
        cleaning()
    except Exception as e:
        logging.error("Error during data cleaning: %s", str(e))