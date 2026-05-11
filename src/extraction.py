import os
import kagglehub
from kagglehub import KaggleDatasetAdapter
from logger import logging


def data_extraction():
    try:

        # 1. Project folder set
        base_dir = os.getcwd()
        data_dir = os.path.join(base_dir, "data/raw")
        os.makedirs(data_dir, exist_ok=True)
        logging.info("Using project folder: %s", base_dir)

        # 2. Dataset download as pandas
        logging.info("Downloading Telco Churn dataset...")
        df = kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            "fedesoriano/traffic-prediction-dataset",
            "traffic.csv"  
        )
        logging.info("Downloaded! Shape: %s", df.shape)
        logging.info("Columns: %s", df.columns.tolist())
        logging.info("First 5 records:")
        logging.info(df.head())

        # 3. Save data to  project folder 
        output_path = os.path.join(data_dir, "traffic.csv")
        df.to_csv(output_path, index=False)
        logging.info("Data saved at: %s", output_path)

        # 4. Final check
        logging.info("\nFinal files in data/raw:")
        logging.info(os.listdir(data_dir))
        logging.info("Data extraction process completed successfully!")
    except Exception as e:
        logging.error("Error during data extraction: %s", str(e))
    

if __name__ == "__main__":
    try:
        data_extraction()
    except Exception as e:
        logging.error("Error during data extraction: %s", str(e))
