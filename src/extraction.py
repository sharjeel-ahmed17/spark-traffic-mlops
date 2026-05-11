import os
import pandas as pd
import kagglehub
from logger import logging

def data_extraction():
    try:
        # 1. Project folder set
        base_dir = os.getcwd()
        data_dir = os.path.join(base_dir, "data", "raw")
        os.makedirs(data_dir, exist_ok=True)
        logging.info("Using project folder: %s", base_dir)

        # 2. Dataset download using the updated method
        logging.info("Downloading traffic dataset...")
        # dataset_download returns the local path to the folder containing the files
        dataset_path = kagglehub.dataset_download("fedesoriano/traffic-prediction-dataset")
        
        # Locate the specific file (traffic.csv)
        file_name = "traffic.csv"
        source_path = os.path.join(dataset_path, file_name)

        # 3. Read with encoding fallback to fix the 'utf-8' error
        try:
            df = pd.read_csv(source_path)
            logging.info("File read successfully with default utf-8 encoding.")
        except UnicodeDecodeError:
            logging.warning("UTF-8 decoding failed. Attempting with ISO-8859-1 encoding...")
            df = pd.read_csv(source_path, encoding='ISO-8859-1')

        logging.info("Downloaded! Shape: %s", df.shape)
        logging.info("Columns: %s", df.columns.tolist())
        logging.info("First 5 records:\n%s", df.head())

        # 4. Save data to project folder 
        output_path = os.path.join(data_dir, "data.csv")
        df.to_csv(output_path, index=False, encoding='utf-8') # Save back as clean utf-8
        logging.info("Data saved at: %s", output_path)

        # 5. Final check
        logging.info("Final files in data/raw: %s", os.listdir(data_dir))
        logging.info("Data extraction process completed successfully!")

    except Exception as e:
        logging.error("Error during data extraction: %s", str(e))
    

if __name__ == "__main__":
    data_extraction()
