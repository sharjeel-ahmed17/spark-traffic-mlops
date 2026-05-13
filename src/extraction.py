import os
import pandas as pd
import kagglehub
from logger import logging
from config_utils import load_config

def data_extraction():
    try:
        config, params = load_config()

        # 1. Project folder set
        data_dir = config["data"]["raw_dir"]
        os.makedirs(data_dir, exist_ok=True)
        logging.info("Using data directory: %s", data_dir)

        # 2. Dataset download using the updated method
        logging.info("Downloading traffic dataset...")
        dataset_id = params["extraction"]["dataset_id"]
        file_name  = params["extraction"]["file_name"]
        
        # dataset_download returns the local path to the folder containing the files
        dataset_path = kagglehub.dataset_download(dataset_id)
        
        # Locate the specific file (traffic.csv)
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

        # 3. Save data to  project folder 
        output_path = config["data"]["raw_file"]
        df.to_csv(output_path, index=False)
        logging.info("Data saved at: %s", output_path)

        # 5. Final check
        logging.info("Final files in data/raw: %s", os.listdir(data_dir))
        logging.info("Data extraction process completed successfully!")

    except Exception as e:
        logging.error("Error during data extraction: %s", str(e))
    

if __name__ == "__main__":
    data_extraction()
