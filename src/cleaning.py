from logger import logging


def cleaning():
    try:
        logging.info("Starting data cleaning process...")
        # Placeholder for data cleaning logic
        logging.info("Data cleaning process completed successfully!")
    except Exception as e:
        logging.error("Error during data cleaning: %s", str(e))

if __name__ == "__main__":
    try:
        cleaning()
    except Exception as e:
        logging.error("Error during data cleaning: %s", str(e))