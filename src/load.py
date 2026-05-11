from logger import logging


def load_data():
    try:
        logging.info("Starting data loading process...")
        # Placeholder for data loading logic
        logging.info("Data loading  process completed successfully!")
    except Exception as e:
        logging.error("Error during data loading: %s", str(e))

if __name__ == "__main__":
    try:
        load_data()
    except Exception as e:
        logging.error("Error during data loading: %s", str(e))