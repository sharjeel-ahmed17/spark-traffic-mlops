import logging
import os
from logging.handlers import TimedRotatingFileHandler

# Create logs directory
LOGS_DIR = os.path.join(os.getcwd(), "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

# Base log file name
LOG_FILE_PATH = os.path.join(LOGS_DIR, "running_logs.log")

# Configure the daily rotating handler
# 'when="midnight"' ensures a new file is created every day
# 'backupCount=30' keeps logs for 30 days (change or remove as needed)
file_handler = TimedRotatingFileHandler(
    LOG_FILE_PATH, 
    when="midnight", 
    interval=1, 
    backupCount=30,
    encoding="utf-8"
)

# Set the date format for the old log files (e.g., running_logs.log.2026-05-19)
file_handler.suffix = "%Y-%m-%d"

# Configure logging system
logging.basicConfig(
    format="[ %(asctime)s ] %(filename)s:%(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    handlers=[
        file_handler, 
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("MLOpsWorkflow")

def log_info(message):
    logger.info(message)
    
def log_warning(message):
    logger.warning(message)

def log_error(message):
    logger.error(message)

def log_debug(message):
    logger.debug(message)

if __name__ == "__main__":
    log_info("Logger initialized with daily rotation.")
