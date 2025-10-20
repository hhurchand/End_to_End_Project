from loguru import logger


# LOGGING CONFIGURATION
LOG_FILE_PATH = "logs/logfile.log"

# ROTATION : NEW FILE WHEN SIZE REACHES 1 MB | RETENTION : KEEP LOGS FOR 10 DAYS | LEVEL : ONLY LOG ERRORS OR HIGHER
logger.add(LOG_FILE_PATH, rotation="1 MB", retention="10 days", level="ERROR")
