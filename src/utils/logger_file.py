from loguru import logger

LOG_FILE_PATH = "logfile.log"
logger.add(LOG_FILE_PATH,rotation='1 MB',retention="10 days",level="ERROR")
