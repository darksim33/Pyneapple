import sys
from loguru import logger
import os

# remove the default logger to avoid duplicate logs
logger.remove()

# configurable log level and format
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
LOG_FORMAT = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"

# terminal loggger
logger.add(
    sys.stderr,
    format=LOG_FORMAT,
    level=LOG_LEVEL,
    colorize=True,
    backtrace=True,
    diagnose=True,
)

# Optional: file logger
# logger.add("logs/pyneapple.log", rotation="10 MB", retention="1 week", level=LOG_LEVEL)

# redirect stdout and stderr to logger
class InterceptOutput:
    def __init__(self, level="INFO"):
        self.level = level

    def write(self, message):
        if message.strip():  # Avoid logging empty messages
            getattr(logger, self.level.lower())(message.strip())

    def flush(self):
        pass

def intercept_stdout_stderr():
    # Redirect stdout and stderr to the logger
    sys.stdout = InterceptOutput(level="INFO")
    sys.stderr = InterceptOutput(level="ERROR")

def restore_stdout_stderr():
    # Restore stdout and stderr to their original state
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
