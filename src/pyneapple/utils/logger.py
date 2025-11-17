import sys
from loguru import logger
import os

# remove the default logger to avoid duplicate logs
logger.remove()

# configurable log level and format
DEFAULT_LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
DEBUG_LOG_FORMAT = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
INFO_LOG_FORMAT = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>"

if DEFAULT_LOG_LEVEL == "DEBUG":
    LOG_FORMAT = DEBUG_LOG_FORMAT
else:
    LOG_FORMAT = INFO_LOG_FORMAT

# terminal logger
_logger_id = logger.add(
    sys.stderr,
    format=LOG_FORMAT,
    level=DEFAULT_LOG_LEVEL,
    colorize=True,
    backtrace=True,
    diagnose=True,
)

# Optional: file logger
_LOG_TO_FILE = True
_logger_id_file = None
if _LOG_TO_FILE:
    _logger_id_file = logger.add(
        "logs/pyneapple.log",
        rotation="10 MB",
        retention="1 week",
        level=DEFAULT_LOG_LEVEL,
    )


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


def set_log_level(level):
    """
    Dynamically change the log level of the logger.

    Args:
        level (str): New log level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
    """

    global _logger_id, _logger_id_file, _LOG_TO_FILE

    if level == "DEBUG":
        LOG_FORMAT = DEBUG_LOG_FORMAT
    else:
        LOG_FORMAT = INFO_LOG_FORMAT

    # Remove all existing handlers
    logger.remove(_logger_id)

    # Add terminal logger with new level
    _logger_id = logger.add(
        sys.stderr,
        format=LOG_FORMAT,
        level=level,
        colorize=True,
        backtrace=True,
        diagnose=True,
    )

    if _LOG_TO_FILE and _logger_id_file is not None:
        # Remove file logger if it exists
        logger.remove(_logger_id_file)
        # Add file logger with new level
        _logger_id_file = logger.add(
            "logs/pyneapple.log", rotation="10 MB", retention="1 week", level=level
        )


def get_log_level():
    """
    Get the current log level.

    Returns:
        str: Current log level
    """

    global _logger_id

    if _logger_id in logger._core.handlers:
        level_no = logger._core.handlers[_logger_id].levelno
        for name, level in logger._core.levels.items():
            if level_no == level.no:
                return name

    # Fallback for old method in case the logger ID is not found
    for handler in logger._core.handlers.values():
        if handler._sink == sys.stderr:
            return handler._levelno_name

    return DEFAULT_LOG_LEVEL


def set_output_mode(mode="both", level=None):
    """
    Switch logger output mode between console, file, or both.
    
    Args:
        mode (str): Output mode - 'console', 'file', or 'both' (default: 'both')
        level (str): Optional log level to set. If None, keeps current level.
    """
    global _logger_id, _logger_id_file, _LOG_TO_FILE
    
    # Determine log level
    current_level = level if level else get_log_level()
    
    # Determine format
    if current_level == "DEBUG":
        log_format = DEBUG_LOG_FORMAT
    else:
        log_format = INFO_LOG_FORMAT
    
    # Remove existing handlers
    if _logger_id is not None:
        try:
            logger.remove(_logger_id)
            _logger_id = None
        except ValueError:
            pass
    
    if _logger_id_file is not None:
        try:
            logger.remove(_logger_id_file)
            _logger_id_file = None
        except ValueError:
            pass
    
    # Add handlers based on mode
    if mode in ["console", "both"]:
        _logger_id = logger.add(
            sys.stderr,
            format=log_format,
            level=current_level,
            colorize=True,
            backtrace=True,
            diagnose=True,
        )
    
    if mode in ["file", "both"]:
        _LOG_TO_FILE = True
        _logger_id_file = logger.add(
            "logs/pyneapple.log",
            rotation="10 MB",
            retention="1 week",
            level=current_level,
        )
    else:
        _LOG_TO_FILE = False


def get_output_mode():
    """
    Get the current output mode.
    
    Returns:
        str: Current output mode - 'console', 'file', 'both', or 'none'
    """
    global _logger_id, _logger_id_file
    
    has_console = _logger_id is not None and _logger_id in logger._core.handlers
    has_file = _logger_id_file is not None and _logger_id_file in logger._core.handlers
    
    if has_console and has_file:
        return "both"
    elif has_console:
        return "console"
    elif has_file:
        return "file"
    else:
        return "none"