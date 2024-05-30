import logging


# def get_logger(name: str, level: int = logging.INFO):
#     logger = logging.getLogger(name)
#     # set level
#     logger.setLevel(level)

class Logger(logging.Logger):
    def __init__(self, name, level=logging.INFO):
        super().__init__(name, level)

    def setup_logger(self, name: str = __name__, level: int = logging.INFO):
        logger = logging.getLogger(name)
        # set level
        logger.setLevel(level)

        # Formatting
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        # handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(level)

        file_handler = logging.FileHandler('pyneapple.log')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)

        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
