import logging
import sys


def get_logger():
    logger = logging.getLogger('uvicorn')
    logger.setLevel(logging.DEBUG)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] [%(name)s] [%(funcName)s():%(lineno)d] [PID:%(process)d TID:%(thread)d] \n %(message)s',
        "%Y-%m-%d %H:%M:%S")
    handler.setFormatter(formatter)

    if (logger.hasHandlers()):
        logger.handlers.clear()

    logger.addHandler(handler)
    logger.propagate = False

    return logger