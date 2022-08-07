import sys
import logging

import numpy as np


def fix_seeds():
    np.random.seed(2022)


def initialize_logger(logger):
    logger.setLevel(logging.INFO)
    log_formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s %(module)s.%(funcName)s"
        "[#%(lineno)s]: %(message)s")
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)
