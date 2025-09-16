# --------------------------------------------------------
# WoundAmbit
# Copyright (c) 2025 Vanessa Borst, Timo Dittus and Contributors.
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import logging
import os
import sys

import tqdm


class TqdmFilter(logging.Filter):
    def filter(self, record):
        return not record.getMessage().startswith('\r')


class TqdmLoggingHandler(logging.StreamHandler):
    def __init__(self, stream=None):
        super().__init__(stream)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg, file=self.stream)
            self.flush()
        except Exception:
            self.handleError(record)


def setup_logger(log_file_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Filter out tqdm status bar updates
    tqdm_filter = TqdmFilter()

    # Set up console logging handler
    console_handler = TqdmLoggingHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.addFilter(tqdm_filter)

    # Set up file logging handler
    file_handler = logging.FileHandler(log_file_path, mode='w')
    file_handler.setLevel(logging.INFO)
    file_handler.addFilter(tqdm_filter)

    # Set log format
    log_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(log_format)
    file_handler.setFormatter(log_format)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


def create_custom_logger(name: str, log_file: str, level=logging.INFO) -> logging.Logger:
    assert os.path.isabs(log_file), f"Path must be absolute: {log_file}"

    logger = logging.getLogger(name)
    logger.setLevel(level)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s]: %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def flatten_dict(d: dict, parent_key: str = '', sep: str = '_') -> dict:
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            for i, elem in enumerate(v):
                if isinstance(elem, dict):
                    items.extend(flatten_dict(elem, f"{new_key}{sep}{i}", sep=sep).items())
                else:
                    items.append((f"{new_key}{sep}{i}", elem))
        else:
            items.append((new_key, v))
    return dict(items)
