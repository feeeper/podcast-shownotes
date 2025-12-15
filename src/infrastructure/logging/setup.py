from __future__ import annotations

import os
from pathlib import Path

import logging
from logging.handlers import TimedRotatingFileHandler

import json_logging
from .file_handler import FileHandler
from .json_formatter import JsonFormatter
from src.shared.args import LoggingArgs



def setup_logging(args: LoggingArgs) -> None:
    assert not logging.root.hasHandlers()

    logging.root.setLevel(args.log_min_level)
    logging.root.addHandler(logging.StreamHandler())

    if args.log_dir is not None:
        file_handler = FileHandler(
            args.log_dir,
            '{date}.log',
            log_rotation_num_days=1,
        )
        logging.root.addHandler(file_handler)

    json_logging.init_non_web(
        enable_json=True, custom_formatter=JsonFormatter
    )


def setup_logger(logger: logging.Logger, log_dir: Path) -> None:
    rotation_logging_handler = TimedRotatingFileHandler(
        log_dir / f'{logger.name}.log' if log_dir else f'.log/{logger.name}.log',
        when='m',
        interval=1,
        backupCount=5)
    rotation_logging_handler.suffix = '%Y%m%d'
    rotation_logging_handler.namer = _get_filename
    rotation_logging_handler.setLevel(logging.ERROR)
    rotation_logging_handler.setFormatter(
        logging.Formatter(
            '{asctime} {levelname} {name} {message}', style='{'
        )
    )
    logger.addHandler(rotation_logging_handler)


def _get_filename(filename: str) -> str:
    # Get path to a log directory
    log_directory = os.path.split(filename)[0]

    # Extension of the filename is the suffix property
    # (in our case - %Y%m%d) (for example, .20181231).
    # We don't need a dot. Our file name will be `{suffix}.log` (ex: 20181231.log)
    date = os.path.splitext(filename)[1][1:]

    # New file name
    filename = os.path.join(log_directory, date)

    if not os.path.exists(f'{filename}.log'):
        return f'{filename}.log'

    # Looking for minimal `i` that we can use (file name with the same `i` that does not exist).
    index = 0
    new_filename = f'{filename}.{index}.log'
    while os.path.exists(new_filename):
        index += 1
        new_filename = f'{filename}.{index}.log'
    return new_filename
