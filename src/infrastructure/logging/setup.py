from __future__ import annotations

import os
from pathlib import Path

import logging
from logging.handlers import TimedRotatingFileHandler
from shared.args import LoggingArgs


def setup_logging(args: LoggingArgs) -> None:
    if isinstance(args.log_dir, Path) and not args.log_dir.exists():
        args.log_dir.mkdir(parents=True)

    rotation_logging_handler = TimedRotatingFileHandler(
        args.log_dir / 'watcher_error.log' if args.log_dir else '.log/watcher_error.log',
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

    logging.basicConfig(filename=args.log_dir / 'watcher.log', level=args.log_min_level)
    logging.root.addHandler(logging.StreamHandler())

    logger = logging.getLogger('watcher')

    logger.addHandler(rotation_logging_handler)


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
