from __future__ import annotations

import logging

import json_logging
from .file_handler import FileHandler
from .json_formatter import JsonFormatter
from shared.args import LoggingArgs


def setup_logging(args: LoggingArgs) -> None:
    assert not logging.root.hasHandlers()

    logging.root.setLevel(args.log_min_level)

    if args.log_to_stderr:
        logging.root.addHandler(logging.StreamHandler())

    if args.log_dir is not None:
        file_handler = FileHandler(
            args.log_dir,
            '{date}.log',
            log_rotation_num_days=args.log_rotation_num_days,
        )
        if not args.log_format_json:
            file_handler.setFormatter(
                logging.Formatter(
                    '{asctime} {levelname} {name} {message}', style='{'
                )
            )
        logging.root.addHandler(file_handler)

    if args.log_format_json:
        json_logging.init_non_web(
            enable_json=True, custom_formatter=JsonFormatter
        )


def flush_logs() -> None:
    for handler in logging.root.handlers:
        handler.flush()


__all__ = ['setup_logging', 'flush_logs', 'FileHandler']
