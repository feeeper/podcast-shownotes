from __future__ import annotations

import argparse
import logging
from logging import _levelToName  # noqa
from typing import Generic, TypeVar, Union
from pathlib import Path

from pydantic import BaseModel, validator

DEFAULT_LOG_ROTATION_NUM_DAYS = 30

TSelf = TypeVar('TSelf', bound='ArgsBase')


class ArgsBase(Generic[TSelf]):
    @classmethod
    def setup(cls, parser: argparse.ArgumentParser) -> None:
        raise NotImplementedError

    @classmethod
    def read(cls, args: argparse.Namespace) -> TSelf:
        raise NotImplementedError

    @classmethod
    def parse(cls, *, description: str) -> TSelf:
        parser = argparse.ArgumentParser(description=description)
        cls.setup(parser)
        args = parser.parse_args()
        return cls.read(args)

    def forward(self) -> list[str]:
        raise NotImplementedError


class LoggingArgs(BaseModel, ArgsBase['LoggingArgs']):
    log_to_stderr: bool
    log_format_json: bool
    log_dir: Union[Path, None]
    log_min_level: str
    log_rotation_num_days: int

    @classmethod
    def setup(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument('--log-to-stderr', action='store_true')
        parser.add_argument(
            '--log-min-level',
            choices=_levelToName.values(),
            default=_levelToName[logging.NOTSET],
        )
        parser.add_argument(
            '--log-format',
            choices=['plain', 'json'],
            help='logging output format',
            default='plain',
        )
        parser.add_argument('--log-dir', type=Path, default=None)
        parser.add_argument(
            '--log-rotation-num-days',
            type=int,
            default=DEFAULT_LOG_ROTATION_NUM_DAYS,
        )

    @validator('log_dir', pre=True)
    def validate_log_dir(cls, v: Path | None) -> Path | None:
        if v is not None:
            v = v.resolve()
        return v

    @validator('log_rotation_num_days')
    def validate_log_rotation_num_days(cls, v: int) -> int:
        if v < 1:
            raise ValueError('log_rotation_num_days must be >= 1')
        return v

    @validator('log_min_level')
    def validate_log_min_level(cls, v: str) -> str:
        if v not in _levelToName.values():
            raise ValueError(f'log_min_level must be one of {_levelToName.values()}')
        return v

    def forward(self) -> list[str]:
        return [
            '--log-to-stderr' if self.log_to_stderr else '',
            f'--log-min-level={self.log_min_level}',
            f'--log-format={"json" if self.log_format_json else "plain"}',
            f'--log-dir={self.log_dir}' if self.log_dir is not None else '',
            f'--log-rotation-num-days={self.log_rotation_num_days}',
        ]

    @classmethod
    def read(cls, args: argparse.Namespace) -> 'LoggingArgs':
        return cls(
            log_to_stderr=args.log_to_stderr,
            log_format_json=args.log_format == 'json',
            log_dir=args.log_dir,
            log_min_level=args.log_min_level,
            log_rotation_num_days=args.log_rotation_num_days,
        )


class DaemonArgs(BaseModel, ArgsBase['DaemonArgs']):
    logging: LoggingArgs

    @classmethod
    def setup(cls, parser: argparse.ArgumentParser) -> None:
        LoggingArgs.setup(parser)

    @classmethod
    def read(cls, args: argparse.Namespace) -> DaemonArgs:
        return DaemonArgs(logging=LoggingArgs.read(args))

    def forward(self) -> list[str]:
        return [
            *self.logging.forward()
        ]


DaemonArgs.model_rebuild()

__all__ = [
    'ArgsBase',
    'LoggingArgs',
    'DaemonArgs'
]
