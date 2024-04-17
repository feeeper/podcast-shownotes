from __future__ import annotations

import argparse
from argparse import ArgumentParser

from pathlib import Path

from typing import TypeVar, Generic, Union
from pydantic import BaseModel, validator

TArgs = TypeVar('TArgs', bound='ArgsBase')


class ArgsBase(Generic[TArgs]):
    @classmethod
    def setup(cls: TArgs, parser: argparse.ArgumentParser) -> None:
        raise NotImplementedError

    @classmethod
    def read(cls: TArgs, args: argparse.Namespace) -> TArgs:
        raise NotImplementedError

    @classmethod
    def parse(cls: TArgs, *, description: str) -> TArgs:
        parser = argparse.ArgumentParser(description=description)
        cls.setup(parser)
        args = parser.parse_args()
        return cls.read(args)


class LoggingArgs(ArgsBase['LoggingArgs'], BaseModel):
    log_dir: Union[Path, None]
    log_min_level: str

    @classmethod
    def setup(cls, parser: ArgumentParser) -> None:
        parser.add_argument('--log-dir', type=Path, default=None)
        parser.add_argument('--log-min-level', type=str, default='INFO')

    @classmethod
    def read(cls, args: argparse.Namespace) -> LoggingArgs:
        return LoggingArgs(
            log_dir=args.log_dir,
            log_min_level=args.log_min_level
        )

    @validator('log_min_level')
    def validate_log_min_level(cls, value: str) -> str:
        if value not in ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'):
            raise ValueError('Invalid log level')
        return value
