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

    def forward(self: TArgs) -> list[str]:
        raise NotImplementedError


class IndexerServerArgs(BaseModel, ArgsBase['IndexerServerArgs']):
    storage: StorageArgs
    logging: LoggingArgs
    port: int

    @validator('port')
    def port_number_is_16bit_unsigned_int(cls, v: int) -> int:
        assert 0 <= v < 2 ** 16, v
        return v

    @classmethod
    def setup(cls, parser: ArgumentParser) -> None:
        StorageArgs.setup(parser)
        LoggingArgs.setup(parser)
        parser.add_argument('--port', type=int, default=8080)

    @classmethod
    def read(cls, args: argparse.Namespace) -> IndexerServerArgs:
        return IndexerServerArgs(
            storage=StorageArgs.read(args),
            logging=LoggingArgs.read(args),
            port=args.port
        )

    def forward(self) -> list[str]:
        return [
            *self.storage.forward(),
            *self.logging.forward(),
            f'--port={self.port}'
        ]


class LoggingArgs(BaseModel, ArgsBase['LoggingArgs']):
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

    def forward(self) -> list[str]:
        return [
            f'--log-dir={self.log_dir}',
            f'--log-min-level={self.log_min_level}'
        ]


class StorageArgs(BaseModel, ArgsBase['StorageArgs']):
    directory: Path

    @classmethod
    def setup(cls, parser: ArgumentParser) -> None:
        parser.add_argument(
            '--storage-dir',
            help='Directory to storing interim data',
            type=Path,
            required=True)

    @classmethod
    def read(cls, args: argparse.Namespace) -> StorageArgs:
        return StorageArgs(
            directory=args.storage_dir
        )

    def forward(self) -> list[str]:
        return [
            f'--storage-dir={self.directory}'
        ]


class DaemonArgs(BaseModel, ArgsBase['DaemonArgs']):
    logging: LoggingArgs
    storage: StorageArgs

    @classmethod
    def setup(cls, parser: ArgumentParser) -> None:
        LoggingArgs.setup(parser)
        StorageArgs.setup(parser)

    @classmethod
    def read(cls, args: argparse.Namespace) -> DaemonArgs:
        return DaemonArgs(
            logging=LoggingArgs.read(args),
            storage=StorageArgs.read(args)
        )

    def forward(self) -> list[str]:
        return [
            *self.logging.forward(),
            *self.storage.forward(),
        ]


__all__ = [
    'ArgsBase',
    'LoggingArgs',
    'DaemonArgs',
]