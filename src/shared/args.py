from __future__ import annotations

import argparse
from argparse import ArgumentParser

from pathlib import Path

from typing import TypeVar, Generic, Union
from enum import Enum
from pydantic import BaseModel, validator


class Provider(Enum):
    DEEPGRAM = 'deepgram'
    OPENAI = 'openai'

    def __str__(self) -> str:
        return self.value


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
    transcription: TranscriptionArgs
    database_connection: DbConnectionArgs
    port: int

    @validator('port')
    def port_number_is_16bit_unsigned_int(cls, v: int) -> int:
        assert 0 <= v < 2 ** 16, v
        return v

    @classmethod
    def setup(cls, parser: ArgumentParser) -> None:
        StorageArgs.setup(parser)
        LoggingArgs.setup(parser)
        TranscriptionArgs.setup(parser)
        DbConnectionArgs.setup(parser)
        parser.add_argument('--port', type=int, default=8080)

    @classmethod
    def read(cls, args: argparse.Namespace) -> IndexerServerArgs:
        return IndexerServerArgs(
            storage=StorageArgs.read(args),
            logging=LoggingArgs.read(args),
            transcription=TranscriptionArgs.read(args),
            database_connection=DbConnectionArgs.read(args),
            port=args.port,
        )

    def forward(self) -> list[str]:
        return [
            *self.storage.forward(),
            *self.logging.forward(),
            f'--port={self.port}',
            *self.database_connection.forward(),
            *self.transcription.forward(),
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


class TranscriptionArgs(BaseModel, ArgsBase['TranscriptionArgs']):
    api_key: str
    provider: Provider
    debug: bool = False

    @classmethod
    def setup(cls, parser: ArgumentParser) -> None:
        parser.add_argument('--api-key', type=str, default='KEY_NOT_PASSED')
        parser.add_argument('--provider', type=Provider, default=Provider.DEEPGRAM, choices=list(Provider))
        parser.add_argument('--debug', type=bool, default=True)

    @classmethod
    def read(cls, args: argparse.Namespace) -> TranscriptionArgs:
        return TranscriptionArgs(
            api_key=args.api_key,
            provider=args.provider,
            debug=args.debug,
        )

    def forward(self) -> list[str]:
        return [
            f'--api-key={self.api_key}',
            f'--provider={self.provider}',
            f'--debug={self.debug}',
        ]


class DbConnectionArgs(BaseModel, ArgsBase['DbConnectionArgs']):
    host: str
    port: int
    dbname: str
    user: str
    password: str

    @classmethod
    def setup(cls, parser: ArgumentParser) -> None:
        parser.add_argument('--dbhost', type=str, default='localhost')
        parser.add_argument('--dbport', type=int, default=5432)
        parser.add_argument('--dbname', type=str, default='podcast_shownotes')
        parser.add_argument('--dbuser', type=str, default='postgres')
        parser.add_argument('--dbpassword', type=str, default='postgres')

    @classmethod
    def read(cls, args: argparse.Namespace) -> DbConnectionArgs:
        return DbConnectionArgs(
            host=args.dbhost,
            port=args.dbport,
            dbname=args.dbname,
            user=args.dbuser,
            password=args.dbpassword,
        )

    def forward(self: TArgs) -> list[str]:
        return [
            f'--dbhost={self.host}',
            f'--dbport={self.port}',
            f'--dbname={self.dbname}',
            f'--dbuser={self.user}',
            f'--dbpassword={self.password}',
        ]

__all__ = [
    'ArgsBase',
    'IndexerServerArgs',
    'LoggingArgs',
    'DaemonArgs',
    'TranscriptionArgs',
    'DbConnectionArgs',
    'Provider',
]
