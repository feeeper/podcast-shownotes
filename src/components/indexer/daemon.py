# export PYTHONPATH="${PWD}/src"

from __future__ import annotations

from logging import getLogger
import os
import time
import signal
from pathlib import Path

from shared.args import DaemonArgs

from infrastructure.logging.setup import setup_logging

from .index_builder import IndexBuilder

DAEMON_NAME = 'Watcher Service: indexer daemon'
logger = getLogger('watcher_daemon')


def main() -> None:
    daemon_args = DaemonArgs.parse(description=DAEMON_NAME)
    setup_logging(daemon_args.logging)

    storage_dir = Path(daemon_args.storage.directory)
    storage_dir.mkdir(parents=True, exist_ok=True)

    daemon_pidfile = Path(storage_dir) / 'indexer.pid'
    daemon_pidfile.write_text(str(os.getpid()))

    index_builder = IndexBuilder(storage_dir=storage_dir)

    logger.info('Begin loop')
    try:
        _loop(
            daemon_pidfile,
            index_builder
        )
    except Exception as e:
        if not isinstance(e, SystemExit) or e.code != 0:
            logger.critical('Unhandled exception', exc_info=True)
        raise
    finally:
        logger.info('End loop')


def _loop(
        pidfile: Path,
        index_builder: IndexBuilder
) -> None:
    def handle_interrupt(signum, frame) -> None:
        logger.info(f'Signal {signum} received')
        pidfile.unlink()
        exit(0)

    signal.signal(signal.SIGINT, handle_interrupt)
    signal.signal(signal.SIGTERM, handle_interrupt)

    while True:
        items = index_builder.pick_episodes()
        if not items:
            time.sleep(1)
            continue
        for item in items:
            logger.info(f'Running item: {item}')
            input()


if __name__ == '__main__':
    main()