# export PYTHONPATH="${PWD}/src"

from __future__ import annotations

from logging import getLogger
import os
import time
import signal
from pathlib import Path

from shared.args import DaemonArgs

from infrastructure.logging.setup import setup_logging

DAEMON_NAME = 'Watcher Service: indexer daemon'
logger = getLogger('watcher_daemon')


def main() -> None:
    daemon_args = DaemonArgs.parse(description=DAEMON_NAME)
    setup_logging(daemon_args.logging)

    daemon_pidfile = Path('./daemon')
    daemon_pidfile.write_text(str(os.getpid()))
    logger.info('Begin loop')
    storage_dir = Path(daemon_args.storage.directory)
    try:
        _loop(daemon_pidfile)
    except Exception as e:
        if not isinstance(e, SystemExit) or e.code != 0:
            logger.critical('Unhandled exception', exc_info=True)
        raise
    finally:
        logger.info('End loop')


def _loop(
        pidfile: Path
) -> None:
    def handle_interrupt(signum, frame) -> None:
        logger.info(f'Signal {signum} received')
        pidfile.unlink()
        exit(0)

    signal.signal(signal.SIGINT, handle_interrupt)
    signal.signal(signal.SIGTERM, handle_interrupt)

    while True:
        items = []
        if not items:
            time.sleep(1)
            continue
        for task in items:
            logger.info(f'Running task: {task}')


if __name__ == '__main__':
    main()
