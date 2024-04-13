from __future__ import annotations
from logging import getLogger
import os
from pathlib import Path
import time
import signal
from shared.args import DaemonArgs
from infrastructure.logging import setup_logging, flush_logs


DAEMON_NAME = 'Podcast Indexer: indexer daemon'

logger = getLogger(__spec__.name)  # type: ignore[name-defined]

def main() -> None:
    daemon_args = DaemonArgs.parse(description=DAEMON_NAME)
    setup_logging(daemon_args.logging)

    logger.info('Begin loop')
    try:
        _loop(daemon_args)
    except Exception as err:
        if not isinstance(err, SystemExit) or err.code != 0:
            logger.critical('Unhandled exception', exc_info=True)
        raise
    finally:
        flush_logs()


def _loop(daemon_args: DaemonArgs) -> None:
    def handle_interrupt(signum, frame) -> None:
        logger.info(f'Signal {signum} received')
        daemon_args.storage.indexer_daemon_pidfile.path.unlink()
        exit(0)

    signal.signal(signal.SIGINT, handle_interrupt)
    signal.signal(signal.SIGTERM, handle_interrupt)

    while True:
        time.sleep(daemon_args.sleep_interval)
        logger.info('Loop iteration')
        # Do work here