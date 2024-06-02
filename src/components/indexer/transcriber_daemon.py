from __future__ import annotations

from logging import getLogger
import os
import time
import signal
from pathlib import Path

import requests
from shared.args import DaemonArgs

from infrastructure.logging.setup import setup_logging

from .index_builder import IndexBuilder, EpisodeMetadata
from .transcriber import Transcriber

DAEMON_NAME = 'Watcher Service: transcriber daemon'
logger = getLogger('transcriber _daemon')


def main() -> None:
    daemon_args = DaemonArgs.parse(description=DAEMON_NAME)
    setup_logging(daemon_args.logging)

    storage_dir = Path(daemon_args.storage.directory)
    storage_dir.mkdir(parents=True, exist_ok=True)

    daemon_pidfile = Path(storage_dir) / 'transcriber.pid'
    daemon_pidfile.write_text(str(os.getpid()))

    transcriber = Transcriber(storage_dir=storage_dir)

    logger.info(f'Begin loop: {DAEMON_NAME}')
    try:
        _loop(
            daemon_pidfile,
            transcriber
        )
    except Exception as e:
        if not isinstance(e, SystemExit) or e.code != 0:
            logger.critical('Unhandled exception', exc_info=True)
        raise
    finally:
        logger.info('End loop')


def _loop(
        pidfile: Path,
        transcriber: Transcriber
) -> None:
    def handle_interrupt(signum, frame) -> None:
        logger.info(f'Signal {signum} received')
        pidfile.unlink()
        exit(0)

    signal.signal(signal.SIGINT, handle_interrupt)
    signal.signal(signal.SIGTERM, handle_interrupt)

    while True:
        items = transcriber.pick_episodes()
        if not items:
            time.sleep(1)
            continue
        for item in items:
            transcriber.transcribe(item)
            logger.info(f'Running item: {item}')


if __name__ == '__main__':
    main()
