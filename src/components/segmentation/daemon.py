from __future__ import annotations

from logging import getLogger
import os
import time
import signal
from pathlib import Path

from shared.args import DaemonArgs

from infrastructure.logging.setup import setup_logging

from .segmentation_builder import SegmentationBuilder, SegmentationResult
from .segmentation_repository import SegmentationRepository

DAEMON_NAME = 'Watcher Service: segmentation daemon'
logger = getLogger('segmentation_daemon')


def main() -> None:
    daemon_args = DaemonArgs.parse(description=DAEMON_NAME)
    setup_logging(daemon_args.logging)

    storage_dir = Path(daemon_args.storage.directory)
    storage_dir.mkdir(parents=True, exist_ok=True)

    daemon_pidfile = Path(storage_dir) / 'segmentator.pid'
    daemon_pidfile.write_text(str(os.getpid()))

    segmentation_builder = SegmentationBuilder(storage_dir=storage_dir)

    logger.info(f'Begin loop: {DAEMON_NAME}')
    try:
        _loop(daemon_pidfile)
    except Exception as e:
        if not isinstance(e, SystemExit) or e.code != 0:
            logger.critical('Unhandled exception', exc_info=True)
        raise
    finally:
        logger.info('End loop')


def chunk_segments(
        segments: list[list[str]],
        overlap: int,
        max_segments: int
) -> list[list[str]]:
    new_segments = []
    for s in segments:
        for i in range(0, len(s)-overlap, overlap):
            new_segments.append(s[i:i+max_segments])
    return new_segments


def _loop(
        pidfile: Path,
        segmentation_builder: SegmentationBuilder,
        segmentation_repository: SegmentationRepository
) -> None:
    def handle_interrupt(signum, frame) -> None:
        logger.info(f'Signal {signum} received')
        pidfile.unlink()
        exit(0)

    signal.signal(signal.SIGINT, handle_interrupt)
    signal.signal(signal.SIGTERM, handle_interrupt)

    while True:
        time.sleep(60*60)  # 1 hour
        items = segmentation_builder.pick_episodes()
        for item in items:
            segments = segmentation_builder.get_segments(item)
            if not segments:
                continue
            else:
                segments_text = [' '.join(s) for s in segments]
                chunked_segments = chunk_segments(segments, overlap=2, max_segments=10)
                segmentation_result = SegmentationResult(item, segments_text, chunked_segments)
                segmentation_repository.save(segmentation_result)
