from __future__ import annotations
from logging import getLogger
import os
import time
import signal
from pathlib import Path
from src.shared.args import IndexerServerArgs, DbConnectionArgs
from src.infrastructure.logging.setup import setup_logging
from src.components.segmentation.segmentation_builder import SegmentationBuilder
from src.components.segmentation.pgvector_repository import DB

DAEMON_NAME = 'Watcher Service: segmentation daemon'
logger = getLogger('segmentation_daemon')


def main() -> None:
    daemon_args = IndexerServerArgs.parse(description=DAEMON_NAME)
    setup_logging(daemon_args.logging)

    storage_dir = Path(daemon_args.storage.directory)
    storage_dir.mkdir(parents=True, exist_ok=True)

    daemon_pidfile = Path(storage_dir) / 'segmentation.pid'
    daemon_pidfile.write_text(str(os.getpid()))

    segmentation_builder = SegmentationBuilder(storage_dir=storage_dir)
    db_connection_args: DbConnectionArgs = daemon_args.database_connection
    segmentation_repository = DB(
        host=db_connection_args.host,
        port=db_connection_args.port,
        dbname=db_connection_args.dbname,
        user=db_connection_args.user,
        password=db_connection_args.password,
        logging_args=daemon_args.logging,
    )

    logger.info(f'Begin loop: {DAEMON_NAME}')
    try:
        _loop(
            daemon_pidfile,
            segmentation_builder,
            segmentation_repository
        )
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
        segmentation_repository: DB
) -> None:
    def handle_interrupt(signum, frame) -> None:
        logger.info(f'Signal {signum} received')
        pidfile.unlink()
        exit(0)

    signal.signal(signal.SIGINT, handle_interrupt)
    signal.signal(signal.SIGTERM, handle_interrupt)

    while True:
        time.sleep(60 * 60)  # 1 hour
        items = segmentation_builder.pick_episodes()
        for item in items:
            if segmentation_repository.find_episode(int(item.stem)) is not None:
                Path(item / 'segmentation_completed').touch()
                Path(item / 'segmentation_in_progress').unlink(missing_ok=True)
                continue

            try:
                segmentation_result = segmentation_builder.get_segments(item)
                logger.info(f'Segmentation built for {item}')
                Path(item / 'segmentation_in_progress').touch()
                if segmentation_result:
                    episode_id = segmentation_repository.insert(segmentation_result)
                    if episode_id is not None:
                        logger.info(f'Segmentation saved for {item}')
                        Path(item / 'segmentation_completed').touch()
                        Path(item / 'segmentation_in_progress').unlink(missing_ok=True)
            except Exception as e:
                logger.error(f'Error processing {item}', exc_info=True)
                Path(item / 'segmentation_in_progress').unlink()
                if 'episode_id' in locals():
                    segmentation_repository.delete(episode_id)
                    Path(item / 'segmentation_completed').unlink()


if __name__ == '__main__':
    main()
