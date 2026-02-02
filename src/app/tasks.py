import hashlib
import json
import logging
import re
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Generator

import feedparser
import requests
from celery import chain
from redis import Redis

from dotenv import load_dotenv

from src.app.app import app, settings
from src.shared.podcast_config import (
    load_podcasts_config,
    get_podcast_config,
)


load_dotenv()

# Redis client for distributed locking
redis_client = Redis.from_url(settings.broker_url)

# Key for tracking episodes currently being processed
EPISODES_PROCESSING_KEY = 'episodes:processing'


@contextmanager
def redis_lock(
    lock_name: str,
    timeout: int = 300,
) -> Generator[bool, None, None]:
    """
    Distributed lock using Redis.

    Args:
        lock_name: Unique name for the lock
        timeout: Lock expiration in seconds

    Yields:
        True if lock was acquired, False otherwise
    """
    lock = redis_client.lock(lock_name, timeout=timeout)
    acquired = lock.acquire(blocking=False)
    try:
        yield acquired
    finally:
        if acquired:
            try:
                lock.release()
            except Exception:
                pass  # Lock may have expired


def _episode_processing_key(
    podcast_slug: str, rss_guid: str
) -> str:
    return f'{podcast_slug}:{rss_guid}'


def is_episode_processing(
    podcast_slug: str, rss_guid: str
) -> bool:
    """Check if episode is already being processed."""
    key = _episode_processing_key(
        podcast_slug, rss_guid
    )
    return redis_client.sismember(
        EPISODES_PROCESSING_KEY, key
    )


def mark_episode_processing(
    podcast_slug: str,
    rss_guid: str,
    ttl: int = 86400,
) -> bool:
    """
    Mark episode as being processed.

    Returns:
        True if marked (wasn't already in set)
    """
    key = _episode_processing_key(
        podcast_slug, rss_guid
    )
    added = redis_client.sadd(
        EPISODES_PROCESSING_KEY, key
    )
    redis_client.expire(EPISODES_PROCESSING_KEY, ttl)
    return bool(added)


def unmark_episode_processing(
    podcast_slug: str, rss_guid: str
) -> None:
    """Remove episode from processing set."""
    key = _episode_processing_key(
        podcast_slug, rss_guid
    )
    redis_client.srem(EPISODES_PROCESSING_KEY, key)


logger = logging.getLogger(__name__)

STORAGE_DIR = Path(settings.episodes_storage_dir)


def _safe_dirname(
    rss_guid: str,
    published: str | None = None,
) -> str:
    """Create a filesystem-safe directory name from guid.

    Prefixes with publication date (YYYY-MM-DD) when
    available so directories sort chronologically.
    """
    guid_hash = hashlib.sha256(
        rss_guid.encode()
    ).hexdigest()[:16]
    if published:
        # published is ISO format: 2024-08-12T...
        date_prefix = published[:10]
        return f'{date_prefix}_{guid_hash}'
    return guid_hash


def _try_extract_episode_number(
    link: str,
) -> int | None:
    """Try to extract episode number from a URL."""
    matches = re.findall(r'/episode-(\d+)', link)
    if matches:
        return int(matches[0])
    return None


def _get_episode_dir(
    podcast_slug: str,
    rss_guid: str,
    episode_link: str | None = None,
    published: str | None = None,
) -> Path:
    """
    Get episode storage directory path.

    Uses episode number if extractable from link,
    otherwise a date-prefixed hash of the RSS guid.
    """
    episode_num = None
    if episode_link:
        episode_num = _try_extract_episode_number(
            episode_link
        )

    if episode_num is not None:
        dirname = str(episode_num)
    else:
        dirname = _safe_dirname(rss_guid, published)

    return STORAGE_DIR / podcast_slug / dirname


@app.task(
    name='src.app.tasks.check_rss_feed',
    bind=True,
    max_retries=3,
    default_retry_delay=60,
)
def check_rss_feed(self):
    """
    Celery Beat task that iterates over all configured
    podcasts and triggers per-podcast feed checks.
    """
    with redis_lock(
        'check_rss_feed_lock', timeout=300
    ) as acquired:
        if not acquired:
            logger.info(
                'RSS feed check already in progress, '
                'skipping'
            )
            return {
                'status': 'skipped',
                'reason': 'lock_not_acquired',
                'checked_at': (
                    datetime.now().isoformat()
                ),
            }

        try:
            configs = load_podcasts_config()
            results = []
            for podcast in configs.podcasts:
                check_podcast_feed.delay(podcast.slug)
                results.append(podcast.slug)

            return {
                'status': 'success',
                'podcasts_queued': results,
                'checked_at': (
                    datetime.now().isoformat()
                ),
            }
        except Exception as e:
            logger.error(
                f'Error checking RSS feeds: {e}',
                exc_info=True,
            )
            raise self.retry(exc=e)


@app.task(
    name='src.app.tasks.check_podcast_feed',
    bind=True,
    max_retries=3,
    default_retry_delay=60,
)
def check_podcast_feed(self, podcast_slug: str):
    """
    Check RSS feed for a single podcast and trigger
    processing chains for new episodes.
    """
    try:
        config = get_podcast_config(podcast_slug)

        logger.info(
            f'Checking RSS feed for {config.name}: '
            f'{config.rss_url}'
        )
        feed = feedparser.parse(config.rss_url)

        if feed.bozo:
            logger.warning(
                f'Feed parsing error for '
                f'{config.name}: '
                f'{feed.bozo_exception}'
            )
            return {
                'status': 'error',
                'podcast': podcast_slug,
                'message': str(feed.bozo_exception),
            }

        logger.info(
            f'Feed checked for {config.name}. '
            f'Found {len(feed.entries)} entries.'
        )

        new_episodes = []
        for entry in feed.entries:
            episode_data = _process_feed_entry(
                entry, config.slug
            )
            if episode_data:
                new_episodes.append(episode_data)

        for episode_data in new_episodes:
            _trigger_episode_processing(episode_data)

        return {
            'status': 'success',
            'podcast': podcast_slug,
            'new_episodes_count': len(new_episodes),
            'checked_at': datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(
            f'Error checking feed for '
            f'{podcast_slug}: {e}',
            exc_info=True,
        )
        raise self.retry(exc=e)


def _process_feed_entry(
    entry, podcast_slug: str
) -> dict[str, Any] | None:
    """
    Process a single RSS feed entry and return episode
    data if it's a new episode.

    Returns None if the entry should be skipped.
    """
    try:
        rss_guid = (
            getattr(entry, 'id', None)
            or getattr(entry, 'guid', None)
            or entry.link
        )

        # Extract mp3 from enclosures
        mp3_link = next(
            (
                e.href
                for e in getattr(
                    entry, 'enclosures', []
                )
                if 'audio' in e.get('type', '')
            ),
            None,
        )
        if not mp3_link:
            logger.debug(
                f'No audio enclosure in {entry.link}'
            )
            return None

        episode_link = getattr(entry, 'link', '')
        published = (
            datetime(
                *entry.published_parsed[:6]
            ).isoformat()
            if getattr(
                entry, 'published_parsed', None
            )
            else None
        )

        # Check if episode is fully processed
        episode_dir = _get_episode_dir(
            podcast_slug, rss_guid,
            episode_link, published,
        )
        if episode_dir.exists() and (episode_dir / 'segmentation_completed').exists():
            logger.debug(
                f'Episode already processed: '
                f'{episode_dir}'
            )
            return None

        # Check if episode is incomplete (has transcription but no segmentation)
        # This handles recovery after app restart
        if _is_episode_incomplete(episode_dir):
            logger.info(
                f'Found incomplete episode: {rss_guid} '
                f'(has transcription but no segmentation). '
                f'Attempting recovery...'
            )
            # Unmark from processing set if it's stuck there
            if is_episode_processing(podcast_slug, rss_guid):
                logger.info(
                    f'Unmarking stuck episode from processing set: '
                    f'{rss_guid}'
                )
                unmark_episode_processing(podcast_slug, rss_guid)

            # Re-mark as processing and trigger segmentation
            if mark_episode_processing(podcast_slug, rss_guid):
                episode_data = {
                    'podcast_slug': podcast_slug,
                    'rss_guid': rss_guid,
                    'title': getattr(entry, 'title', ''),
                    'episode_link': episode_link,
                    'mp3_link': mp3_link,
                    'published': published,
                    'summary': getattr(entry, 'summary', ''),
                    'authors': [
                        a.get('name', '')
                        for a in getattr(
                            entry, 'authors', []
                        )
                    ],
                    'episode_number': (
                        _try_extract_episode_number(
                            episode_link
                        )
                    ),
                    'episode_path': str(episode_dir),
                }
                _trigger_segmentation_only(episode_data)
                return None  # Don't trigger full chain
            else:
                logger.debug(
                    f'Episode recovery already in progress by '
                    f'another process: {rss_guid}'
                )
                return None

        # Check if already being processed (Redis)
        if is_episode_processing(podcast_slug, rss_guid):
            logger.debug(
                f'Episode already being processed: '
                f'{rss_guid}'
            )
            return None

        # Mark as processing atomically
        if not mark_episode_processing(
            podcast_slug, rss_guid
        ):
            logger.debug(
                f'Episode was just marked by another '
                f'process: {rss_guid}'
            )
            return None

        episode_data = {
            'podcast_slug': podcast_slug,
            'rss_guid': rss_guid,
            'title': getattr(entry, 'title', ''),
            'episode_link': episode_link,
            'mp3_link': mp3_link,
            'published': published,
            'summary': getattr(entry, 'summary', ''),
            'authors': [
                a.get('name', '')
                for a in getattr(
                    entry, 'authors', []
                )
            ],
            'episode_number': (
                _try_extract_episode_number(
                    episode_link
                )
            ),
        }

        logger.info(
            f'Found new episode for {podcast_slug}: '
            f'{entry.title}'
        )
        return episode_data

    except Exception as e:
        link = getattr(entry, 'link', 'unknown')
        logger.error(
            f'Error processing entry {link}: {e}',
            exc_info=True,
        )
        return None


def _is_episode_incomplete(
    episode_dir: Path,
) -> bool:
    """
    Check if episode is incomplete: has transcription but
    no segmentation_completed marker.

    Returns:
        True if episode is incomplete, False otherwise
    """
    if not episode_dir.exists():
        return False

    # Check if segmentation is completed
    if (episode_dir / 'segmentation_completed').exists():
        return False

    # Check if transcription exists
    transcription = next(
        episode_dir.glob('transcription-*.json'), None
    )
    return transcription is not None


def _trigger_episode_processing(
    episode_data: dict[str, Any],
) -> None:
    """Trigger the processing chain for a new episode."""
    logger.info(
        f'Triggering processing chain for '
        f'{episode_data["podcast_slug"]}:'
        f'{episode_data.get("title", "")}'
    )

    workflow = chain(
        download_episode_metadata.s(episode_data),
        download_episode_mp3.s(),
        transcribe_episode.s(),
        segment_episode.s(),
    )
    workflow.apply_async()


def _trigger_segmentation_only(
    episode_data: dict[str, Any],
) -> None:
    """
    Trigger segmentation only for an incomplete episode.

    Used for recovery when an episode has transcription
    but segmentation was not completed.
    """
    logger.info(
        f'Triggering segmentation recovery for '
        f'{episode_data["podcast_slug"]}:'
        f'{episode_data.get("title", "")}'
    )

    # Load episode.json if it exists to get full metadata
    episode_dir = Path(
        episode_data.get(
            'episode_path',
            str(
                _get_episode_dir(
                    episode_data['podcast_slug'],
                    episode_data['rss_guid'],
                    episode_data.get('episode_link', ''),
                    episode_data.get('published'),
                )
            ),
        )
    )

    if (episode_dir / 'episode.json').exists():
        try:
            episode_json = json.loads(
                (episode_dir / 'episode.json').read_text(
                    encoding='utf-8'
                )
            )
            # Merge with existing episode_data, prioritizing
            # episode.json values
            episode_data = {**episode_data, **episode_json}
        except Exception as e:
            logger.warning(
                f'Failed to load episode.json for '
                f'{episode_data["podcast_slug"]}:{episode_data["rss_guid"]}: '
                f'{e}'
            )

    # Ensure episode_path is set
    episode_data['episode_path'] = str(episode_dir)

    # Find transcription file
    transcription = next(
        episode_dir.glob('transcription-*.json'), None
    )
    if transcription:
        episode_data['transcription_path'] = str(transcription)

    # Trigger segmentation task directly
    segment_episode.delay(episode_data)


@app.task(
    name='src.app.tasks.download_episode_metadata',
    bind=True,
    max_retries=3,
    default_retry_delay=300,
)
def download_episode_metadata(
    self, episode_data: dict[str, Any]
) -> dict[str, Any]:
    """
    Save episode metadata from RSS feed data.

    Creates episode directory and saves episode.json.
    """
    podcast_slug = episode_data['podcast_slug']
    rss_guid = episode_data['rss_guid']
    title = episode_data.get('title', '')
    episode_link = episode_data.get('episode_link', '')
    published = episode_data.get('published')

    episode_dir = _get_episode_dir(
        podcast_slug, rss_guid,
        episode_link, published,
    )

    logger.info(
        f'Saving metadata for {podcast_slug}: {title}'
    )

    try:
        episode_dir.mkdir(parents=True, exist_ok=True)

        # Save episode.json with RSS-derived fields
        episode_json = {
            'podcast_slug': podcast_slug,
            'rss_guid': rss_guid,
            'title': title,
            'episode_link': episode_link,
            'mp3_link': episode_data.get(
                'mp3_link', ''
            ),
            'published': episode_data.get(
                'published', ''
            ),
            'summary': episode_data.get('summary', ''),
            'authors': episode_data.get('authors', []),
            'episode_number': episode_data.get(
                'episode_number'
            ),
        }
        episode_json_file = (
            episode_dir / 'episode.json'
        )
        episode_json_file.write_text(
            json.dumps(
                episode_json,
                indent=4,
                ensure_ascii=False,
            ),
            encoding='utf-8',
        )
        logger.info(
            f'Saved episode.json to {episode_dir}'
        )

        episode_data['episode_path'] = str(episode_dir)
        return episode_data

    except Exception as e:
        logger.error(
            f'Error saving metadata for '
            f'{podcast_slug}:{title}: {e}',
            exc_info=True,
        )
        raise self.retry(exc=e)


@app.task(
    name='src.app.tasks.download_episode_mp3',
    bind=True,
    max_retries=3,
    default_retry_delay=300,
)
def download_episode_mp3(
    self, episode_data: dict[str, Any]
) -> dict[str, Any]:
    """
    Download the MP3 file for an episode.

    Streams the MP3 file to disk to handle large files.
    Skips download if file already exists.
    """
    podcast_slug = episode_data['podcast_slug']
    rss_guid = episode_data['rss_guid']
    mp3_link = episode_data.get('mp3_link')
    episode_link = episode_data.get('episode_link', '')
    published = episode_data.get('published')

    episode_dir = Path(
        episode_data.get(
            'episode_path',
            str(
                _get_episode_dir(
                    podcast_slug,
                    rss_guid,
                    episode_link,
                    published,
                )
            ),
        )
    )
    mp3_path = episode_dir / 'episode.mp3'

    logger.info(
        f'Downloading MP3 for {podcast_slug}: '
        f'{episode_data.get("title", "")} '
        f'to {episode_dir}'
    )

    # Skip if already downloaded
    if mp3_path.exists():
        logger.info(
            f'MP3 already exists at {mp3_path}, '
            'skipping'
        )
        episode_data['mp3_path'] = str(mp3_path)
        return episode_data

    if not mp3_link:
        raise ValueError(
            f'No MP3 link provided for '
            f'{podcast_slug}:{rss_guid}'
        )

    try:
        episode_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f'Downloading from: {mp3_link}')
        response = requests.get(
            mp3_link, stream=True, timeout=600
        )
        response.raise_for_status()

        total_size = int(
            response.headers.get('content-length', 0)
        )
        if total_size:
            logger.info(
                f'MP3 file size: '
                f'{total_size / (1024 * 1024):.1f} MB'
            )

        downloaded = 0
        chunk_size = 8192
        with open(mp3_path, 'wb') as f:
            for chunk in response.iter_content(
                chunk_size=chunk_size
            ):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)

        logger.info(
            f'MP3 download completed for '
            f'{podcast_slug} at "{mp3_path}": '
            f'{downloaded / (1024 * 1024):.1f} MB'
        )

        episode_data['mp3_path'] = str(mp3_path)
        return episode_data

    except requests.RequestException as e:
        logger.error(
            f'Error downloading MP3 for '
            f'{podcast_slug}:{rss_guid}: {e}',
            exc_info=True,
        )
        if mp3_path.exists():
            mp3_path.unlink()
        raise self.retry(exc=e)


@app.task(
    name='src.app.tasks.transcribe_episode',
    bind=True,
    max_retries=2,
    default_retry_delay=600,
)
def transcribe_episode(
    self, episode_data: dict[str, Any]
) -> dict[str, Any]:
    """
    Transcribe episode audio via Deepgram.

    Uses podcast config for model and language settings.
    """
    from deepgram import (
        DeepgramClient,
        PrerecordedOptions,
        FileSource,
    )
    import httpx

    podcast_slug = episode_data['podcast_slug']
    rss_guid = episode_data['rss_guid']
    episode_link = episode_data.get('episode_link', '')
    published = episode_data.get('published')

    podcast_config = get_podcast_config(podcast_slug)

    episode_dir = Path(
        episode_data.get(
            'episode_path',
            str(
                _get_episode_dir(
                    podcast_slug,
                    rss_guid,
                    episode_link,
                    published,
                )
            ),
        )
    )
    mp3_path = episode_dir / 'episode.mp3'
    transcription_path = (
        episode_dir / 'transcription-deepgram.json'
    )

    logger.info(
        f'Transcribing {podcast_slug}: '
        f'{episode_data.get("title", "")}'
    )

    # Skip if transcription already exists
    existing = next(
        episode_dir.glob('transcription-*.json'), None
    )
    if existing:
        logger.info(
            f'Transcription already exists at '
            f'{existing}, skipping'
        )
        episode_data['transcription_path'] = str(
            existing
        )
        return episode_data

    lock_key = (
        f'transcribe_{podcast_slug}_{rss_guid}'
    )
    with redis_lock(
        lock_key, timeout=3600
    ) as acquired:
        if not acquired:
            logger.info(
                f'Transcription already in progress '
                f'for {podcast_slug}:{rss_guid}, '
                'skipping'
            )
            return episode_data

        # Re-check after lock
        existing = next(
            episode_dir.glob(
                'transcription-*.json'
            ),
            None,
        )
        if existing:
            logger.info(
                f'Transcription appeared while '
                f'waiting for lock: {existing}'
            )
            episode_data['transcription_path'] = str(
                existing
            )
            return episode_data

        if not mp3_path.exists():
            raise ValueError(
                f'MP3 file not found: {mp3_path}'
            )

        api_key = settings.deepgram_api_key
        if not api_key:
            raise ValueError(
                'DEEPGRAM_API_KEY not configured'
            )

        try:
            logger.info(
                f'Reading MP3 file: {mp3_path}'
            )
            with open(mp3_path, 'rb') as f:
                buffer_data = f.read()
            logger.info(
                f'MP3 file size: '
                f'{len(buffer_data) / (1024*1024):.1f}'
                ' MB'
            )

            client = DeepgramClient(api_key)
            options = PrerecordedOptions(
                model=podcast_config.transcription_model,
                language=podcast_config.language,
                paragraphs=True,
                diarize=True,
            )

            payload: FileSource = {
                'buffer': buffer_data,
            }

            logger.info(
                f'Sending to Deepgram API '
                f'(model={podcast_config.transcription_model}, '
                f'lang={podcast_config.language})'
            )
            response = (
                client.listen.prerecorded.v(
                    '1'
                ).transcribe_file(
                    source=payload,
                    options=options,
                    timeout=httpx.Timeout(
                        600.0, connect=30.0
                    ),
                )
            )

            logger.info(
                f'Saving transcription to: '
                f'{transcription_path}'
            )
            transcription_path.write_text(
                response.to_json(ensure_ascii=False),
                encoding='utf-8',
            )

            logger.info(
                f'Transcription completed for '
                f'{podcast_slug} at '
                f'{transcription_path}'
            )
            episode_data['transcription_path'] = str(
                transcription_path
            )
            return episode_data

        except Exception as e:
            logger.error(
                f'Error transcribing '
                f'{podcast_slug}:{rss_guid}: {e}',
                exc_info=True,
            )
            raise self.retry(exc=e)


@app.task(
    name='src.app.tasks.segment_episode',
    bind=True,
    max_retries=2,
    default_retry_delay=300,
)
def segment_episode(
    self, episode_data: dict[str, Any]
) -> dict[str, Any]:
    """
    Segment transcription using LLM and store in
    pgvector.

    This is the final task in the processing chain.
    """
    from src.components.segmentation.segmentation_builder import (
        SegmentationBuilder,
    )
    from src.components.segmentation.pgvector_repository import (
        DB,
    )

    podcast_slug = episode_data['podcast_slug']
    rss_guid = episode_data['rss_guid']
    episode_link = episode_data.get('episode_link', '')
    published = episode_data.get('published')

    podcast_config = get_podcast_config(podcast_slug)

    episode_dir = Path(
        episode_data.get(
            'episode_path',
            str(
                _get_episode_dir(
                    podcast_slug,
                    rss_guid,
                    episode_link,
                    published,
                )
            ),
        )
    )

    logger.info(
        f'Segmenting {podcast_slug}: '
        f'{episode_data.get("title", "")}'
    )

    # Check transcription exists
    transcription_path = episode_data.get(
        'transcription_path'
    )
    if not transcription_path:
        existing = next(
            episode_dir.glob('transcription-*.json'),
            None,
        )
        if not existing:
            raise FileNotFoundError(
                f'Transcription not found: '
                f'{episode_dir}'
            )
        transcription_path = str(existing)

    # Check LLM API configuration
    api_key = settings.segmentation_llm_api_key
    base_url = settings.segmentation_llm_api_url
    if not api_key or not base_url:
        raise ValueError(
            'SEGMENTATION_LLM_API_KEY and '
            'SEGMENTATION_LLM_API_URL '
            'must be configured'
        )

    lock_key = f'segment_{podcast_slug}_{rss_guid}'
    with redis_lock(
        lock_key, timeout=1800
    ) as acquired:
        if not acquired:
            logger.info(
                f'Segmentation already in progress '
                f'for {podcast_slug}:{rss_guid}, '
                'skipping'
            )
            return {
                'status': 'skipped',
                'podcast_slug': podcast_slug,
                'rss_guid': rss_guid,
                'reason': 'lock_not_acquired',
            }

        try:
            db = DB(
                host=settings.db_host,
                port=settings.db_port,
                dbname=settings.db_name,
                user=settings.db_user,
                password=settings.db_password,
                embedding_model=(
                    podcast_config.embedding_model
                ),
            )

            # Skip if already in database
            podcast_id = db.get_podcast_id(
                podcast_slug
            )
            if podcast_id and rss_guid:
                existing_episode = (
                    db.find_episode_by_guid(
                        podcast_id, rss_guid
                    )
                )
            else:
                ep_num = episode_data.get(
                    'episode_number'
                )
                existing_episode = (
                    db.find_episode(ep_num)
                    if ep_num
                    else None
                )

            if existing_episode:
                logger.info(
                    f'Episode already in database '
                    f'(id={existing_episode.id}), '
                    'skipping'
                )
                unmark_episode_processing(
                    podcast_slug, rss_guid
                )
                return {
                    'status': 'skipped',
                    'podcast_slug': podcast_slug,
                    'episode_id': str(
                        existing_episode.id
                    ),
                    'completed_at': (
                        datetime.now().isoformat()
                    ),
                }

            # Build segmentation
            logger.info(
                f'Building segments for '
                f'{podcast_slug}:{rss_guid}'
            )
            builder = SegmentationBuilder(
                storage_dir=STORAGE_DIR / podcast_slug,
                api_key=api_key,
                base_url=base_url,
                language=podcast_config.language,
                embedding_model=(
                    podcast_config.embedding_model
                ),
            )
            segmentation_result = builder.get_segments(
                episode_dir
            )

            logger.info(
                f'Created '
                f'{len(segmentation_result.segments)} '
                f'segments for '
                f'{podcast_slug}:{rss_guid}'
            )

            # Store in database
            logger.info(
                f'Storing segments in database for '
                f'{podcast_slug}:{rss_guid}'
            )
            episode_id = db.insert(
                segmentation_result,
                podcast_slug=podcast_slug,
                rss_guid=rss_guid,
                episode_link=episode_link,
                title=episode_data.get('title'),
                summary=episode_data.get(
                    'summary', ''
                ),
                authors=episode_data.get(
                    'authors', []
                ),
                published=episode_data.get(
                    'published'
                ),
            )

            if episode_id:
                logger.info(
                    f'Stored episode with id '
                    f'{episode_id}'
                )
                # Mark episode as fully processed
                marker = (
                    episode_dir / 'segmentation_completed'
                )
                marker.write_text(
                    datetime.now().isoformat()
                )
            else:
                logger.warning(
                    f'Failed to store episode '
                    f'{podcast_slug}:{rss_guid}'
                )

            unmark_episode_processing(
                podcast_slug, rss_guid
            )

            return {
                'status': 'completed',
                'podcast_slug': podcast_slug,
                'rss_guid': rss_guid,
                'episode_path': str(episode_dir),
                'episode_id': (
                    str(episode_id)
                    if episode_id
                    else None
                ),
                'segments_count': len(
                    segmentation_result.segments
                ),
                'completed_at': (
                    datetime.now().isoformat()
                ),
            }

        except Exception as e:
            logger.error(
                f'Error segmenting '
                f'{podcast_slug}:{rss_guid}: {e}',
                exc_info=True,
            )
            if (
                self.request.retries
                >= self.max_retries
            ):
                logger.warning(
                    f'Max retries reached for '
                    f'{podcast_slug}:{rss_guid}, '
                    'removing from processing set'
                )
                unmark_episode_processing(
                    podcast_slug, rss_guid
                )
            raise self.retry(exc=e)
        finally:
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
