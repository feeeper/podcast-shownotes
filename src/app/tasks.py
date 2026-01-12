import json
import logging
import re
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Generator

import bs4
import feedparser
import requests
from celery import chain
from redis import Redis

from dotenv import load_dotenv

from src.app.app import app, settings


load_dotenv()

# Redis client for distributed locking
# Uses the same Redis instance as Celery broker
redis_client = Redis.from_url(settings.broker_url)

# Key for tracking episodes currently being processed
EPISODES_PROCESSING_KEY = "episodes:processing"


@contextmanager
def redis_lock(
    lock_name: str,
    timeout: int = 300
) -> Generator[bool, None, None]:
    """
    Distributed lock using Redis.

    Args:
        lock_name: Unique name for the lock
        timeout: Lock expiration in seconds (prevents deadlock if process dies)

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


def is_episode_processing(episode_num: int) -> bool:
    """Check if episode is already being processed."""
    return redis_client.sismember(EPISODES_PROCESSING_KEY, str(episode_num))


def mark_episode_processing(episode_num: int, ttl: int = 86400) -> bool:
    """
    Mark episode as being processed.

    Args:
        episode_num: Episode number to mark
        ttl: Time-to-live in seconds (default 24h, prevents stale entries)

    Returns:
        True if marked (wasn't already in set), False if already existed
    """
    added = redis_client.sadd(EPISODES_PROCESSING_KEY, str(episode_num))
    redis_client.expire(EPISODES_PROCESSING_KEY, ttl)
    return bool(added)


def unmark_episode_processing(episode_num: int) -> None:
    """Remove episode from processing set."""
    redis_client.srem(EPISODES_PROCESSING_KEY, str(episode_num))

logger = logging.getLogger(__name__)

STORAGE_DIR = Path(settings.episodes_storage_dir)

FEED_URL = "https://devzen.ru/feed/"
NON_EPISODE_LINK_PREFIXES = (
    "https://devzen.ru/themes",
    "https://devzen.ru/no-themes",
)


@app.task(
    name="src.app.tasks.check_rss_feed",
    bind=True,
    max_retries=3,
    default_retry_delay=60,
)
def check_rss_feed(self):
    """
    Celery Beat task that checks the RSS feed at https://devzen.ru/feed/
    and triggers processing chains for new episodes.

    Uses a distributed Redis lock to ensure only one instance runs at a time,
    preventing duplicate processing chains when multiple Beat instances exist.
    """
    with redis_lock("check_rss_feed_lock", timeout=300) as acquired:
        if not acquired:
            logger.info("RSS feed check already in progress, skipping")
            return {
                "status": "skipped",
                "reason": "lock_not_acquired",
                "checked_at": datetime.now().isoformat(),
            }

        try:
            logger.info(f"Checking RSS feed: {FEED_URL}")
            feed = feedparser.parse(FEED_URL)

            if feed.bozo:
                logger.warning(f"Feed parsing error: {feed.bozo_exception}")
                return {"status": "error", "message": str(feed.bozo_exception)}

            feed_info = {
                "title": feed.feed.get("title", "Unknown"),
                "link": feed.feed.get("link", ""),
                "updated": feed.feed.get("updated", ""),
                "entry_count": len(feed.entries),
            }

            logger.info(
                f"Feed checked successfully. "
                f"Found {feed_info['entry_count']} entries."
            )

            new_episodes = []
            for entry in feed.entries:
                episode_data = _process_feed_entry(entry)
                if episode_data:
                    new_episodes.append(episode_data)

            # Trigger processing chain for each new episode
            for episode_data in new_episodes:
                _trigger_episode_processing(episode_data)

            return {
                "status": "success",
                "feed_info": feed_info,
                "new_episodes_count": len(new_episodes),
                "new_episodes": [ep["episode_number"] for ep in new_episodes],
                "checked_at": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error checking RSS feed: {e}", exc_info=True)
            raise self.retry(exc=e)


def _process_feed_entry(entry) -> dict[str, Any] | None:
    """
    Process a single RSS feed entry and return episode data if it's a new episode.

    Returns None if the entry should be skipped:
    - Non-episode entries (themes, etc.)
    - Episode already exists on disk
    - Episode already being processed (tracked in Redis)
    """
    try:
        # Skip non-episode entries
        if entry.link.startswith(NON_EPISODE_LINK_PREFIXES):
            logger.debug(f"Episode {entry.link} is not an episode")
            return None

        # Extract episode number from link
        matches = re.findall(r"/episode-(\d+)", entry.link)
        if not matches:
            logger.debug(f"Episode {entry.link} is not a valid episode link")
            return None

        episode_num = int(matches[0])
        episode_path = STORAGE_DIR / str(episode_num)

        # Check if episode already exists with MP3
        if episode_path.exists() and (episode_path / "episode.mp3").exists():
            logger.debug(f"Episode {episode_num} already exists on disk")
            return None

        # Check if episode is already being processed (Redis set)
        if is_episode_processing(episode_num):
            logger.debug(f"Episode {episode_num} already being processed")
            return None

        # Mark episode as processing BEFORE returning
        # This is atomic - if another process marked it first, we skip
        if not mark_episode_processing(episode_num):
            logger.debug(
                f"Episode {episode_num} was just marked by another process"
            )
            return None

        # Build episode data
        episode_data = {
            "episode_number": episode_num,
            "title": entry.title,
            "episode_link": entry.link,
            "mp3_link": (
                entry.enclosures[0].href if entry.enclosures else None
            ),
            "published": (
                datetime(*entry.published_parsed[:6]).isoformat()
                if entry.published_parsed
                else None
            ),
            "summary": entry.summary,
            "authors": (
                [author.name for author in entry.authors]
                if hasattr(entry, "authors")
                else []
            ),
            "html_content": (
                entry.content[0].value if entry.content else ""
            ),
        }

        logger.info(f"Found new episode: {episode_num} - {entry.title}")
        return episode_data

    except Exception as e:
        logger.error(
            f"Error processing entry {entry.link}: {e}", exc_info=True
        )
        return None


def _trigger_episode_processing(episode_data: dict[str, Any]) -> None:
    """Trigger the processing chain for a new episode."""
    episode_num = episode_data["episode_number"]
    logger.info(f"Triggering processing chain for episode {episode_num}")

    workflow = chain(
        download_episode_metadata.s(episode_data),
        download_episode_mp3.s(),
        transcribe_episode.s(),
        segment_episode.s(),
    )
    workflow.apply_async()


@app.task(
    name="src.app.tasks.download_episode_metadata",
    bind=True,
    max_retries=3,
    default_retry_delay=300,
)
def download_episode_metadata(
    self, episode_data: dict[str, Any]
) -> dict[str, Any]:
    """
    Download and save episode metadata.

    Creates episode directory and saves:
    - episode.json: Full episode data from RSS feed
    - episode.html: HTML content from RSS feed
    - metadata.json: Parsed metadata from episode page (speakers, shownotes, etc.)
    """
    episode_num = episode_data["episode_number"]
    episode_path = STORAGE_DIR / str(episode_num)

    logger.info(f"Downloading metadata for episode {episode_num}")

    try:
        # Create episode directory
        episode_path.mkdir(parents=True, exist_ok=True)

        # Save episode.html
        html_content = episode_data.get("html_content", "")
        if html_content:
            html_file = episode_path / "episode.html"
            html_file.write_text(html_content, encoding="utf-8")
            logger.info(f"Saved episode.html for episode {episode_num}")

        # Save episode.json
        episode_json = {
            "title": episode_data.get("title", ""),
            "episode_link": episode_data.get("episode_link", ""),
            "mp3_link": episode_data.get("mp3_link", ""),
            "episode": episode_num,
            "published": episode_data.get("published", ""),
            "summary": episode_data.get("summary", ""),
            "authors": episode_data.get("authors", []),
            "html_content": html_content,
            "path": str(episode_path),
        }
        episode_json_file = episode_path / "episode.json"
        episode_json_file.write_text(
            json.dumps(episode_json, indent=4, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info(f"Saved episode.json for episode {episode_num}")

        # Fetch and save metadata.json from episode page
        episode_link = episode_data.get("episode_link", "")
        if episode_link:
            metadata = _fetch_episode_metadata(episode_link)
            metadata_file = episode_path / "metadata.json"
            metadata_file.write_text(
                json.dumps(metadata, indent=4, ensure_ascii=False),
                encoding="utf-8",
            )
            logger.info(f"Saved metadata.json for episode {episode_num}")

        episode_data["episode_path"] = str(episode_path)
        return episode_data

    except Exception as e:
        logger.error(
            f"Error downloading metadata for episode {episode_num}: {e}",
            exc_info=True,
        )
        raise self.retry(exc=e)


def _fetch_episode_metadata(episode_link: str) -> dict[str, Any]:
    """
    Fetch and parse metadata from the episode page.

    Returns dict with: release_date, title, shownotes, speakers, music, mp3
    """
    response = requests.get(episode_link, timeout=30)
    response.raise_for_status()

    parser = bs4.BeautifulSoup(response.text, "html.parser")

    # Extract release date
    time_elem = parser.find("time", class_="entry-date")
    release_date = time_elem.text if time_elem else ""

    # Extract title
    title_elem = parser.find("h1", class_="entry-title")
    title = title_elem.text if title_elem else ""

    # Extract shownotes and content div
    content_div = parser.find("div", class_="entry-content clearfix")
    shownotes = content_div.text if content_div else ""

    # Extract speakers
    speakers = _extract_speakers(content_div)

    # Extract music
    music = _extract_music(content_div)

    # Extract mp3 link
    mp3_elem = parser.find("a", class_="powerpress_link_d")
    mp3 = mp3_elem.attrs.get("href", "") if mp3_elem else ""

    return {
        "release_date": release_date,
        "title": title,
        "shownotes": shownotes,
        "speakers": speakers,
        "music": music,
        "mp3": mp3,
    }


def _extract_speakers(content_div) -> list[dict[str, str]]:
    """Extract speaker information from episode content."""
    if not content_div:
        return []

    speakers = []
    try:
        paragraphs = content_div.find_all("p")
        for p in paragraphs:
            if "голоса выпуска" in p.text.lower():
                for a in p.find_all("a"):
                    speakers.append({
                        "name": a.text,
                        "href": a.attrs.get("href", ""),
                    })
                break
    except Exception:
        pass

    return speakers


def _extract_music(content_div) -> dict[str, str]:
    """Extract background music information from episode content."""
    if not content_div:
        return {}

    try:
        paragraphs = content_div.find_all("p")
        for p in paragraphs:
            if "фоновая музыка" in p.text.lower():
                music_anchor = p.find("a")
                if music_anchor:
                    return {
                        "name": music_anchor.text,
                        "href": music_anchor.attrs.get("href", ""),
                    }
                break
    except Exception:
        pass

    return {}


@app.task(
    name="src.app.tasks.download_episode_mp3",
    bind=True,
    max_retries=3,
    default_retry_delay=300,
)
def download_episode_mp3(
    self, episode_data: dict[str, Any]
) -> dict[str, Any]:
    """
    Download the MP3 file for an episode.

    Streams the MP3 file to disk to handle large files efficiently.
    Skips download if file already exists.
    """
    episode_num = episode_data["episode_number"]
    mp3_link = episode_data.get("mp3_link")
    episode_path = STORAGE_DIR / str(episode_num)
    mp3_path = episode_path / "episode.mp3"

    logger.info(f"Downloading MP3 for episode {episode_num} to {episode_path}")

    # Skip if already downloaded
    if mp3_path.exists():
        logger.info(f"MP3 already exists for episode {episode_num}, skipping")
        episode_data["mp3_path"] = str(mp3_path)
        return episode_data

    if not mp3_link:
        raise ValueError(f"No MP3 link provided for episode {episode_num}")

    try:
        # Ensure directory exists
        episode_path.mkdir(parents=True, exist_ok=True)

        # Stream download to handle large files
        logger.info(f"Downloading from: {mp3_link}")
        response = requests.get(mp3_link, stream=True, timeout=600)
        response.raise_for_status()

        # Get file size for logging
        total_size = int(response.headers.get("content-length", 0))
        if total_size:
            logger.info(
                f"MP3 file size: {total_size / (1024 * 1024):.1f} MB"
            )

        # Write to file in chunks
        downloaded = 0
        chunk_size = 8192
        with open(mp3_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)

        logger.info(
            f"MP3 download completed for episode {episode_num} at \"{mp3_path}\": "
            f"{downloaded / (1024 * 1024):.1f} MB"
        )

        episode_data["mp3_path"] = str(mp3_path)
        return episode_data

    except requests.RequestException as e:
        logger.error(
            f"Error downloading MP3 for episode {episode_num}: {e}",
            exc_info=True,
        )
        # Clean up partial download
        if mp3_path.exists():
            mp3_path.unlink()
        raise self.retry(exc=e)


@app.task(
    name="src.app.tasks.transcribe_episode",
    bind=True,
    max_retries=2,
    default_retry_delay=600,
)
def transcribe_episode(
    self, episode_data: dict[str, Any]
) -> dict[str, Any]:
    """
    Transcribe episode audio via Deepgram.

    Uses Deepgram's nova-2 model with Russian language support,
    paragraph detection, and speaker diarization.
    """
    from deepgram import DeepgramClient, PrerecordedOptions, FileSource
    import httpx

    episode_num = episode_data["episode_number"]
    episode_path = Path(
        episode_data.get("episode_path", STORAGE_DIR / str(episode_num))
    )
    mp3_path = episode_path / "episode.mp3"
    transcription_path = episode_path / "transcription-deepgram.json"

    logger.info(f"Transcribing episode {episode_num}")

    # Skip if any transcription already exists (first check before lock)
    existing_transcription = next(episode_path.glob("transcription-*.json"), None)
    if existing_transcription:
        logger.info(
            f"Transcription already exists for episode {episode_num} "
            f"at {existing_transcription}, skipping"
        )
        episode_data["transcription_path"] = str(existing_transcription)
        return episode_data

    # Acquire distributed lock to prevent duplicate Deepgram calls
    # Timeout is 1 hour since transcription can take a long time
    with redis_lock(f"transcribe_episode_{episode_num}", timeout=3600) as acquired:
        if not acquired:
            logger.info(
                f"Transcription already in progress for episode {episode_num}, "
                "skipping duplicate task"
            )
            return episode_data

        # Re-check after acquiring lock (double-checked locking)
        existing_transcription = next(
            episode_path.glob("transcription-*.json"), None
        )
        if existing_transcription:
            logger.info(
                f"Transcription appeared while waiting for lock "
                f"for episode {episode_num}: {existing_transcription}"
            )
            episode_data["transcription_path"] = str(existing_transcription)
            return episode_data

        # Check MP3 exists
        if not mp3_path.exists():
            raise ValueError(
                f"MP3 file not found for episode {episode_num}: {mp3_path}"
            )

        # Check API key
        api_key = settings.deepgram_api_key
        if not api_key:
            raise ValueError("DEEPGRAM_API_KEY not configured")

        try:
            # Read MP3 file
            logger.info(f"Reading MP3 file: {mp3_path}")
            with open(mp3_path, "rb") as f:
                buffer_data = f.read()
            logger.info(
                f"MP3 file size: {len(buffer_data) / (1024 * 1024):.1f} MB"
            )

            # Configure Deepgram
            client = DeepgramClient(api_key)
            options = PrerecordedOptions(
                model="nova-3",
                language="ru",
                paragraphs=True,
                diarize=True,
            )

            payload: FileSource = {
                "buffer": buffer_data,
            }

            # Send to Deepgram
            logger.info(f"Sending episode {episode_num} to Deepgram API")
            response = client.listen.prerecorded.v("1").transcribe_file(
                source=payload,
                options=options,
                timeout=httpx.Timeout(600.0, connect=30.0),
            )

            # Save transcription
            logger.info(f"Saving transcription to: {transcription_path}")
            transcription_path.write_text(
                response.to_json(ensure_ascii=False),
                encoding="utf-8",
            )

            logger.info(
                f"Transcription completed for episode {episode_num} "
                f"at {transcription_path}"
            )
            episode_data["transcription_path"] = str(transcription_path)
            return episode_data

        except Exception as e:
            logger.error(
                f"Error transcribing episode {episode_num}: {e}",
                exc_info=True,
            )
            raise self.retry(exc=e)


@app.task(
    name="src.app.tasks.segment_episode",
    bind=True,
    max_retries=2,
    default_retry_delay=300,
)
def segment_episode(self, episode_data: dict[str, Any]) -> dict[str, Any]:
    """
    Segment transcription using LLM and store in pgvector.

    Uses LLM-based segmentation to split the transcription into
    topic-based segments, generates embeddings, and stores everything
    in PostgreSQL with pgvector for similarity search.

    This is the final task in the processing chain - it removes the
    episode from the Redis processing set when done.
    """
    from src.components.segmentation.segmentation_builder import (
        SegmentationBuilder,
    )
    from src.components.segmentation.pgvector_repository import DB

    episode_num = episode_data["episode_number"]
    episode_path = Path(
        episode_data.get("episode_path", STORAGE_DIR / str(episode_num))
    )

    logger.info(f"Segmenting episode {episode_num}")

    # Check transcription exists
    transcription_path = episode_data.get("transcription_path")
    if not transcription_path:
        existing = next(episode_path.glob("transcription-*.json"), None)
        if not existing:
            raise FileNotFoundError(
                f"Transcription file not found: {episode_path}"
            )
        transcription_path = str(existing)

    # Check LLM API configuration
    api_key = settings.segmentation_llm_api_key
    base_url = settings.segmentation_llm_api_url
    if not api_key or not base_url:
        raise ValueError(
            "SEGMENTATION_LLM_API_KEY and SEGMENTATION_LLM_API_URL "
            "must be configured"
        )

    # Acquire distributed lock to prevent duplicate LLM/embedding calls
    # Timeout is 30 minutes for segmentation
    with redis_lock(f"segment_episode_{episode_num}", timeout=1800) as acquired:
        if not acquired:
            logger.info(
                f"Segmentation already in progress for episode {episode_num}, "
                "skipping duplicate task"
            )
            return {
                "status": "skipped",
                "episode_number": episode_num,
                "reason": "lock_not_acquired",
            }

        try:
            # Connect to database
            db = DB(
                host=settings.db_host,
                port=settings.db_port,
                dbname=settings.db_name,
                user=settings.db_user,
                password=settings.db_password,
            )

            # Skip if already in database (also serves as double-check after lock)
            existing_episode = db.find_episode(episode_num)
            if existing_episode:
                logger.info(
                    f"Episode {episode_num} already in database "
                    f"(id={existing_episode.id}), skipping"
                )
                # Clean up processing marker
                unmark_episode_processing(episode_num)
                return {
                    "status": "skipped",
                    "episode_number": episode_num,
                    "episode_path": str(episode_path),
                    "episode_id": str(existing_episode.id),
                    "completed_at": datetime.now().isoformat(),
                }

            # Build segmentation
            logger.info(f"Building segments for episode {episode_num}")
            builder = SegmentationBuilder(
                storage_dir=STORAGE_DIR,
                api_key=api_key,
                base_url=base_url,
            )
            segmentation_result = builder.get_segments(episode_path)

            logger.info(
                f"Created {len(segmentation_result.segments)} segments "
                f"for episode {episode_num}"
            )

            # Store in database
            logger.info(
                f"Storing segments in database for episode {episode_num}"
            )
            episode_id = db.insert(segmentation_result)

            if episode_id:
                logger.info(
                    f"Stored episode {episode_num} with id {episode_id}"
                )
            else:
                logger.warning(
                    f"Failed to store episode {episode_num} in database"
                )

            # Clean up processing marker on success
            unmark_episode_processing(episode_num)

            return {
                "status": "completed",
                "episode_number": episode_num,
                "episode_path": str(episode_path),
                "episode_id": str(episode_id) if episode_id else None,
                "segments_count": len(segmentation_result.segments),
                "completed_at": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(
                f"Error segmenting episode {episode_num}: {e}",
                exc_info=True,
            )
            # Only unmark if we've exhausted retries
            if self.request.retries >= self.max_retries:
                logger.warning(
                    f"Max retries reached for episode {episode_num}, "
                    "removing from processing set"
                )
                unmark_episode_processing(episode_num)
            raise self.retry(exc=e)
