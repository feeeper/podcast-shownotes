import os
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv

from celery import Celery
from celery.schedules import crontab


load_dotenv()

@dataclass
class CelerySettings:
    """Celery configuration loaded from environment variables."""

    broker_url: str = field(
        default_factory=lambda: os.getenv(
            "CELERY_BROKER_URL", "redis://localhost:6379/0"
        )
    )
    result_backend: str = field(
        default_factory=lambda: os.getenv(
            "CELERY_RESULT_BACKEND", "redis://localhost:6379/1"
        )
    )
    timezone: str = field(
        default_factory=lambda: os.getenv("CELERY_TIMEZONE", "UTC")
    )

    # Storage configuration
    episodes_storage_dir: str = field(
        default_factory=lambda: os.getenv("EPISODES_STORAGE_DIR", "episodes")
    )

    # Deepgram configuration
    deepgram_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("DEEPGRAM_API_KEY")
    )

    # LLM segmentation configuration
    segmentation_llm_api_url: Optional[str] = field(
        default_factory=lambda: os.getenv("SEGMENTATION_LLM_API_URL")
    )
    segmentation_llm_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("SEGMENTATION_LLM_API_KEY")
    )

    # Database configuration
    db_host: str = field(
        default_factory=lambda: os.getenv("DB_HOST", "localhost")
    )
    db_port: int = field(
        default_factory=lambda: int(os.getenv("DB_PORT", "5432"))
    )
    db_name: str = field(
        default_factory=lambda: os.getenv("DB_NAME", "podcast")
    )
    db_user: Optional[str] = field(
        default_factory=lambda: os.getenv("DB_USER")
    )
    db_password: Optional[str] = field(
        default_factory=lambda: os.getenv("DB_PASSWORD")
    )


settings = CelerySettings()


def create_celery_app(
    name: str = "podcast_shownotes",
    include: list[str] | None = None
) -> Celery:
    """
    Create and configure a Celery application instance.

    Args:
        name: Application name for Celery
        include: List of task modules to include

    Returns:
        Configured Celery application
    """
    if include is None:
        include = ["src.app.tasks"]

    celery_app = Celery(
        name,
        broker=settings.broker_url,
        backend=settings.result_backend,
        include=include
    )

    # General configuration
    celery_app.conf.update(
        timezone=settings.timezone,
        enable_utc=True,

        # Task execution settings
        task_acks_late=True,
        task_reject_on_worker_lost=True,

        # Serialization
        task_serializer="json",
        accept_content=["json"],
        result_serializer="json",

        # Result settings
        result_expires=3600,  # Results expire after 1 hour

        # Worker settings
        worker_prefetch_multiplier=1,
        worker_concurrency=4,
    )

    # Task routing for different queue priorities
    celery_app.conf.task_routes = {
        "src.app.tasks.check_rss_feed": {"queue": "scheduler"},
        "src.app.tasks.download_episode_metadata": {"queue": "downloads"},
        "src.app.tasks.download_episode_mp3": {"queue": "downloads"},
        "src.app.tasks.transcribe_episode": {"queue": "transcription"},
        "src.app.tasks.segment_episode": {"queue": "segmentation"},
    }

    # Beat schedule - RSS feed check every hour
    celery_app.conf.beat_schedule = {
        "check-rss-feed-hourly": {
            "task": "src.app.tasks.check_rss_feed",
            "schedule": crontab(minute=0),  # Every hour at :00
        },
    }

    return celery_app


# Default application instance
app = create_celery_app()
