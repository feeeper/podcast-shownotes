from pathlib import Path

import yaml
from pydantic import BaseModel


class PodcastConfig(BaseModel):
    name: str
    slug: str
    rss_url: str
    language: str = 'en'
    transcription_model: str = 'nova-3'
    embedding_model: str = 'deepvk/USER-bge-m3'


class PodcastsConfig(BaseModel):
    podcasts: list[PodcastConfig]


def load_podcasts_config(
    path: Path = Path('podcasts.yml'),
) -> PodcastsConfig:
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    return PodcastsConfig(**data)


def get_podcast_config(
    slug: str,
    path: Path = Path('podcasts.yml'),
) -> PodcastConfig:
    configs = load_podcasts_config(path)
    for podcast in configs.podcasts:
        if podcast.slug == slug:
            return podcast
    raise ValueError(f'Podcast with slug "{slug}" not found')


def add_podcast_config(
    podcast: PodcastConfig,
    path: Path = Path('podcasts.yml'),
) -> None:
    """Add a new podcast to the config file."""
    try:
        configs = load_podcasts_config(path)
    except FileNotFoundError:
        configs = PodcastsConfig(podcasts=[])

    # Check if already exists
    for existing in configs.podcasts:
        if existing.slug == podcast.slug:
            return

    configs.podcasts.append(podcast)

    with open(path, 'w') as f:
        yaml.dump(
            {'podcasts': [p.model_dump() for p in configs.podcasts]},
            f,
            default_flow_style=False,
            allow_unicode=True,
        )


def remove_podcast_config(
    slug: str,
    path: Path = Path('podcasts.yml'),
) -> bool:
    """Remove a podcast from the config file by slug.

    Returns True if removed, False if not found.
    """
    try:
        configs = load_podcasts_config(path)
    except FileNotFoundError:
        return False

    original_count = len(configs.podcasts)
    configs.podcasts = [p for p in configs.podcasts if p.slug != slug]

    if len(configs.podcasts) == original_count:
        return False

    with open(path, 'w') as f:
        yaml.dump(
            {'podcasts': [p.model_dump() for p in configs.podcasts]},
            f,
            default_flow_style=False,
            allow_unicode=True,
        )
    return True
