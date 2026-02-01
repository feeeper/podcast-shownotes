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
