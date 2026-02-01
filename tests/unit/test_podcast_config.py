import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parent.parent.parent)
)

import pytest
from src.shared.podcast_config import (
    PodcastConfig,
    load_podcasts_config,
    get_podcast_config,
)


def test_podcast_config_defaults():
    config = PodcastConfig(
        name='Test',
        slug='test',
        rss_url='https://example.com/feed',
    )
    assert config.language == 'en'
    assert config.transcription_model == 'nova-3'
    assert config.embedding_model == 'deepvk/USER-bge-m3'


def test_podcast_config_custom():
    config = PodcastConfig(
        name='Test',
        slug='test',
        rss_url='https://example.com/feed',
        language='ru',
        transcription_model='nova-2',
        embedding_model='custom-model',
    )
    assert config.language == 'ru'
    assert config.transcription_model == 'nova-2'
    assert config.embedding_model == 'custom-model'


def test_load_podcasts_config(tmp_path):
    config_file = tmp_path / 'podcasts.yml'
    config_file.write_text(
        'podcasts:\n'
        '  - name: Test Podcast\n'
        '    slug: test\n'
        '    rss_url: https://example.com/feed\n'
        '    language: ru\n'
        '    transcription_model: nova-3\n'
        '    embedding_model: deepvk/USER-bge-m3\n'
    )

    configs = load_podcasts_config(config_file)
    assert len(configs.podcasts) == 1
    assert configs.podcasts[0].slug == 'test'
    assert configs.podcasts[0].language == 'ru'


def test_load_podcasts_config_multiple(tmp_path):
    config_file = tmp_path / 'podcasts.yml'
    config_file.write_text(
        'podcasts:\n'
        '  - name: Podcast A\n'
        '    slug: podcast-a\n'
        '    rss_url: https://a.example.com/feed\n'
        '    language: en\n'
        '  - name: Podcast B\n'
        '    slug: podcast-b\n'
        '    rss_url: https://b.example.com/feed\n'
        '    language: de\n'
    )

    configs = load_podcasts_config(config_file)
    assert len(configs.podcasts) == 2
    assert configs.podcasts[0].slug == 'podcast-a'
    assert configs.podcasts[1].slug == 'podcast-b'
    assert configs.podcasts[1].language == 'de'


def test_get_podcast_config(tmp_path):
    config_file = tmp_path / 'podcasts.yml'
    config_file.write_text(
        'podcasts:\n'
        '  - name: Podcast A\n'
        '    slug: alpha\n'
        '    rss_url: https://a.example.com/feed\n'
        '  - name: Podcast B\n'
        '    slug: beta\n'
        '    rss_url: https://b.example.com/feed\n'
    )

    config = get_podcast_config('beta', config_file)
    assert config.name == 'Podcast B'
    assert config.slug == 'beta'


def test_get_podcast_config_not_found(tmp_path):
    config_file = tmp_path / 'podcasts.yml'
    config_file.write_text(
        'podcasts:\n'
        '  - name: Podcast A\n'
        '    slug: alpha\n'
        '    rss_url: https://a.example.com/feed\n'
    )

    with pytest.raises(ValueError, match='not found'):
        get_podcast_config('nonexistent', config_file)
