import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import datetime
from uuid import uuid4
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient

from src.components.segmentation.models import (
    EpisodeDto,
    SearchResultDto,
    SearchResults,
)


@pytest.fixture()
def client():
    # Patch DB so it never connects to a real database
    with patch('src.api.app.get_db') as mock_get_db:
        mock_db = MagicMock()
        mock_get_db.return_value = mock_db

        # Import after patching to avoid startup DB init
        from src.api.app import app

        app.dependency_overrides.clear()
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c, mock_db


def _make_search_results(n: int = 2) -> SearchResults:
    return SearchResults(
        results=[
            SearchResultDto(
                episode=400 + i,
                podcast_slug='devzen',
                sentence=f'sentence {i}',
                segment=f'segment text {i}',
                distance=0.1 * i,
                starts_at=100.0 * i,
                ends_at=100.0 * i + 60.0,
            )
            for i in range(n)
        ]
    )


def _make_episode(num: int = 400) -> EpisodeDto:
    return EpisodeDto(
        id=uuid4(),
        podcast_slug='devzen',
        num=num,
        title=f'Episode {num}',
        shownotes='some shownotes',
        hosts=['host1'],
        release_date=datetime.datetime(2024, 1, 1),
        link=f'https://devzen.ru/episode-{num}/',
    )


class TestPing:
    def test_ping(self, client):
        c, _ = client
        resp = c.get('/ping')
        assert resp.status_code == 200
        body = resp.json()
        assert body['pong'] is True
        assert 'utcnow' in body


class TestPostSearch:
    def test_search_returns_results(self, client):
        c, mock_db = client
        podcast_id = uuid4()
        mock_db.get_podcast_id.return_value = podcast_id
        mock_db.find_similar.return_value = (
            _make_search_results(2)
        )

        resp = c.post(
            '/search',
            json={
                'query': 'test',
                'podcast_slug': 'devzen',
                'limit': 5,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2
        assert data[0]['sentence'] == 'sentence 0'
        assert data[1]['episode'] == 401
        mock_db.get_podcast_id.assert_called_once_with(
            'devzen'
        )
        mock_db.find_similar.assert_called_once_with(
            query='test',
            limit=5,
            offset=0,
            podcast_id=podcast_id,
        )

    def test_search_with_unknown_podcast_slug(self, client):
        c, mock_db = client
        mock_db.get_podcast_id.return_value = None

        resp = c.post(
            '/search',
            json={
                'query': 'test',
                'podcast_slug': 'unknown',
            },
        )
        assert resp.status_code == 404
        assert resp.json()['detail'] == 'Podcast not found'
        mock_db.find_similar.assert_not_called()


class TestV2Search:
    def test_search_without_include_episode(self, client):
        c, mock_db = client
        podcast_id = uuid4()
        mock_db.get_podcast_id.return_value = podcast_id
        mock_db.find_similar_complex.return_value = (
            _make_search_results(2)
        )

        resp = c.get(
            '/v2/search',
            params={
                'query': 'kubernetes',
                'podcast_slug': 'devzen',
                'limit': 5,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2
        assert data[0]['episode'] == 400
        mock_db.find_episode.assert_not_called()

    def test_search_with_include_episode(self, client):
        c, mock_db = client
        podcast_id = uuid4()
        mock_db.get_podcast_id.return_value = podcast_id
        mock_db.find_similar_complex.return_value = (
            _make_search_results(1)
        )
        episode = _make_episode(400)
        mock_db.find_episode.return_value = episode

        resp = c.get(
            '/v2/search',
            params={
                'query': 'kubernetes',
                'podcast_slug': 'devzen',
                'include_episode': True,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        item = data[0]
        assert isinstance(item['episode'], dict)
        assert item['episode']['title'] == 'Episode 400'
        assert item['episode']['num'] == 400
        assert item['sentence'] == 'sentence 0'

    def test_search_with_missing_episode(self, client):
        c, mock_db = client
        podcast_id = uuid4()
        mock_db.get_podcast_id.return_value = podcast_id
        mock_db.find_similar_complex.return_value = (
            _make_search_results(1)
        )
        mock_db.find_episode.return_value = None

        resp = c.get(
            '/v2/search',
            params={
                'query': 'test',
                'podcast_slug': 'devzen',
                'include_episode': True,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        # Falls back to plain SearchResultDto format
        assert data[0]['episode'] == 400

    def test_search_verifies_db_calls(self, client):
        c, mock_db = client
        podcast_id = uuid4()
        mock_db.get_podcast_id.return_value = podcast_id
        mock_db.find_similar_complex.return_value = (
            _make_search_results(1)
        )

        resp = c.get(
            '/v2/search',
            params={
                'query': 'test',
                'podcast_slug': 'devzen',
            },
        )
        assert resp.status_code == 200
        mock_db.get_podcast_id.assert_called_once_with(
            'devzen'
        )
        mock_db.find_similar_complex.assert_called_once_with(
            query='test',
            limit=10,
            offset=0,
            language='ru',
            podcast_id=podcast_id,
        )

    def test_search_with_unknown_podcast_slug(self, client):
        c, mock_db = client
        mock_db.get_podcast_id.return_value = None

        resp = c.get(
            '/v2/search',
            params={
                'query': 'test',
                'podcast_slug': 'unknown',
            },
        )
        assert resp.status_code == 404
        assert resp.json()['detail'] == 'Podcast not found'
        mock_db.find_similar_complex.assert_not_called()
