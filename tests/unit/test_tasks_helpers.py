import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parent.parent.parent)
)

from src.app.tasks import (  # noqa: E402
    _try_extract_episode_number,
    _safe_dirname,
)


class TestTryExtractEpisodeNumber:
    def test_devzen_link(self):
        link = 'https://devzen.ru/episode-472/'
        assert _try_extract_episode_number(link) == 472

    def test_devzen_link_no_trailing_slash(self):
        link = 'https://devzen.ru/episode-123'
        assert _try_extract_episode_number(link) == 123

    def test_no_episode_number(self):
        link = 'https://example.com/some-podcast/feed'
        assert _try_extract_episode_number(link) is None

    def test_empty_link(self):
        assert _try_extract_episode_number('') is None


class TestSafeDirname:
    def test_no_published_produces_hex_string(self):
        result = _safe_dirname(
            'https://example.com/episode/123'
        )
        assert len(result) == 16
        assert all(
            c in '0123456789abcdef' for c in result
        )

    def test_with_published_has_date_prefix(self):
        result = _safe_dirname(
            'some-guid',
            published='2024-08-12T08:58:42',
        )
        assert result.startswith('2024-08-12_')
        assert len(result) == 11 + 16  # date_ + hash

    def test_date_prefix_sorts_chronologically(self):
        older = _safe_dirname(
            'guid-a', published='2024-01-01T00:00:00'
        )
        newer = _safe_dirname(
            'guid-b', published='2024-06-15T00:00:00'
        )
        assert older < newer

    def test_deterministic(self):
        guid = 'some-guid-value'
        pub = '2024-03-01T12:00:00'
        assert (
            _safe_dirname(guid, pub)
            == _safe_dirname(guid, pub)
        )

    def test_different_guids_differ(self):
        pub = '2024-03-01T12:00:00'
        assert (
            _safe_dirname('guid-a', pub)
            != _safe_dirname('guid-b', pub)
        )
