from __future__ import annotations

import json
from dataclasses import dataclass
import dataclasses
from pathlib import Path
import feedparser
from datetime import datetime


@dataclass
class EpisodeMetadata:
    title: str
    episode_link: str
    mp3_link: str
    episode: int
    published: datetime
    summary: str
    authors: list[str]
    html_content: str
    path: Path


class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o):
            result = dataclasses.asdict(o)
            for k, v in result.items():
                if isinstance(v, Path):
                    result[k] = str(v)
                elif isinstance(v, datetime):
                    result[k] = v.isoformat()
            return result
        return super().default(o)


class IndexBuilder:
    def __init__(self, storage_dir: Path) -> None:
        self._url = 'https://devzen.ru/feed/'
        self._parser = feedparser
        self._storage_dir = storage_dir

    def pick_episodes(self) -> list[EpisodeMetadata]:
        feed = self._parser.parse(self._url)
        episodes = []
        for entry in feed.entries:
            try:
                # entry with `themes` in the link is not an episode
                non_episode_link_prefixes = (
                    'https://devzen.ru/themes',
                    'https://devzen.ru/no-themes'
                )
                if entry.link.startswith(non_episode_link_prefixes):
                    continue

                episode = EpisodeMetadata(
                    title=entry.title,
                    episode_link=entry.link,
                    mp3_link=entry.enclosures[0].href,
                    episode=int(entry.title.split(' ')[-1]),
                    published=datetime(*entry.published_parsed[:6]),
                    summary=entry.summary,
                    authors=[author.name for author in entry.authors],
                    html_content=entry.content[0].value,
                    path=self._storage_dir / f'{int(entry.title.split(" ")[-1])}'  # noqa E501
                )

                # already downloaded
                if episode.path.exists():
                    continue
                else:
                    episode.path.mkdir(parents=True)

                with open(episode.path / 'episode.html', 'w') as file:
                    file.write(episode.html_content)

                with open(episode.path / 'episode.json', 'w') as file:
                    json.dump(
                        episode,
                        file,
                        indent=4,
                        sort_keys=True,
                        cls=EnhancedJSONEncoder,
                        ensure_ascii=False)

                episodes.append(episode)
            except Exception as e:
                print(f'Error: {entry}')
                print(f'Error: {e}')
                raise e

        return episodes


__all__ = [
    'IndexBuilder',
    'EpisodeMetadata'
]
