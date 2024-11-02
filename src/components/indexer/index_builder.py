from __future__ import annotations

import json
from dataclasses import dataclass
import dataclasses
from pathlib import Path
import feedparser
from datetime import datetime
import bs4
import requests


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

    @classmethod
    def from_path(cls, path: Path) -> EpisodeMetadata | None:
        if not path.exists():
            return EpisodeMetadata(
                title='',
                episode_link='',
                mp3_link='',
                episode=-1,
                published=datetime.now(),
                summary='',
                authors=[],
                html_content='',
                path=path
            )

        with open(path / 'episode.json', 'r') as file:
            json_data = json.load(file)
            return EpisodeMetadata(
                title=json_data['title'],
                episode_link=json_data['episode_link'],
                mp3_link=json_data['mp3_link'],
                episode=json_data['episode'],
                published=datetime.fromisoformat(json_data['published']),
                summary=json_data['summary'],
                authors=json_data['authors'],
                html_content=json_data['html_content'],
                path=path
            )


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

                with open(episode.path / 'metadata.json', 'w') as md_file:
                    metadata = self.get_metadata(episode.episode_link)
                    json.dump(
                        metadata,
                        md_file,
                        indent=4,
                        sort_keys=True,
                        ensure_ascii=False
                    )

                episodes.append(episode)
            except Exception as e:
                print(f'Error: {entry}')
                print(f'Error: {e}')
                raise e

        return episodes

    @classmethod
    def get_metadata(cls, episode_link: str) -> dict:
        def get_speakers():
            paragraphs = parser.find('div', class_='entry-content clearfix').find_all('p')
            for a in list(filter(lambda x: 'голоса выпуска' in x.text.lower(), paragraphs))[0].find_all('a'):
                yield {
                    'name': a.text,
                    'href': a.attrs['href']
                }

        def get_music():
            paragraphs = parser.find('div', class_='entry-content clearfix').find_all('p')
            music_paragraph = next(filter(lambda x: 'фоновая музыка' in x.text.lower(), paragraphs), None)
            if music_paragraph is None:
                return {}
            else:
                music_anchor = music_paragraph.find('a')
                if music_anchor is None:
                    return {}
                else:
                    return {
                        'name': music_anchor.text,
                        'href': music_anchor.attrs['href']
                    }

        response = requests.get(episode_link)
        episode_html = response.text
        parser = bs4.BeautifulSoup(episode_html, 'html.parser')
        release_date = parser.find('time', class_='entry-date').text
        title = parser.find('h1', class_='entry-title').text
        shownotes = parser.find('div', class_='entry-content clearfix').text
        speakers = get_speakers()
        music = get_music()
        mp3 = parser.find('a', class_='powerpress_link_d').attrs['href']

        return {
            'release_date': release_date,
            'title': title,
            'shownotes': shownotes,
            'speakers': list(speakers),
            'music': music,
            'mp3': mp3
        }


__all__ = [
    'IndexBuilder',
    'EpisodeMetadata'
]
