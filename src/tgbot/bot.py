from collections import defaultdict
from dataclasses import dataclass
import os
import asyncio
import logging
from aiogram import Bot, Dispatcher, types
from aiogram.types import (
    InlineQuery,
    InlineQueryResultArticle,
    InputTextMessageContent,
)
from aiogram.filters import CommandStart
import requests
import hashlib

API_TOKEN = os.getenv('TG_BOT_API_TOKEN')
SEARCH_API_URL = 'http://localhost:8080/v2/search'


logging.basicConfig(level=logging.INFO)

bot = Bot(token=API_TOKEN)
dp = Dispatcher()


_search_tasks: dict[str, asyncio.Task] = {}
DEBOUNCE_TIME = 2  # seconds
MIN_QUERY_LENGTH = 3
NO_RESULTS = 'There are no results for your query'
TOO_SHORT_REQUEST = (
    'Your query is too short. '
    f'Minimum length is {MIN_QUERY_LENGTH + 1}'
)


@dataclass
class Timestamp:
    hour: int
    minute: int
    second: int

    def __str__(self):
        return (
            f'{self.hour:0>2}:'
            f'{self.minute:0>2}:'
            f'{self.second:0>2}'
        )


async def execute_search(
    query: str,
    limit: int = 10,
    offset: int = 0,
) -> list[dict]:
    querystring = {
        'query': query,
        'limit': limit,
        'offset': offset,
        'include_episode': True,
    }

    response = requests.get(
        SEARCH_API_URL, params=querystring
    )

    return response.json()


def get_description(
    search_result: dict, query: str
) -> str:
    segment = search_result['segment']
    idx = segment.lower().find(query.lower())
    if idx == -1:
        return segment[:1000]

    start = max(0, idx - 200)
    while start > 0 and not segment[start].isalnum():
        start -= 1
    start = max(0, start - 1)

    end = min(len(segment), idx + 1000)
    while (
        end < len(segment)
        and not segment[end].isalnum()
    ):
        end += 1

    subsegment = segment[start:end]
    query_pos = idx - start
    formatted = (
        subsegment[:query_pos]
        + f'<b>{subsegment[query_pos:query_pos + len(query)]}</b>'
        + subsegment[query_pos + len(query):]
    )

    if start > 0:
        formatted = f'[...] {formatted}'
    if end < len(segment):
        formatted = f'{formatted} [...]'

    return formatted


def get_sentence_borders(
    search_result: dict,
) -> tuple[Timestamp, Timestamp]:
    def _sec_to_timestamp(sec: int) -> Timestamp:
        hours = int(sec // 3600)
        remaining = int(sec % 3600)
        minutes = int(remaining // 60)
        seconds = int(remaining % 60)

        return Timestamp(
            hour=hours,
            minute=minutes,
            second=seconds,
        )

    starts_at = _sec_to_timestamp(
        search_result['starts_at']
    )
    ends_at = _sec_to_timestamp(
        search_result['ends_at']
    )

    return starts_at, ends_at


def _get_episode_link(episode: dict) -> str:
    """Get episode link from DB-stored field."""
    return episode.get('link', '')


def get_message_text(
    episode: dict,
    sentence: str,
    description: str,
    sentence_borders: tuple[Timestamp, Timestamp],
) -> str:
    link = _get_episode_link(episode)
    title = episode.get('title', '')
    episode_link = f'<a href="{link}">{title}</a>'
    sentence = (
        f'[{sentence_borders[0]}] <i>{sentence}</i>'
    )
    return (
        f'{episode_link}\n\n'
        f'{sentence}\n\n'
        f'{description}'
    )


def get_single_search_result_text(
    sentence: str,
    description: str,
    sentence_borders: tuple[Timestamp, Timestamp],
) -> str:
    sentence = (
        f'[{sentence_borders[0]}] <i>{sentence}</i>'
    )
    expandable_description = (
        f'<blockquote expandable>'
        f'{description}'
        f'</blockquote>'
    )
    return (
        f'{sentence}\n\n{expandable_description}'
    )


@dp.message(CommandStart())
async def start_handler(
    message: types.Message,
) -> None:
    await message.answer(
        'Welcome to Podcast Search Bot!\n\n'
        'Search through podcast transcripts.\n\n'
        'How to use:\n'
        '- Type any word or phrase to search\n'
        '- Use inline mode in any chat\n'
        '- Results include timestamps and context\n\n'
        'Try searching for any topic you\'re '
        'interested in!',
        parse_mode='HTML',
    )


@dp.message()
async def search(message: types.Message):
    def _get_grouped_by_episode_message(
        episode: dict,
        episode_results: list[str],
    ) -> str:
        link = _get_episode_link(episode)
        title = episode.get('title', '')
        episode_link = (
            f'<a href="{link}">{title}</a>'
        )
        parts = [episode_link] + episode_results
        return '\n\n'.join(parts)

    if len(message.text) <= MIN_QUERY_LENGTH:
        await message.answer(TOO_SHORT_REQUEST)
        return

    results = await execute_search(
        message.text,
        limit=10,
        offset=0,
    )

    if not results:
        await message.answer(NO_RESULTS)
        return

    results_by_episode = defaultdict(list)
    for item in results:
        ep_key = item['episode'].get(
            'num'
        ) or item['episode'].get('id', '')
        results_by_episode[ep_key].append(item)

    at_least_one_result_was_sent = False
    for _, episode_items in results_by_episode.items():
        episode_results = []
        seen_segments = set()
        for item in sorted(
            episode_items,
            key=lambda x: x['starts_at'],
        ):
            segment = item['segment']
            if segment in seen_segments:
                continue

            sentence = item['sentence']
            if len(sentence) > 10:
                description = get_description(
                    item, message.text
                )
                start, end = get_sentence_borders(item)
                message_text = (
                    get_single_search_result_text(
                        sentence=sentence,
                        description=description,
                        sentence_borders=(start, end),
                    )
                )
                episode_results.append(message_text)
            seen_segments.add(segment)

        if len(episode_results) > 0:
            episode = episode_items[0]['episode']
            episode_message = (
                _get_grouped_by_episode_message(
                    episode,
                    episode_results=episode_results,
                )
            )
            await message.answer(
                episode_message, parse_mode='HTML'
            )
            await asyncio.sleep(0.5)
            at_least_one_result_was_sent = True

    if not at_least_one_result_was_sent:
        await message.answer(NO_RESULTS)


@dp.inline_query()
async def inline_search(inline_query: InlineQuery):
    query = inline_query.query.strip()
    if not query:
        return

    if len(query) <= MIN_QUERY_LENGTH:
        await inline_query.answer(
            [], cache_time=300
        )
        return

    # Cancel previous search task for this user
    user_id = inline_query.from_user.id
    if (
        user_id in _search_tasks
        and not _search_tasks[user_id].done()
    ):
        _search_tasks[user_id].cancel()

    async def debounced_search():
        try:
            await asyncio.sleep(DEBOUNCE_TIME)
            results = await execute_search(query)
            articles = []
            for idx, item in enumerate(results):
                description = get_description(
                    item, query
                )
                start, end = get_sentence_borders(item)
                message_text = get_message_text(
                    item['episode'],
                    item['sentence'],
                    '<blockquote expandable>'
                    f'{description}'
                    '</blockquote>',
                    (start, end),
                )
                articles.append(
                    InlineQueryResultArticle(
                        id=hashlib.md5(
                            f'{query}-{idx}'.encode()
                        ).hexdigest(),
                        title=item['sentence'],
                        description=description,
                        input_message_content=(
                            InputTextMessageContent(
                                message_text=(
                                    message_text
                                ),
                                parse_mode='HTML',
                            )
                        ),
                        parse_mode='HTML',
                    )
                )
            await inline_query.answer(
                articles, cache_time=15
            )
        except asyncio.CancelledError:
            pass
        finally:
            _search_tasks.pop(user_id, None)

    _search_tasks[user_id] = asyncio.create_task(
        debounced_search()
    )


if __name__ == '__main__':
    asyncio.run(dp.start_polling(bot))
