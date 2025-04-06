from dataclasses import dataclass
import os
import asyncio
import logging
from aiogram import Bot, Dispatcher, types
from aiogram.types import InlineQuery, InlineQueryResultArticle, InputTextMessageContent
import requests
import hashlib

API_TOKEN = os.getenv("TG_BOT_API_TOKEN")
SEARCH_API_URL = "http://localhost:8080/v2/search"

logging.basicConfig(level=logging.INFO)

bot = Bot(token=API_TOKEN)
dp = Dispatcher()


@dataclass
class Timestamp:
    hour: int
    minute: int
    second: int

    def __str__(self):
        return f"{self.hour:0<2}:{self.minute:0<2}:{self.second:0<2}"


async def execute_search(
    query: str,
    limit: int = 10,
    offset: int = 0,
) -> list[dict]:
    querystring = {
        "query": query,
        "limit": limit,
        "offset": offset,
        "include_episode": True
    }

    response = requests.get(SEARCH_API_URL, params=querystring)

    return response.json()


def get_description(search_result: dict, query: str) -> str:
    segment = search_result["segment"]
    idx = segment.lower().find(query.lower())
    if idx == -1:
        return segment[:1000]

    start = max(0, idx - 200)
    while start > 0 and not segment[start].isalnum():
        start -= 1
    start = max(0, start - 1)

    end = min(len(segment), idx + 1000)
    while end < len(segment) and not segment[end].isalnum():
        end += 1

    subsegment = segment[start:end]
    query_pos = idx - start
    formatted = subsegment[:query_pos] + f"<b>{subsegment[query_pos:query_pos+len(query)]}</b>" + subsegment[query_pos + len(query):]

    if start > 0:
        formatted = f"[...] {formatted}"
    if end < len(segment):
        formatted = f"{formatted} [...]"

    return formatted


def get_sentence_borders(search_result: dict) -> tuple[Timestamp, Timestamp]:
    def _sec_to_timestamp(sec: int) -> Timestamp:
        hours = int(sec // 3600)
        remaining = int(sec % 3600)
        minutes = int(remaining // 60)
        seconds = int(remaining % 60)
        
        return Timestamp(hour=hours, minute=minutes, second=seconds)

    starts_at = _sec_to_timestamp(search_result["starts_at"])
    ends_at = _sec_to_timestamp(search_result["ends_at"])

    return starts_at, ends_at


def get_message_text(
    episode: dict,
    sentence: str,
    description: str,
    sentense_borders: tuple[Timestamp, Timestamp]
) -> str:
    episode_link = f'<a href=\"{episode["link"]}\">{episode["title"]}</a>'
    sentence = f'[{sentense_borders[0]}] <i>{sentence}</i>'
    return f"""{episode_link}\n\n{sentence}\n\n{description}"""


@dp.message()
async def search(message: types.Message):
    results = await execute_search(
        message.text,
        limit=10,
        offset=0
    )
    if not results:
        await message.answer('No results found')
        return
    
    for item in results:
        if len(item["sentence"]) > 10:
            description = get_description(item, message.text)
            start, end = get_sentence_borders(item)
            message_text = get_message_text(
                item["episode"],
                item["sentence"],
                f"<blockquote expandable>{description}</blockquote>",
                (start, end)
            )
            await message.answer(message_text, parse_mode="HTML")


@dp.inline_query()
async def inline_search(inline_query: InlineQuery):
    query = inline_query.query.strip()
    if not query:
        return

    results = await execute_search(query)
    articles = []
    for idx, item in enumerate(results):
        description = get_description(item, query)
        start, end = get_sentence_borders(item)
        message_text = get_message_text(
            item["episode"],
            item["sentence"],
            f"<blockquote expandable>{description}</blockquote>",
            (start, end)
        )
        articles.append(
            InlineQueryResultArticle(
                id=hashlib.md5(f"{query}-{idx}".encode()).hexdigest(),
                title=item["sentence"],
                description=description,
                input_message_content=InputTextMessageContent(
                    message_text=message_text,
                    parse_mode="HTML"
                ),
                parse_mode="HTML"
            )
        )

    await inline_query.answer(articles, cache_time=15)


if __name__ == "__main__":
    asyncio.run(dp.start_polling(bot))
