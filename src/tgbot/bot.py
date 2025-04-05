import os
import asyncio
import logging
from aiogram import Bot, Dispatcher, types
from aiogram.types import InlineQuery, InlineQueryResultArticle, InputTextMessageContent
from aiogram.utils.formatting import ExpandableBlockQuote
import requests
import hashlib

API_TOKEN = os.getenv('TG_BOT_API_TOKEN')
SEARCH_API_URL = 'http://localhost:8080/v2/search'

logging.basicConfig(level=logging.INFO)

bot = Bot(token=API_TOKEN)
dp = Dispatcher()


async def execute_search(query: str) -> list[dict]:
    querystring = {
        "query": query,
        "limit":100,
        "offset":0,
        "include_episode": True
    }

    response = requests.get(SEARCH_API_URL, params=querystring)

    return response.json()


@dp.message()
async def search(message: types.Message):
    results = await execute_search(message.text)
    if not results:
        await message.answer('No results found')
        return
    
    for item in results:
        await message.answer(f"{item.get('episode', 'No Episode')}: {item.get('sentence', 'No sentence')}")


@dp.inline_query()
async def inline_search(inline_query: InlineQuery):
    def _get_description(item: dict) -> str:
        segment = item["segment"]
        idx = segment.lower().find(query.lower())
        if idx == -1:
            return segment[:1000]

        start = max(0, idx - 1000)
        while start > 0 and not segment[start].isalnum():
            start -= 1
        start = max(0, start - 1)

        end = min(len(segment), idx + 1000)
        while end < len(segment) and not segment[end].isalnum():
            end += 1

        subsegment = segment[start:end]
        query_pos = idx - start
        formatted = subsegment[:query_pos] + f"<b>{subsegment[query_pos:query_pos+len(query)]}</b>" + subsegment[query_pos + len(query):]
        return f"[...] {formatted} [...]"
    
    def _get_message_text(
            episode: dict,
            sentence: str,
            description: str) -> str:
        episode_link = f'<a href=\"{episode["link"]}\">{episode["title"]}</a>'
        sentence = f'<i>{sentence}</i>'
        return f"""{episode_link}\n\n{sentence}\n\n{description}"""

    query = inline_query.query.strip()
    if not query:
        return

    results = await execute_search(query)
    articles = []
    for idx, item in enumerate(results):
        description = _get_description(item)
        details = ExpandableBlockQuote(description)
        message_text = _get_message_text(item["episode"], item["sentence"], f"<blockquote expandable>{description}</blockquote>")
        articles.append(
            InlineQueryResultArticle(
                id=hashlib.md5(f"{query}-{idx}".encode()).hexdigest(),
                title=item["sentence"],
                description=description,
                input_message_content=InputTextMessageContent(
                    message_text=message_text,
                    parse_mode='HTML'
                ),
                parse_mode="HTML"
            )
        )

    await inline_query.answer(articles, cache_time=15)


if __name__ == '__main__':
    asyncio.run(dp.start_polling(bot))
