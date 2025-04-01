import os
import asyncio
import logging
from aiogram import Bot, Dispatcher, types
from aiogram.types import InlineQuery, InlineQueryResultArticle, InputTextMessageContent
import requests
import hashlib

API_TOKEN = os.getenv('TG_BOT_API_TOKEN')
SEARCH_API_URL = 'http://localhost:8080/v2/search'

logging.basicConfig(level=logging.INFO)

bot = Bot(token=API_TOKEN)
dp = Dispatcher()


async def execute_search(query: str) -> list[dict]:
    # payload = {
    #     "query": query,
    #     "limit": 10,
    #     "offset": 0
    # }
    # headers = {"content-type": "application/json"}
    # response = requests.post(SEARCH_API_URL, json=payload, headers=headers)

    # if response.status_code != 200:
    #     logging.error(f"API call failed: {response.status_code}")
    #     return []

    # return response.json()

    url = "http://localhost:8080/v2/search"

    querystring = {
        "query": query,
        "limit":100,
        "offset":0
    }

    response = requests.get(url, params=querystring)

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
    query = inline_query.query.strip()
    if not query:
        return

    # Call the /v2/search API
    params = {
        'query': query,
        'limit': 10,  # Default limit
        'offset': 0   # Default offset
    }
    # response = requests.get(SEARCH_API_URL, params=params)
    response = requests.post(SEARCH_API_URL, data=params)
    if response.status_code != 200:
        logging.error(f"API call failed: {response.status_code}")
        return

    results = response.json()
    articles = []
    for idx, item in enumerate(results):
        articles.append(
            InlineQueryResultArticle(
                id=hashlib.md5(f"{query}-{idx}".encode()).hexdigest(),
                title=item.get('episode', 'No Episode'),
                input_message_content=InputTextMessageContent(item.get('sentence', 'No sentence'))
            )
        )

    await inline_query.answer(articles, cache_time=1)

if __name__ == '__main__':
    asyncio.run(dp.start_polling(bot))
