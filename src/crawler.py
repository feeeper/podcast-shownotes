import feedparser
import asyncio
import aiohttp


async def fetch(url: str) -> str:
    ...

def crawl(url) -> feedparser.util.FeedParserDict:
    return feedparser.parse(url)


def get_entries(feed: feedparser.util.FeedParserDict) -> list:
    return feed.entries


if __name__ == '__main__':
    feed = crawl('https://devzen.ru/feed/')
    entries = get_entries(feed)
    for entry in entries:
        print(entry)
        print(entry.title)
        print(entry.link)
        print(entry.description)
        print(entry.published)
        print(entry.summary_detail['value'])
        print()
        break
