import asyncio
import aiohttp
import os
import json
from pathlib import Path


async def download_file(session, data, output_dir='data'):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    url = data[0]
    episode = data[1]

    filename = f'/mnt/d/Projects/podcast-shownotes/episodes/___/episode-{episode}.mp3'
    async with session.get(url) as response:
        if response.status == 200:
            with open(filename, 'wb') as f:
                while True:
                    chunk = await response.content.read(1024)
                    if not chunk:
                        break
                    f.write(chunk)
            print(f"Downloaded {url} to {filename}")
        else:
            print(f"Failed to download {url}")


async def main():
    urls = []
    for p in sorted(list(Path('components/indexer/data/').iterdir())):
        if int(p.name) <= 417:
            continue
        j = json.load(open(f'components/indexer/data/{p.name}/episode.json', 'r', encoding='utf-8'))
        urls.append((j['mp3_link'], j['episode']))

    async with aiohttp.ClientSession() as session:
        tasks = [download_file(session, url) for url in urls]
        await asyncio.gather(*tasks)


if __name__ == '__main__':
    asyncio.run(main())
