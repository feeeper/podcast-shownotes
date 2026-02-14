from __future__ import annotations

import os
import re
import shutil
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from logging import getLogger
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from text_unidecode import unidecode

from src.components.segmentation.pgvector_repository import DB
from src.components.segmentation.models import (
    SearchResultComplexDto,
)

from dotenv import load_dotenv

load_dotenv()

logger = getLogger('api')


def slugify(text: str) -> str:
    text = unidecode(text).lower()
    text = re.sub(r'[^a-z0-9]+', '-', text)
    return text.strip('-')


db: DB | None = None


def get_db() -> DB:
    global db
    if db is None:
        db = DB(
            host=os.getenv('DB_HOST', 'localhost'),
            port=int(os.getenv('DB_PORT', '5432')),
            dbname=os.getenv('DB_NAME', 'podcast_shownotes'),
            user=os.getenv('DB_USER', 'postgres'),
            password=os.getenv('DB_PASSWORD', 'postgres'),
            embedding_model=os.getenv(
                'EMBEDDING_MODEL', 'deepvk/USER-bge-m3'
            ),
        )
    return db


@asynccontextmanager
async def lifespan(application: FastAPI):
    get_db()
    logger.info('DB connection established')
    yield


app = FastAPI(lifespan=lifespan)


class SearchBody(BaseModel):
    query: str
    podcast_slug: str
    limit: int = 10
    offset: int = 0


class CreatePodcastBody(BaseModel):
    name: str
    rss_url: str
    slug: str | None = None
    language: str = 'en'
    max_episodes: int | None = None


@app.get('/ping')
def ping() -> dict:
    return {
        'pong': True,
        'utcnow': datetime.now(timezone.utc).isoformat(),
    }


@app.post('/podcasts', status_code=201)
def create_podcast(body: CreatePodcastBody) -> dict:
    from src.app.tasks import check_podcast_feed
    from src.shared.podcast_config import (
        PodcastConfig,
        add_podcast_config,
    )

    database = get_db()
    slug = body.slug if body.slug else slugify(body.name)

    existing = database.get_podcast_id(slug)
    if existing is not None:
        raise HTTPException(
            status_code=409,
            detail='Podcast with this slug already exists',
        )

    podcast_id = database.create_podcast(
        slug=slug,
        name=body.name,
        rss_url=body.rss_url,
        language=body.language,
    )

    # todo: not great but fast.
    # Add to podcasts.yml for Celery tasks
    add_podcast_config(PodcastConfig(
        name=body.name,
        slug=slug,
        rss_url=body.rss_url,
        language=body.language,
    ))

    # Trigger RSS feed check to process episodes
    check_podcast_feed.delay(slug, max_episodes=body.max_episodes)

    return {
        'id': str(podcast_id),
        'slug': slug,
        'name': body.name,
        'rss_url': body.rss_url,
        'language': body.language,
    }


@app.post('/search')
def search_post(body: SearchBody) -> list[dict]:
    database = get_db()
    podcast_id = database.get_podcast_id(body.podcast_slug)
    if podcast_id is None:
        raise HTTPException(
            status_code=404,
            detail='Podcast not found',
        )

    results = database.find_similar(
        query=body.query,
        limit=body.limit,
        offset=body.offset,
        podcast_id=podcast_id,
    )
    return [r.model_dump() for r in results.results]


@app.get('/v2/search')
def search_v2(
    query: str,
    podcast_slug: str,
    limit: int = 10,
    offset: int = 0,
    include_episode: bool = False,
    language: str = Query(default='ru'),
) -> list[dict]:
    database = get_db()
    podcast_id = database.get_podcast_id(podcast_slug)
    if podcast_id is None:
        raise HTTPException(
            status_code=404,
            detail='Podcast not found',
        )

    results = database.find_similar_complex(
        query=query,
        limit=limit,
        offset=offset,
        language=language,
        podcast_id=podcast_id,
    )

    if not include_episode:
        return [
            r.model_dump() for r in results.results
        ]

    enriched: list[dict] = []
    for r in results.results:
        episode = None
        if r.episode is not None:
            episode = database.find_episode(r.episode)

        if episode is None:
            enriched.append(r.model_dump())
            continue

        dto = SearchResultComplexDto(
            episode=episode,
            sentence=r.sentence,
            segment=r.segment,
            distance=r.distance,
            starts_at=r.starts_at,
            ends_at=r.ends_at,
        )
        enriched.append(dto.model_dump())

    return enriched


@app.delete('/podcasts/{podcast_slug}', status_code=200)
def delete_podcast(podcast_slug: str) -> dict:
    from src.shared.podcast_config import remove_podcast_config

    database = get_db()

    # Check if podcast exists
    podcast_id = database.get_podcast_id(podcast_slug)
    if podcast_id is None:
        raise HTTPException(
            status_code=404,
            detail='Podcast not found',
        )

    # Delete from database (cascades to episodes, segments, etc.)
    database.delete_podcast(podcast_slug)

    # Delete storage directory
    storage_dir = Path(os.getenv('STORAGE_DIR', 'episodes')) / podcast_slug
    if storage_dir.exists():
        shutil.rmtree(storage_dir)

    # Remove from podcasts.yml
    remove_podcast_config(podcast_slug)

    return {
        'deleted': True,
        'slug': podcast_slug,
    }
