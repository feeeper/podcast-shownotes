from __future__ import annotations

import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from logging import getLogger

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from src.components.segmentation.pgvector_repository import DB
from src.components.segmentation.models import (
    SearchResultComplexDto,
)

from dotenv import load_dotenv

load_dotenv()

logger = getLogger('api')


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


@app.get('/ping')
def ping() -> dict:
    return {
        'pong': True,
        'utcnow': datetime.now(timezone.utc).isoformat(),
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
