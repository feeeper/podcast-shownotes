import typing as t
import polars as pl
from pathlib import Path
import re
import shutil
from src.components.segmentation.embedding_builder import (
    EmbeddingBuilder,
)
from logging import getLogger
from datetime import datetime
import psycopg2.extras
import numpy as np
from dataclasses import dataclass

import logging

# Initialize console logger
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
file_handler = logging.FileHandler('utils.log')
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - '
    '%(message)s'
)
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)
logger = getLogger('utils')
logger.setLevel(logging.DEBUG)
logger.addHandler(console_handler)
logger.addHandler(file_handler)


@dataclass
class SearchResult:
    episode: int
    sentence: str
    segment: str
    distance: float


def init_db(
    host: str = 'localhost',
    port: int = 5432,
    user: str = 'postgres',
    password: str = 'postgres',
    dbname: str = 'podcast_shownotes',
) -> None:
    import pgvector
    import psycopg2

    conn = psycopg2.connect(
        host=host,
        port=port,
        dbname='postgres',
        user=user,
        password=password,
    )
    conn.autocommit = True
    cursor = conn.cursor()
    cursor.execute(f'CREATE DATABASE {dbname}')
    cursor.close()
    conn.close()

    try:
        conn = psycopg2.connect(
            host=host,
            port=port,
            dbname=dbname,
            user=user,
            password=password,
        )
        cursor = conn.cursor()
        cursor.execute(
            'CREATE EXTENSION IF NOT EXISTS vector'
        )

        conn.commit()

        # create podcasts
        cursor.execute('''
            CREATE TABLE podcasts (
                id UUID DEFAULT gen_random_uuid()
                    PRIMARY KEY,
                slug VARCHAR(128) NOT NULL UNIQUE,
                name VARCHAR(256) NOT NULL,
                rss_url VARCHAR(512) NOT NULL,
                language VARCHAR(10)
                    NOT NULL DEFAULT 'en'
        )''')

        # create episodes
        cursor.execute('''
            CREATE TABLE episodes (
                id UUID DEFAULT gen_random_uuid()
                    PRIMARY KEY,
                podcast_id UUID NOT NULL
                    REFERENCES podcasts(id),
                episode_number int,
                rss_guid VARCHAR(512),
                episode_link VARCHAR(512),
                released_at date NOT NULL,
                title VARCHAR(256) NOT NULL,
                shownotes VARCHAR(8096) NOT NULL,
                shownotes_embedding
                    VECTOR(1024) NOT NULL
        )''')
        cursor.execute(
            'CREATE INDEX ix_shownotes_embedding '
            'ON episodes USING hnsw '
            '(shownotes_embedding vector_cosine_ops);'
        )
        cursor.execute(
            'CREATE UNIQUE INDEX ix_podcast_rss_guid '
            'ON episodes (podcast_id, rss_guid);'
        )

        # create speakers
        cursor.execute('''
            CREATE TABLE speakers (
                id UUID DEFAULT gen_random_uuid()
                    PRIMARY KEY,
                name VARCHAR(256) NOT NULL,
                link VARCHAR(256)
        )''')

        # create speaker <-> episode
        cursor.execute('''
            CREATE TABLE speaker_episode (
                id UUID DEFAULT gen_random_uuid()
                    PRIMARY KEY,
                speaker_id UUID NOT NULL,
                episode_id UUID NOT NULL,
                CONSTRAINT fk_episode
                    FOREIGN KEY(episode_id)
                    REFERENCES episodes(id)
                    ON DELETE SET NULL,
                CONSTRAINT fk_speaker
                    FOREIGN KEY(speaker_id)
                    REFERENCES speakers(id)
                    ON DELETE SET NULL
        )''')

        # create segments
        cursor.execute('''
            CREATE TABLE segments (
                id UUID DEFAULT gen_random_uuid()
                    PRIMARY KEY,
                episode_id UUID NOT NULL,
                start_at float4 NOT NULL,
                end_at float4 NOT NULL,
                text VARCHAR NOT NULL,
                segment_number int NOT NULL,
                segment_embedding
                    VECTOR(1024) NOT NULL,
                CONSTRAINT fk_episode
                    FOREIGN KEY(episode_id)
                    REFERENCES episodes(id)
                    ON DELETE SET NULL
        )''')
        cursor.execute(
            'CREATE INDEX ix_segment_embedding '
            'ON segments USING hnsw '
            '(segment_embedding vector_cosine_ops);'
        )

        # create sentences
        cursor.execute('''
            CREATE TABLE sentences (
                id UUID DEFAULT gen_random_uuid()
                    NOT NULL,
                segment_id UUID NOT NULL,
                speaker_id UUID NULL,
                speaker_number int NULL,
                start_at float4 NOT NULL,
                end_at float4 NOT NULL,
                "text" varchar NOT NULL,
                sentence_number int NOT NULL,
                sentence_embedding
                    VECTOR(1024) NOT NULL,
                CONSTRAINT fk_segment
                    FOREIGN KEY(segment_id)
                    REFERENCES segments(id)
                    ON DELETE SET NULL,
                CONSTRAINT fk_speaker
                    FOREIGN KEY(speaker_id)
                    REFERENCES speakers(id)
                    ON DELETE SET NULL
            )''')
        cursor.execute(
            'CREATE INDEX ix_sentence_embedding '
            'ON sentences USING hnsw '
            '(sentence_embedding vector_cosine_ops);'
        )
        conn.commit()

        cursor.close()
    except Exception as e:
        print(e)
    finally:
        if 'cursor' in locals():
            cursor.close()
        conn.close()


def drop_db(
    host: str = 'localhost',
    port: int = 5432,
    user: str = 'postgres',
    password: str = 'postgres',
    dbname: str = 'podcast_shownotes',
) -> None:
    import psycopg2

    try:
        conn = psycopg2.connect(
            host=host,
            port=port,
            dbname=dbname,
            user=user,
            password=password,
        )
        conn.autocommit = True
        cursor = conn.cursor()
        cursor.execute(f'DROP DATABASE {dbname}')
        cursor.close()
    except Exception as e:
        print(e)
    finally:
        if 'cursor' in locals():
            cursor.close()
        conn.close()


def get_segmentation(
    df: pl.DataFrame, target: str
) -> dict[int, str]:
    k = 0
    episodes = {}
    for group in df[
        ['ru_sentence', target, 'episode']
    ].group_by('episode'):
        topic = []
        topics = []
        for row in group[1].iter_rows():
            topic.append(row[0])
            if row[1] == 1:
                topics.append(' '.join(topic))
                topic = []
        if topic:
            topics.append(' '.join(topic))

        episodes[group[0]] = '|'.join(topics)
    return episodes


def sync_cache(from_: Path, to_: Path) -> None:
    model_sizes = ['large', 'medium', 'small']
    episodes_from = [
        x
        for x in from_.iterdir()
        if x.suffix.lower() == '.json'
    ]
    regexp = 'episode-(\\d\\d\\d\\d?)'
    episodes_nums = sorted(
        {
            re.findall(regexp, x.name)[0]
            for x in episodes_from
            if 'episode' in x.name
        }
    )
    for ep_num_from in episodes_nums:
        sync_to_dir = Path(
            to_ / str(int(ep_num_from))
        )
        if not sync_to_dir.exists():
            sync_to_dir.mkdir(parents=True)

        for size in model_sizes:
            ep_from = next(
                (
                    x
                    for x in episodes_from
                    if f'episode-{ep_num_from}' in x.name
                    and size in x.name
                ),
                None,
            )
            if not ep_from:
                continue

            ep_to = (
                sync_to_dir / f'transcription-{size}.json'
            )
            shutil.copy(ep_from, ep_to)
            break


def find(
    query: str, conn: t.Any, limit: int = 5
) -> list[t.Any]:
    cursor = conn.cursor(
        cursor_factory=psycopg2.extras.DictCursor
    )
    embedding_builder = EmbeddingBuilder()
    embedding = embedding_builder.get_embeddings(query)
    cursor.execute(
        'SELECT * FROM sentences s '
        'ORDER BY s.sentence_embedding <=> %s '
        f'LIMIT {limit}',
        (np.array(embedding.tolist()),),
    )
    results = cursor.fetchall()
    return results


def find_segment(
    query: str, conn: t.Any, limit: int = 5
) -> list[t.Any]:
    cursor = conn.cursor(
        cursor_factory=psycopg2.extras.DictCursor
    )
    embedding_builder = EmbeddingBuilder()
    embedding = embedding_builder.get_embeddings(query)
    cursor.execute(
        'SELECT * FROM segments s '
        'ORDER BY s.segment_embedding <=> %s '
        f'LIMIT {limit}',
        (np.array(embedding.tolist()),),
    )
    results = cursor.fetchall()
    return results


def find2(
    query: str, conn: t.Any, limit: int = 5
) -> list[t.Any]:
    cursor = conn.cursor(
        cursor_factory=psycopg2.extras.DictCursor
    )
    embedding_builder = EmbeddingBuilder()
    embedding = embedding_builder.get_embeddings(query)
    cursor.execute(
        f'''SELECT
            e.episode_number as episode,
            s.text as sentence,
            seg.text as segment,
            s.sentence_embedding <=> %s as distance
        FROM sentences s
        LEFT JOIN segments seg
            ON s.segment_id = seg.id
        LEFT JOIN episodes e
            ON e.id = seg.episode_id
        ORDER BY
            s.sentence_embedding <=> %s
        LIMIT {limit}''',
        (
            np.array(embedding.tolist()),
            np.array(embedding.tolist()),
        ),
    )
    results = cursor.fetchall()
    return [SearchResult(**x) for x in results]
