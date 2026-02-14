from __future__ import annotations
import json
from datetime import datetime
from logging import getLogger
from uuid import UUID

import numpy as np
import psycopg2
import psycopg2.extras
from src.components.segmentation.embedding_builder import (
    EmbeddingBuilder,
)
from src.components.segmentation.segmentation_builder import (
    SegmentationResult,
)
from pgvector.psycopg2 import register_vector

from .models import (
    SearchResultDto,
    SearchResults,
    SearchResult,
    EpisodeDto,
)

logger = getLogger('pgvector_repository')

# Map ISO 639-1 codes to PostgreSQL FTS language names
_FTS_LANGUAGE_MAP = {
    'en': 'english',
    'ru': 'russian',
    'de': 'german',
    'fr': 'french',
    'es': 'spanish',
    'it': 'italian',
    'pt': 'portuguese',
    'nl': 'dutch',
    'sv': 'swedish',
    'no': 'norwegian',
    'da': 'danish',
    'fi': 'finnish',
    'tr': 'turkish',
}


def _iso_to_fts_language(language: str) -> str:
    return _FTS_LANGUAGE_MAP.get(language, 'simple')


class DB:
    def __init__(
            self,
            host: str = 'localhost',
            port: int = 5432,
            dbname: str = 'podcast_shownotes',
            user: str = 'postgres',
            password: str = 'postgres',
            embedding_model: str = 'deepvk/USER-bge-m3',
    ) -> None:
        self.conn = psycopg2.connect(
            host=host,
            port=port,
            dbname=dbname,
            user=user,
            password=password,
        )
        register_vector(self.conn)

        psycopg2.extras.register_uuid()
        self.cursor = self.conn.cursor(
            cursor_factory=psycopg2.extras.DictCursor
        )
        self.embedder = EmbeddingBuilder(
            model_name=embedding_model
        )

    def __del__(self):
        self.cursor.close()
        self.conn.close()

    def get_podcast_id(
        self, podcast_slug: str
    ) -> UUID | None:
        self.cursor.execute(
            'SELECT id FROM podcasts WHERE slug = %s',
            (podcast_slug,),
        )
        row = self.cursor.fetchone()
        return row['id'] if row else None

    def get_podcast(
        self, podcast_slug: str
    ) -> dict | None:
        self.cursor.execute(
            '''
            SELECT id, slug, name, rss_url, language
            FROM podcasts WHERE slug = %s
            ''',
            (podcast_slug,),
        )
        row = self.cursor.fetchone()
        if row is None:
            return None
        return {
            'id': row['id'],
            'slug': row['slug'],
            'name': row['name'],
            'rss_url': row['rss_url'],
            'language': row['language'],
        }

    def create_podcast(
        self,
        slug: str,
        name: str,
        rss_url: str,
        language: str = 'en',
    ) -> UUID:
        self.cursor.execute(
            '''
            INSERT INTO podcasts (slug, name, rss_url, language)
            VALUES (%s, %s, %s, %s)
            RETURNING id
            ''',
            (slug, name, rss_url, language),
        )
        podcast_id = self.cursor.fetchone()[0]
        self.conn.commit()
        return podcast_id

    def delete_podcast(self, podcast_slug: str) -> bool:
        """
        Delete a podcast and all related data.

        Deletes in order: sentences, segments,
        speaker_episode, episodes, podcast.

        Returns True if podcast was deleted.
        """
        podcast_id = self.get_podcast_id(podcast_slug)
        if podcast_id is None:
            return False

        # Delete sentences via segments via episodes
        self.cursor.execute(
            '''
            DELETE FROM sentences
            WHERE segment_id IN (
                SELECT s.id FROM segments s
                JOIN episodes e ON e.id = s.episode_id
                WHERE e.podcast_id = %s
            )
            ''',
            (podcast_id,),
        )

        # Delete segments via episodes
        self.cursor.execute(
            '''
            DELETE FROM segments
            WHERE episode_id IN (
                SELECT id FROM episodes
                WHERE podcast_id = %s
            )
            ''',
            (podcast_id,),
        )

        # Delete speaker_episode via episodes
        self.cursor.execute(
            '''
            DELETE FROM speaker_episode
            WHERE episode_id IN (
                SELECT id FROM episodes
                WHERE podcast_id = %s
            )
            ''',
            (podcast_id,),
        )

        # Delete episodes
        self.cursor.execute(
            'DELETE FROM episodes WHERE podcast_id = %s',
            (podcast_id,),
        )

        # Delete podcast
        self.cursor.execute(
            'DELETE FROM podcasts WHERE id = %s',
            (podcast_id,),
        )

        self.conn.commit()
        return True

    def insert(
            self,
            segmentation_result: SegmentationResult,
            podcast_slug: str = 'devzen',
            rss_guid: str | None = None,
            episode_link: str | None = None,
            title: str | None = None,
            summary: str = '',
            authors: list[str] | None = None,
            published: str | None = None,
    ) -> UUID | None:
        try:
            path = segmentation_result.item

            # Resolve podcast_id
            podcast_id = self.get_podcast_id(podcast_slug)
            if podcast_id is None:
                logger.error(
                    f'Podcast "{podcast_slug}" not found'
                )
                return None

            # Try episode number from path
            try:
                episode_num = int(path.stem)
            except ValueError:
                episode_num = None

            # Check for existing episode
            if rss_guid:
                existing = self.find_episode_by_guid(
                    podcast_id, rss_guid
                )
            elif episode_num is not None:
                existing = self.find_episode(episode_num)
            else:
                existing = None

            if existing is not None:
                return existing.id

            # Determine metadata: prefer passed args,
            # fall back to episode.json on disk
            if title is None:
                episode_json_path = path / 'episode.json'
                if episode_json_path.exists():
                    with open(episode_json_path, 'r') as f:
                        episode_data = json.load(f)
                    title = episode_data.get('title', '')
                    summary = episode_data.get(
                        'summary', summary
                    )
                    authors = authors or episode_data.get(
                        'authors', []
                    )
                    published = published or episode_data.get(
                        'published', ''
                    )
                    episode_link = (
                        episode_link
                        or episode_data.get(
                            'episode_link', ''
                        )
                    )
                    rss_guid = rss_guid or episode_link
                else:
                    title = ''

            if authors is None:
                authors = []

            # Parse release date
            release_date = None
            if published:
                for fmt in (
                    '%Y-%m-%dT%H:%M:%S',
                    '%Y-%m-%d',
                    '%d.%m.%Y',
                ):
                    try:
                        release_date = datetime.strptime(
                            published, fmt
                        )
                        break
                    except ValueError:
                        continue
            if release_date is None:
                release_date = datetime.now()

            shownotes_embedding = self.embedder.get_embeddings(
                [summary or title]
            )[0].tolist()

            self.cursor.execute(
                'INSERT INTO episodes '
                '(episode_number, released_at, title, '
                'shownotes, shownotes_embedding, '
                'podcast_id, rss_guid, episode_link) '
                'VALUES '
                '(%s, %s, %s, %s, %s, %s, %s, %s) '
                'RETURNING id',
                (
                    episode_num,
                    release_date,
                    title,
                    summary,
                    shownotes_embedding,
                    podcast_id,
                    rss_guid,
                    episode_link,
                ),
            )
            episode_id = self.cursor.fetchone()[0]
            logger.info(
                f'Inserted episode "{title}" '
                f'with id {episode_id}'
            )

            total_segments_count = len(
                segmentation_result.segments
            )
            total_sentences_count = sum(
                [
                    len(x)
                    for x in (
                        segmentation_result
                        .sentences_by_segment
                    )
                ]
            )
            total_inserted_sentences = 0
            batch_size = 10
            for i in range(total_segments_count):
                segment = segmentation_result.segments[i]
                start = datetime.now()
                segment_embedding = (
                    self.embedder.get_embeddings(
                        [segment.text]
                    )[0].tolist()
                )
                segment_embedding_time = (
                    datetime.now() - start
                ).total_seconds()
                seg_data = {
                    'episode_id': episode_id,
                    'embedding': segment_embedding,
                    'start_at': segment.start_at,
                    'end_at': segment.end_at,
                    'text': segment.text,
                    'segment_number': i + 1,
                }
                start = datetime.now()
                self.cursor.execute(
                    'INSERT INTO segments '
                    '(episode_id, start_at, end_at, text, '
                    'segment_number, segment_embedding) '
                    'VALUES (%s, %s, %s, %s, %s, %s) '
                    'RETURNING id',
                    (
                        episode_id,
                        seg_data['start_at'],
                        seg_data['end_at'],
                        seg_data['text'],
                        seg_data['segment_number'],
                        seg_data['embedding'],
                    ),
                )
                segment_id = self.cursor.fetchone()[0]
                logger.info(
                    f'Inserted segment '
                    f'{seg_data["segment_number"]}'
                    f'/{total_segments_count} '
                    f'(episode={path.stem}) '
                    f'with id "{segment_id}" took '
                    f'{(datetime.now() - start).total_seconds()}s '
                    f'(embedding took '
                    f'{segment_embedding_time}s)'
                )

                segment_sentences = (
                    segmentation_result
                    .sentences_by_segment[i]
                )
                inserted_sentences = 0
                total_segment_sentences = len(
                    segment_sentences
                )
                for j in range(
                    0, len(segment_sentences), batch_size
                ):
                    sentences_batch = segment_sentences[
                        j:j + batch_size
                    ]
                    texts = [x.text for x in sentences_batch]
                    start = datetime.now()
                    sentence_embeddings_ = [
                        x.tolist()
                        for x in self.embedder.get_embeddings(
                            texts
                        )
                    ]
                    sentence_embeddings_time = (
                        datetime.now() - start
                    ).total_seconds()

                    data = [
                        (
                            segment_id,
                            None,
                            s.speaker_id,
                            sentence_embeddings_[k],
                            s.start_at,
                            s.end_at,
                            s.text,
                            s.num,
                        )
                        for k, s in enumerate(
                            sentences_batch
                        )
                    ]
                    start = datetime.now()
                    psycopg2.extras.execute_values(
                        self.cursor,
                        '''INSERT INTO sentences (
                            segment_id,
                            speaker_id,
                            speaker_number,
                            sentence_embedding,
                            start_at,
                            end_at,
                            text,
                            sentence_number
                        ) VALUES %s''',
                        data,
                    )
                    inserted_sentences += len(data)
                    total_inserted_sentences += len(data)
                    logger.info(
                        f'Inserted '
                        f'{inserted_sentences}/'
                        f'{total_segment_sentences} '
                        f'sentences (total: '
                        f'{total_inserted_sentences}/'
                        f'{total_sentences_count}) took '
                        f'{(datetime.now() - start).total_seconds()}s '
                        f'(embedding took '
                        f'{sentence_embeddings_time}s)'
                    )

            self.conn.commit()
        except Exception as e:
            logger.error(e)
            return None

        return episode_id

    def find_episode(
        self, episode_num: int
    ) -> EpisodeDto | None:
        logger.info(f'Find episode {episode_num}')
        self.cursor.execute(
            '''
            SELECT
                e.id,
                e.episode_number,
                e.title,
                e.released_at,
                e.shownotes,
                e.episode_link,
                p.slug as podcast_slug
            FROM episodes e
            JOIN podcasts p ON p.id = e.podcast_id
            WHERE episode_number = %s
            ''',
            (episode_num,),
        )
        row = self.cursor.fetchone()
        if row is None:
            return None

        return EpisodeDto(
            id=row['id'],
            podcast_slug=row['podcast_slug'],
            num=row['episode_number'],
            title=row['title'],
            release_date=row['released_at'],
            shownotes=row['shownotes'],
            hosts=[],
            link=row['episode_link'] or '',
        )

    def find_episode_by_guid(
        self, podcast_id: UUID, rss_guid: str
    ) -> EpisodeDto | None:
        logger.info(
            f'Find episode by guid {rss_guid}'
        )
        self.cursor.execute(
            '''
            SELECT
                e.id,
                e.episode_number,
                e.title,
                e.released_at,
                e.shownotes,
                e.episode_link,
                p.slug as podcast_slug
            FROM episodes e
            JOIN podcasts p ON p.id = e.podcast_id
            WHERE e.podcast_id = %s
              AND e.rss_guid = %s
            ''',
            (podcast_id, rss_guid),
        )
        row = self.cursor.fetchone()
        if row is None:
            return None

        return EpisodeDto(
            id=row['id'],
            podcast_slug=row['podcast_slug'],
            num=row['episode_number'],
            title=row['title'],
            release_date=row['released_at'],
            shownotes=row['shownotes'],
            hosts=[],
            link=row['episode_link'] or '',
        )

    def find_similar(
            self,
            query: str,
            limit: int = 10,
            offset: int = 0,
            podcast_id: UUID | None = None,
    ) -> SearchResults:
        embedding = self.embedder.get_embeddings(query)

        podcast_filter = ''
        params: list = [
            np.array(embedding.tolist()),
            np.array(embedding.tolist()),
        ]
        if podcast_id is not None:
            podcast_filter = (
                'WHERE e.podcast_id = %s'
            )
            params.append(podcast_id)

        self.cursor.execute(
            f'''SELECT
                s.id,
                e.episode_number as episode,
                p.slug as podcast_slug,
                s.text as sentence,
                s.start_at as starts_at,
                s.end_at as ends_at,
                seg.text as segment,
                s.sentence_embedding <=> %s as distance
            FROM sentences s
            LEFT JOIN segments seg
                ON s.segment_id = seg.id
            LEFT JOIN episodes e
                ON e.id = seg.episode_id
            LEFT JOIN podcasts p
                ON p.id = e.podcast_id
            {podcast_filter}
            ORDER BY
                s.sentence_embedding <=> %s
            OFFSET {offset}
            LIMIT {limit}''',
            tuple(params),
        )
        records = self.cursor.fetchall()
        results = SearchResults(
            results=[
                SearchResultDto(**x) for x in records
            ]
        )

        self._log_query_with_results(
            query,
            [SearchResult(**x) for x in records[:10]],
            comment='search_v1',
        )

        return results

    def find_similar_complex(
        self,
        query: str,
        limit: int = 10,
        offset: int = 0,
        language: str = 'ru',
        podcast_id: UUID | None = None,
    ) -> SearchResults:
        embedding = self.embedder.get_embeddings(query)
        fts_lang = _iso_to_fts_language(language)

        podcast_filter = ''
        extra_params: list = []
        if podcast_id is not None:
            podcast_filter = (
                'AND e.podcast_id = %s'
            )
            extra_params.append(podcast_id)

        self.cursor.execute(
            'SET hnsw.ef_search = 200;'
        )
        self.cursor.execute(
            f"""SELECT
                id,
                episode_number as episode,
                podcast_slug,
                n.text as sentence,
                starts_at,
                ends_at,
                segment,
                (n.word_similarity + n.cosine_distance
                 + n.fts_distance) as distance
            FROM
            (
                SELECT
                    s.id,
                    s.text,
                    (1-ts_rank(
                        to_tsvector(%s, s."text"),
                        plainto_tsquery(%s)
                    )) as fts_distance,
                    (1-word_similarity(
                        lower(s."text"), lower(%s)
                    )) as word_similarity,
                    s.sentence_embedding <=> %s
                        as cosine_distance,
                    e.episode_number,
                    p.slug as podcast_slug,
                    s.start_at as starts_at,
                    s.end_at as ends_at,
                    seg.text as segment
                FROM sentences s
                LEFT JOIN segments seg
                    ON seg.id = s.segment_id
                LEFT JOIN episodes e
                    ON e.id = seg.episode_id
                LEFT JOIN podcasts p
                    ON p.id = e.podcast_id
                WHERE 1=1 {podcast_filter}
                ORDER BY
                    s.sentence_embedding <=> %s ASC
                LIMIT 100
            ) as n
            LIMIT %s
            OFFSET %s""",
            (
                fts_lang,
                query,
                query,
                np.array(embedding.tolist()),
                *extra_params,
                np.array(embedding.tolist()),
                limit,
                offset,
            ),
        )

        records = self.cursor.fetchall()
        results = SearchResults(
            results=[
                SearchResultDto(**x) for x in records
            ]
        )

        self._log_query_with_results(
            query,
            [SearchResult(**x) for x in records[:10]],
            comment='search_v2',
        )

        return results

    def delete(self, episode_id: UUID) -> None:
        raise NotImplementedError

    def _log_query_with_results(
        self,
        query: str,
        results: list[SearchResult],
        comment: str | None = None,
    ) -> None:
        try:
            self.cursor.execute(
                'INSERT INTO search_history '
                '(query, comment) '
                'VALUES (%s, %s) RETURNING id',
                (query, comment),
            )
            search_history_id = self.cursor.fetchone()[0]

            result_data = [
                (
                    search_history_id,
                    str(result.id),
                    result.distance,
                )
                for result in results
            ]

            psycopg2.extras.execute_values(
                self.cursor,
                'INSERT INTO search_results '
                '(search_history_id, sentence_id, '
                'similarity) VALUES %s',
                result_data,
                template='(%s, %s::uuid, %s)',
            )
            self.conn.commit()
        except Exception as e:
            logger.error(
                f'Error logging query with results: {e}'
            )
            self.conn.rollback()
