from dataclasses import dataclass
from logging import getLogger
import psycopg2
import psycopg2.extras
from pgvector.psycopg2 import register_vector
import json
from datetime import datetime
from uuid import UUID
import numpy as np

from components.segmentation.embedding_builder import EmbeddingBuilder
from components.segmentation.segmentation_builder import SegmentationResult, SegmentationBuilder

from .models import (
    SearchResult,
    SearchResults,
    Episode,
)


logger = getLogger('pgvector_repository')

class DB:
    def __init__(
            self,
            host: str = 'localhost',
            port: int = 5432,
            dbname: str = 'podcast_shownotes',
            user: str = 'postgres',
            password: str = 'postgres'
    ) -> None:
        logger.debug('1')
        self.conn = psycopg2.connect(
            host=host,
            port=port,
            dbname=dbname,
            user=user,
            password=password
        )
        register_vector(self.conn)

        self.cursor = self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        self.embedder = EmbeddingBuilder()

    def __del__(self):
        self.cursor.close()
        self.conn.close()

    def insert(
            self,
            segmentation_result: SegmentationResult
    ) -> UUID | None:
        try:
            episode = json.load(open(segmentation_result.item / 'metadata.json', 'r'))
            path = segmentation_result.item

            # skip presented episode
            existing_episode = self.find_episode(int(path.stem))
            if existing_episode is not None:
                return existing_episode.id

            release_date = datetime.strptime(episode['release_date'], '%d.%m.%Y')
            self.cursor.execute(
                'INSERT INTO episodes (episode_number, released_at, title, shownotes, shownotes_embedding) VALUES (%s, %s, %s, %s, %s) RETURNING id',
                (int(path.stem), release_date, episode['title'],
                 episode['shownotes'], self.embedder.get_embeddings([episode['shownotes']])[0].tolist()))
            episode_id = self.cursor.fetchone()[0]
            logger.info(f'Inserted episode "{episode["title"]}" with id {episode_id}')

            for speaker in episode['speakers']:
                self.cursor.execute('SELECT id FROM speakers WHERE link = %s', (speaker['href'],))
                if self.cursor.rowcount > 0:
                    speaker_id = self.cursor.fetchone()[0]
                else:
                    self.cursor.execute('INSERT INTO speakers (name, link) VALUES (%s, %s) RETURNING id',
                                   (speaker['name'], speaker['href']))
                    logger.info(f'Inserted speaker {speaker["name"]}')
                    speaker_id = self.cursor.fetchone()[0]

                self.cursor.execute('INSERT INTO speaker_episode (speaker_id, episode_id) VALUES (%s, %s)',
                               (speaker_id, episode_id))
                logger.info(f'Linked speaker {speaker["name"]} to episode "{episode["title"]}"')

            start = datetime.now()
            sb = SegmentationBuilder(storage_dir=path.parent)
            segmentation_result = sb.get_segments(path)
            logger.info(f'Segmentation took {(datetime.now() - start).total_seconds()}s')
            total_segments_count = len(segmentation_result.segments)
            total_sentences_count = sum([len(x) for x in segmentation_result.sentences_by_segment])
            total_inserted_sentences = 0
            batch_size = 10
            for i in range(total_segments_count):
                segment = segmentation_result.segments[i]
                start = datetime.now()
                segment_embedding = self.embedder.get_embeddings([segment.text])[0].tolist()
                segment_embedding_time = (datetime.now() - start).total_seconds()
                segment = {
                    'episode_id': episode_id,
                    'embedding': segment_embedding,
                    'start_at': segment.start_at,
                    'end_at': segment.end_at,
                    'text': segment.text,
                    'segment_number': i + 1,
                }
                start = datetime.now()
                self.cursor.execute(
                    'INSERT INTO segments (episode_id, start_at, end_at, text, segment_number, segment_embedding) VALUES (%s, %s, %s, %s, %s, %s) RETURNING id',
                    (episode_id, segment['start_at'], segment['end_at'], segment['text'], segment['segment_number'],
                     segment['embedding']))
                segment_id = self.cursor.fetchone()[0]
                logger.info(
                    f'Inserted segment {segment["segment_number"]}/{total_segments_count} (episode={path.stem}) ' +
                    f'with id "{segment_id}" took {(datetime.now() - start).total_seconds()}s ' +
                    f'(embedding took {segment_embedding_time}s)')

                segment_sentences = segmentation_result.sentences_by_segment[i]
                inserted_sentences = 0
                total_segment_sentences = len(segment_sentences)
                for j in range(0, len(segment_sentences), batch_size):
                    sentences_batch = segment_sentences[j:j + batch_size]
                    texts = [x.text for x in sentences_batch]
                    start = datetime.now()
                    sentence_embeddings_ = [x.tolist() for x in self.embedder.get_embeddings(texts)]
                    sentence_embeddings_time = (datetime.now() - start).total_seconds()

                    data = [
                        (
                            segment_id,
                            None,
                            s.speaker_id,
                            sentence_embeddings_[k],
                            s.start_at,
                            s.end_at,
                            s.text,
                            s.num
                        ) for k, s in enumerate(sentences_batch)
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
                        data
                    )
                    inserted_sentences += len(data)
                    total_inserted_sentences += len(data)
                    logger.info(
                        f'Inserted {inserted_sentences}/{total_segment_sentences} sentences ' +
                        f'(total: {total_inserted_sentences}/{total_sentences_count}) took ' +
                        f'{(datetime.now() - start).total_seconds()}s (embedding took {sentence_embeddings_time}s)')

            self.conn.commit()
        except Exception as e:
            logger.error(e)
            return None

        return episode_id

    def find_episode(self, episode_num: int) -> Episode | None:
        logger.info(f'Find episode {episode_num}')
        self.cursor.execute('''
        select
            e.id,
            e.episode_number,
	        e.title,
            e.released_at,
            e.shownotes,
            s."name",
            s.link
        from episodes e
        left join speaker_episode se ON se.episode_id = e.id 
        left join speakers s on s.id = se.speaker_id 
        where episode_number = %s
        ''', (episode_num,))
        episode_data = self.cursor.fetchall()
        if episode_data is None:
            return None

        data = {
            'id': episode_data[0]['id'],
            'num': episode_data[0]['episode_number'],
            'title': episode_data[0]['title'],
            'release_date': episode_data[0]['released_at'],
            'shownotes': episode_data[0]['shownotes'],
            'hosts': [{'name': x['name'], 'link': x['link']} for x in episode_data]
        }
        episode = Episode(**data)
        return episode

    def find_similar(
            self,
            query: str,
            limit: int = 10,
            offset: int = 0
    ) -> SearchResults:
        embedding = self.embedder.get_embeddings(query)
        self.cursor.execute(
            f'''SELECT
                e.episode_number as episode,
                s.text as sentence,
                s.start_at as starts_at,
                s.end_at as ends_at,
                seg.text as segment,
                s.sentence_embedding <=> %s as distance
            FROM sentences s
            left join segments seg on s.segment_id = seg.id
            left join episodes e on e.id = seg.episode_id 
            ORDER by
                s.sentence_embedding <=> %s
            OFFSET {offset}
            LIMIT {limit}''',
            (np.array(embedding.tolist()), np.array(embedding.tolist()),)
        )
        records = self.cursor.fetchall()
        results = SearchResults(results=[SearchResult(**x) for x in records])
        return results

    def delete(self, episode_id: UUID) -> None:
        raise NotImplementedError
