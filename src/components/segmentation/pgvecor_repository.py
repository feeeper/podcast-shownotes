from logging import getLogger
import psycopg2
import psycopg2.extras
import json
from datetime import datetime

from src.components.segmentation.embedding_builder import EmbeddingBuilder
from src.components.segmentation.segmentation_builder import SegmentationResult


logger = getLogger('pgvector_repository')


class DB:
    def __init__(
            self,
            host: str = 'localhost',
            port: int = 5432,
            dbname: str = 'podcast_shownotes',
            user: str = 'postgres',
            password: str = 'postgres',
    ) -> None:
        self.conn = psycopg2.connect(
            host=host,
            port=port,
            dbname=dbname,
            user=user,
            password=password
        )
        self.cursor =  self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        self.embedder = EmbeddingBuilder()

    def __del__(self):
        self.cursor.close()
        self.conn.close()

    def insert(
            self,
            segmentation_result: SegmentationResult
    ) -> int | None:
        episode = json.load(open(segmentation_result.item / 'metadata.json', 'r'))
        batch_size = 100
        total_segments_count = len(segmentation_result.segments)

        episode = json.load(open(segmentation_result.item / 'metadata.json', 'r'))

        self.cursor.execute(
            'INSERT INTO episodes (episode_number, released_at, title, shownotes, shownotes_embedding) VALUES (%s, %s, %s, %s, %s) RETURNING id',
            (
                int(segmentation_result.item.stem),
                datetime.strptime(episode['release_date'], '%d.%m.%Y'),
                episode['title'],
                episode['shownotes'],
                self.embedder.get_embeddings([episode['shownotes']])[0].tolist()
            )
        )
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

        total_segments_count = len(segmentation_result.segments)
        batch_size = 100
        for i in range(total_segments_count):
            segment = segmentation_result.segments[i]
            segment_embedding = self.embedder.get_embeddings([segment.text])[0].tolist()
            segment = {
                'episode_id': episode_id,
                'embedding': segment_embedding,
                'start_at': segment.start_at,
                'end_at': segment.end_at,
                'text': segment.text,
                'segment_number': i + 1,
            }
            self.cursor.execute(
                'INSERT INTO segments (episode_id, start_at, end_at, text, segment_number, segment_embedding) VALUES (%s, %s, %s, %s, %s, %s) RETURNING id',
                (
                    episode_id,
                    segment['start_at'],
                    segment['end_at'],
                    segment['text'],
                    segment['segment_number'],
                    segment['embedding']
                )
            )
            segment_id = self.cursor.fetchone()[0]
            logger.info(f'Inserted segment #{segment["segment_number"]} with id {segment_id}')

            segment_sentences = segmentation_result.sentences_by_segment[i]
            inserted_sentences = 0
            total_segment_sentences = len(segment_sentences)
            for j in range(0, len(segment_sentences), batch_size):
                sentences_batch = segment_sentences[j:j + batch_size]
                texts = [x.text for x in sentences_batch]
                sentence_embeddings_ =  [x.tolist() for x in self.embedder.get_embeddings(texts)]

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
                logger.info(f'Inserted {inserted_sentences}/{total_segment_sentences} sentences')

        self.conn.commit()
