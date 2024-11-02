from __future__ import annotations

import json
from logging import getLogger
from datetime import datetime
from src.components.segmentation.segmentation_builder import SegmentationResult
from src.components.segmentation.embedding_builder import EmbeddingBuilder
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, MilvusException
from pymilvus.exceptions import SchemaNotReadyException


segments_collection_name = 'segments'
sentences_collection_name = 'sentences'
episodes_collection_name = 'episodes'
logger = getLogger('segmentation_repository')


class SegmentationRepository:
    def __init__(self):
        self.embedder = EmbeddingBuilder()
        connections.connect(host='localhost', port='19530')
        self._segments_collection = self._get_or_create_segments_collection()
        self._sentences_collection = self._get_or_create_sentences_collection()
        self._episodes_collection = self._get_or_create_episodes_collection()

    def save(self, segmentation_result: SegmentationResult) -> int | None:
        assert len(segmentation_result.segments) == len(segmentation_result.sentences_by_segment), \
            'Each segment text has to have it\'s own segments'

        batch_size = 10
        total_segments_count = len(segmentation_result.segments)

        episode = json.load(open(segmentation_result.item / 'metadata.json', 'r'))
        pub_ts = datetime.strptime(episode['release_date'], '%d.%m.%Y').timestamp()
        episode_data = {
            'released_at': int(pub_ts),
            'title': episode['title'],
            'shownotes': episode['shownotes'],
            'shownotes_embedding': self.embedder.get_embeddings([episode['shownotes']])[0],
            'speakers': [f'{x["name"]} ({x.get("href", "")})' for x in episode['speakers']],
        }

        episode_insert_result = self._episodes_collection.insert(episode_data)
        if episode_insert_result.err_count > 0:
            logger.error(f'There are {episode_insert_result.err_count} errors were occur while inserting metadata for episode "{segmentation_result.item}"')
            return None
        else:
            episode_id = episode_insert_result.primary_keys[0]

        segment_field = next((x for x in self._segments_collection.schema.fields if x.name == 'text'), None)
        if segment_field is None:
            raise ValueError('text field not found in segments collection schema')
        segment_max_length = int(
            segment_field.params['max_length'] / 2) if 'max_length' in segment_field.params else int(65_535 / 2)

        sentence_field = next((x for x in self._sentences_collection.schema.fields if x.name == 'text'), None)
        if sentence_field is None:
            raise ValueError('text field not found in sentences collection schema')
        sentence_max_length = int(
            sentence_field.params['max_length'] / 2) if 'max_length' in sentence_field.params else int(65_535 / 2)

        for i in range(total_segments_count):
            segment = segmentation_result.segments[i]
            segment_sentences = segmentation_result.sentences_by_segment[i]

            segment_embedding = self.embedder.get_embeddings([segment.text])[0]
            segment_req = {
                'episode_id': episode_id,
                'embedding': segment_embedding,
                'start_at': segment.start_at,
                'end_at': segment.end_at,
                'text': segment.text[:segment_max_length],
                'num': i + 1,
            }

            try:
                segment_insert_result = self._segments_collection.insert(segment_req)
                if segment_insert_result.err_count != 0:
                    logger.error(f'Error inserting segment {segment_req}')
                    return None
                else:
                    segment_id = segment_insert_result.primary_keys[0]
            except MilvusException as me:
                logger.error(f'Error inserting segment {segment_req}')
                raise

            for j in range(0, len(segment_sentences), batch_size):
                sentences_batch = segment_sentences[j:j + batch_size]
                texts = [x.text for x in sentences_batch]
                sentence_embeddings_ = self.embedder.get_embeddings(texts)

                data = [
                    {
                        'segment_id': segment_id,
                        'episode_id': episode_id,
                        'embedding': sentence_embeddings_[k],
                        'start_at': s.start_at,
                        'end_at': s.end_at,
                        'text': s.text[:sentence_max_length],
                        'num': s.num
                    } for k, s in enumerate(sentences_batch)
                ]

                sentences_insert_result = self._sentences_collection.insert(data)
                if sentences_insert_result.err_count > 0:
                    logger.error(f'There are {sentences_insert_result.err_count} errors were occur ' +
                                 f'while inserting sentences [{sentences_batch[0].num}:{sentences_batch[-1].num}] ' +
                                 f'for segment "{segment.num}" of episode "{segmentation_result.item}"')
                    return None

        return episode_id

    def find(self, item: str) -> SegmentationResult:
        ...

    def delete(self, episode_id: int) -> bool:
        self._episodes_collection.load()
        self._segments_collection.load()
        self._sentences_collection.load()

        episode_delete_result = self._episodes_collection.delete(f'id in [{episode_id}]')
        if episode_delete_result.err_count > 0:
            logger.error(f'Error deleting episode with id {episode_id}')
            return False

        segments_delete_result = self._segments_collection.delete(f'episode_id in [{episode_id}]')
        if segments_delete_result.err_count > 0:
            logger.error(f'Error deleting segments with episode_id {episode_id}')
            return False

        sentences_delete_result = self._sentences_collection.delete(f'episode_id in [{episode_id}]')
        if sentences_delete_result.err_count > 0:
            logger.error(f'Error deleting sentences with episode_id {episode_id}')
            return False

        return True

    @staticmethod
    def _get_or_create_segments_collection() -> Collection:
        try:
            segments_collection = Collection(segments_collection_name)
            logger.info(f'Collection "{segments_collection_name}" exists')
        except SchemaNotReadyException as e:
            logger.info(f'Collection "{segments_collection_name}" doesn\'t exist. Create new one.')
            default_fields = [
                FieldSchema(name='id', dtype=DataType.INT64, is_primary=True, auto_id=True, description='id'),
                FieldSchema(name='episode_id', dtype=DataType.INT64, description='episode id'),
                FieldSchema(name='embedding', dtype=DataType.FLOAT_VECTOR, dim=1_024, description='segment\' embedding'),
                FieldSchema(name='start_at', dtype=DataType.FLOAT, description='timestamp when the segment starts'),
                FieldSchema(name='end_at', dtype=DataType.FLOAT, description='timestamp when the segment ends'),
                FieldSchema(name='text', dtype=DataType.VARCHAR, description='segment\'s text', max_length=65_535),
                FieldSchema(name='num', dtype=DataType.INT16, description='segment number in the episode')
            ]
            default_schema = CollectionSchema(fields=default_fields, description='devzen segments')
            segments_collection = Collection(name=segments_collection_name, schema=default_schema)
            # create indexes
            segments_collection.create_index(
                field_name='embedding',
                index_params={
                    'metric_type': 'COSINE',
                    'index_type': 'HNSW',
                    'params': {
                        'M': 128,
                        'efConstruction': 42
                    }
                }
            )
        return segments_collection

    @staticmethod
    def _get_or_create_sentences_collection() -> Collection:
        try:
            sentences_collection = Collection(sentences_collection_name)
            logger.info(f'Collection "{sentences_collection_name}" exists')
        except SchemaNotReadyException as e:
            logger.info(f'Collection "{sentences_collection_name}" doesn\'t exist. Create new one.')
            default_fields = [
                FieldSchema(name='id', dtype=DataType.INT64, is_primary=True, auto_id=True, description='id'),
                FieldSchema(name='segment_id', dtype=DataType.INT64, description='segment id'),
                FieldSchema(name='episode_id', dtype=DataType.INT64, description='episode id'),
                FieldSchema(name='embedding', dtype=DataType.FLOAT_VECTOR, dim=1_024, description='segment\' embedding'),
                FieldSchema(name='start_at', dtype=DataType.FLOAT, description='timestamp when the segment starts'),
                FieldSchema(name='end_at', dtype=DataType.FLOAT, description='timestamp when the segment ends'),
                FieldSchema(name='text', dtype=DataType.VARCHAR, description='sentence\'s text', max_length=4_096),
                FieldSchema(name='num', dtype=DataType.INT16, description='sentence number in the segment')
            ]
            default_schema = CollectionSchema(fields=default_fields, description='devzen segments')
            sentences_collection = Collection(name=sentences_collection_name, schema=default_schema)
            # create indexes
            sentences_collection.create_index(
                field_name='embedding',
                index_params={
                    'metric_type': 'COSINE',
                    'index_type': 'HNSW',
                    'params': {
                        'M': 128,
                        'efConstruction': 42
                    }
                }
            )

        return sentences_collection

    @staticmethod
    def _get_or_create_episodes_collection() -> Collection:
        try:
            metadata_collection = Collection(episodes_collection_name)
            logger.info(f'Collection "{episodes_collection_name}" exists')
        except SchemaNotReadyException as e:
            logger.info(f'Collection "{episodes_collection_name}" doesn\'t exist. Create new one.')
            default_fields = [
                FieldSchema(name='id', dtype=DataType.INT64, is_primary=True, auto_id=True, description='id'),
                FieldSchema(name='released_at', dtype=DataType.INT64, description='episode\'s release date'),
                FieldSchema(name='title', dtype=DataType.VARCHAR, description='episode\'s title', max_length=256),
                FieldSchema(name='shownotes', dtype=DataType.VARCHAR, description='shownotes\'s text', max_length=8_096),
                FieldSchema(name='speakers', dtype=DataType.ARRAY, element_type=DataType.VARCHAR, description='hosts', max_length=256, max_capacity=16),
                FieldSchema(name='shownotes_embedding', dtype=DataType.FLOAT_VECTOR, dim=1_024, description='episode shownote\'s embedding'),
            ]
            default_schema = CollectionSchema(fields=default_fields, description='devzen episodes')
            metadata_collection = Collection(name=episodes_collection_name, schema=default_schema)
            # create indexes
            metadata_collection.create_index(
                field_name='shownotes_embedding',
                index_params={
                    'metric_type': 'COSINE',
                    'index_type': 'HNSW',
                    'params': {
                        'M': 128,
                        'efConstruction': 42
                    }
                }
            )

        return metadata_collection
