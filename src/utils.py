import typing as t
import polars as pl
from pathlib import Path
import re
import shutil
from src.components.segmentation.embedding_builder import EmbeddingBuilder
from src.components.segmentation.segmentation_builder import SegmentationBuilder
from logging import getLogger
from datetime import datetime


import logging

# Initialize console logger
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
file_handler = logging.FileHandler('utils.log')
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)
logger = getLogger('utils')
logger.setLevel(logging.DEBUG)
logger.addHandler(console_handler)
logger.addHandler(file_handler)


def insert_not_presented_episodes(
        path_to_episodes: Path,
        host: str = 'localhost',
        port: int = 5432,
        user: str = 'postgres',
        password: str = 'postgres',
        dbname: str = 'podcast_shownotes'
) -> None:
    import psycopg2
    import psycopg2.extras

    try:
        embedding_builder = EmbeddingBuilder()
        conn = psycopg2.connect(host=host,
                                port=port,
                                dbname=dbname,
                                user=user,
                                password=password)
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cursor.execute('SELECT episode_number FROM episodes')
        episodes = set([x[0] for x in cursor.fetchall()])
        episodes_to_insert = sorted(list(set([int(x.stem) for x in path_to_episodes.iterdir() if x.is_dir()]) - episodes))
        logger.info(f'Episodes to insert: {episodes_to_insert}')
        for i in episodes_to_insert:
            start = datetime.now()
            insert_episode(
                path_to_episodes / str(i),
                host,
                port,
                user,
                password,
                dbname,
                cur=cursor,
                embedding_builder=embedding_builder
            )
            logger.info(f'Inserting episode {i} took {(datetime.now() - start).total_seconds()}s')
            start = datetime.now()
            conn.commit()
            logger.info(f'Commit took {(datetime.now() - start).total_seconds()}s')
    except Exception as e:
        logger.error(e)
    finally:
        if 'conn' in locals():
            conn.close()


def insert_episodes_range(
        episodes: t.List[int],
        host: str = 'localhost',
        port: int = 5432,
        user: str = 'postgres',
        password: str = 'postgres',
        dbname: str = 'podcast_shownotes'
) -> None:
    import psycopg2
    import psycopg2.extras

    embedding_builder = EmbeddingBuilder()
    conn = psycopg2.connect(host=host,
                        port=port,
                        dbname=dbname,
                        user=user,
                        password=password)
    cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    for i in episodes:
        start = datetime.now()
        insert_episode(Path(f'src/components/indexer/data/{i}'), host, port, user, password, dbname, cur=cursor, embedding_builder=embedding_builder)
        logger.info(f'Inserting episode {i} took {(datetime.now() - start).total_seconds()}s')
        start = datetime.now()
        conn.commit()
        logger.info(f'Commit took {(datetime.now() - start).total_seconds()}s')    


def insert_episodes(
        start_from: int,
        end_at: int,
        host: str = 'localhost',
        port: int = 5432,
        user: str = 'postgres',
        password: str = 'postgres',
        dbname: str = 'podcast_shownotes'
) -> None:
    import psycopg2
    import psycopg2.extras

    embedding_builder = EmbeddingBuilder()
    conn = psycopg2.connect(host=host,
                        port=port,
                        dbname=dbname,
                        user=user,
                        password=password)
    cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    for i in range(start_from, end_at + 1):
        start = datetime.now()
        insert_episode(Path(f'src/components/indexer/data/{i}'), host, port, user, password, dbname, cur=cursor, embedding_builder=embedding_builder)
        logger.info(f'Inserting episode {i} took {(datetime.now() - start).total_seconds()}s')
        start = datetime.now()
        conn.commit()
        logger.info(f'Commit took {(datetime.now() - start).total_seconds()}s')


def insert_episode(
        path: Path,
        host: str = 'localhost',
        port: int = 5432,
        user: str = 'postgres',
        password: str = 'postgres',
        dbname: str = 'podcast_shownotes',
        cur: t.Any = None,
        embedding_builder: t.Any = None,
    ) -> None:
    import psycopg2
    import psycopg2.extras
    import json
    from datetime import datetime

    with open(path / 'metadata.json', 'r') as f:
        episode = json.load(f)
        logger.info(f'Inserting episode {episode["title"]}')    
    
    if cur:
        cursor = cur
    else:
        conn = psycopg2.connect(host=host,
                                port=port,
                                dbname=dbname,
                                user=user,
                                password=password)
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        logger.info(f'Connected to database {dbname}')

    if not embedding_builder:
        embedding_builder = EmbeddingBuilder()    
    
    cursor.execute('INSERT INTO episodes (episode_number, released_at, title, shownotes, shownotes_embedding) VALUES (%s, %s, %s, %s, %s) RETURNING id',
                   (int(path.stem), datetime.strptime(episode['release_date'], '%d.%m.%Y'), episode['title'], episode['shownotes'], embedding_builder.get_embeddings([episode['shownotes']])[0].tolist()))
    episode_id = cursor.fetchone()[0]
    logger.info(f'Inserted episode "{episode["title"]}" with id {episode_id}')

    for speaker in episode['speakers']:
        cursor.execute('SELECT id FROM speakers WHERE link = %s', (speaker['href'],))
        if cursor.rowcount > 0:
            speaker_id = cursor.fetchone()[0]
        else:
            cursor.execute('INSERT INTO speakers (name, link) VALUES (%s, %s) RETURNING id',
                           (speaker['name'], speaker['href']))
            logger.info(f'Inserted speaker {speaker["name"]}')
            speaker_id = cursor.fetchone()[0]

        cursor.execute('INSERT INTO speaker_episode (speaker_id, episode_id) VALUES (%s, %s)',
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
        segment_embedding = embedding_builder.get_embeddings([segment.text])[0].tolist()
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
        cursor.execute('INSERT INTO segments (episode_id, start_at, end_at, text, segment_number, segment_embedding) VALUES (%s, %s, %s, %s, %s, %s) RETURNING id',
                    (episode_id, segment['start_at'], segment['end_at'], segment['text'], segment['segment_number'], segment['embedding']))
        segment_id = cursor.fetchone()[0]
        logger.info(f'Inserted segment {segment["segment_number"]}/{total_segments_count} (episode={path.stem}) with id "{segment_id}" took {(datetime.now() - start).total_seconds()}s (embedding took {segment_embedding_time}s)')

        segment_sentences = segmentation_result.sentences_by_segment[i]
        inserted_sentences = 0
        total_segment_sentences = len(segment_sentences)
        for j in range(0, len(segment_sentences), batch_size):
            sentences_batch = segment_sentences[j:j + batch_size]
            texts = [x.text for x in sentences_batch]
            start = datetime.now()
            sentence_embeddings_ =  [x.tolist() for x in embedding_builder.get_embeddings(texts)]
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
                cursor,
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
            logger.info(f'Inserted {inserted_sentences}/{total_segment_sentences} sentences (total: {total_inserted_sentences}/{total_sentences_count}) took {(datetime.now() - start).total_seconds()}s (embedding took {sentence_embeddings_time}s)')

    if 'conn' in locals():
        conn.commit()


def init_db(
        host: str = 'localhost',
        port: int = 5432,
        user: str = 'postgres',
        password: str = 'postgres',
        dbname: str = 'podcast_shownotes'
) -> None:
    import pgvector
    import psycopg2

    conn = psycopg2.connect(host=host,
                            port=port,
                            dbname='postgres',
                            user=user,
                            password=password)
    conn.autocommit = True
    cursor = conn.cursor()
    cursor.execute(f'CREATE DATABASE {dbname}')
    cursor.close()
    conn.close()

    try:
        conn = psycopg2.connect(host=host,
                                port=port,
                                dbname=dbname,
                                user=user,
                                password=password)
        cursor = conn.cursor()
        cursor.execute('CREATE EXTENSION IF NOT EXISTS vector')

        # Commit the changes and close the connection
        conn.commit()

        # create episodes
        cursor.execute('''
            CREATE TABLE episodes (
                id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
                episode_number int NOT NULL,
                released_at date NOT NULL,
                title VARCHAR(256) NOT NULL,
                shownotes VARCHAR(8096) NOT NULL,
                shownotes_embedding VECTOR(1024) NOT NULL
        )''')
        cursor.execute('CREATE INDEX ix_shownotes_embedding ON episodes USING hnsw (shownotes_embedding vector_cosine_ops);')
        cursor.execute('CREATE UNIQUE INDEX ix_episode_number ON episodes (episode_number);')
        
        # create speakers
        cursor.execute('''
            CREATE TABLE speakers (
                id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
                name VARCHAR(256) NOT NULL,
                link VARCHAR(256)
        )''')

        # create speaker <-> episode
        cursor.execute('''
            CREATE TABLE speaker_episode (
                id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
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
                id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
                episode_id UUID NOT NULL,
                start_at float4 NOT NULL,
                end_at float4 NOT NULL,
                text VARCHAR NOT NULL,
                segment_number int NOT NULL,
                segment_embedding VECTOR(1024) NOT NULL,
                CONSTRAINT fk_episode
                    FOREIGN KEY(episode_id) 
	                REFERENCES episodes(id)
	                ON DELETE SET NULL
        )''')
        cursor.execute('CREATE INDEX ix_segment_embedding ON segments USING hnsw (segment_embedding vector_cosine_ops);')
        # conn.commit()

        # create sentences
        cursor.execute('''
            CREATE TABLE sentences (
                id UUID DEFAULT gen_random_uuid() NOT NULL,
                segment_id UUID NOT NULL,
                speaker_id UUID NULL,
                speaker_number int NULL,
                start_at float4 NOT NULL,
                end_at float4 NOT NULL,
                "text" varchar NOT NULL,
                sentence_number int NOT NULL,
                sentence_embedding VECTOR(1024) NOT NULL,
                CONSTRAINT fk_segment
                    FOREIGN KEY(segment_id) 
	                REFERENCES segments(id)
	                ON DELETE SET NULL,                
                CONSTRAINT fk_speaker
                    FOREIGN KEY(speaker_id) 
	                REFERENCES speakers(id)
	                ON DELETE SET NULL
            )''')
        cursor.execute('CREATE INDEX ix_sentence_embedding ON sentences USING hnsw (sentence_embedding vector_cosine_ops);')
        conn.commit()

        # close connection
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
        dbname: str = 'podcast_shownotes'
) -> None:
    import psycopg2

    try:
        conn = psycopg2.connect(host=host,
                                port=port,
                                dbname=dbname,
                                user=user,
                                password=password)
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


def get_segmentation(df: pl.DataFrame, target: str) -> dict[int, str]:
    k = 0
    episodes = {}
    for group in df[['ru_sentence', target, 'episode']].group_by('episode'):
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
        # raise
    return episodes


def sync_cache(from_: Path, to_: Path) -> None:
    model_sizes = ['large', 'medium', 'small']
    episodes_from = [x for x in from_.iterdir() if x.suffix.lower() == '.json']
    regexp = 'episode-(\d\d\d\d?)'
    episodes_nums = sorted({re.findall(regexp, x.name)[0] for x in episodes_from if 'episode' in x.name})
    for ep_num_from in episodes_nums:
        sync_to_dir = Path(to_ / str(int(ep_num_from)))  # ep_num_from could be with leading zeros
        if not sync_to_dir.exists():
            sync_to_dir.mkdir(parents=True)

        # sync transcription, metadata and rename
        for size in model_sizes:
            ep_from = next((x for x in episodes_from if f'episode-{ep_num_from}' in x.name and size in x.name), None)
            if not ep_from:
                continue

            ep_to = sync_to_dir / f'transcription-{size}.json'
            shutil.copy(ep_from, ep_to)
            break
