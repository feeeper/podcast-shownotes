# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Build "split by topic" dataset

# %% [markdown]
# The first task is to build dataset to validate any algorithm that can build shownotes. The dataset has to contain timestamp and topic title.
#
# As a reference I will use the Russian podcast called [DevZen](https://devzen.ru) about tech, IT, programming and databases (mostly PostgreSQL).
#
# The podcast has a timestamps for a discussed topics from the [120th episode](https://devzen.ru/episode-0120/). So at the beginning let's take a look into the 120th episode data.

# %% [markdown]
# ## Build a dataset with timestamps and title for each shownote

# %%
import sys
sys.path.append('../src')

import pandas as pd
import numpy as np
import polars as pl
from tqdm import notebook
import json
from datetime import datetime
import time
import re
from pathlib import Path
from itertools import groupby
from shownotes import get_topic_texts, get_shownotes_with_timestamps, get_sentences_with_timestamps, get_sentences_with_timestamps_by_letter
from models import Shownotes, Segment, Transcription
import pickle

pl.Config(fmt_str_lengths=100)

data_path = Path('/mnt/d/Projects/podcast-shownotes')

# %%
episode_number_regex = re.compile('https?\:\/\/devzen\.ru\/episode-0*(?P<episode_number>\d*)')
def get_episode_num_from_href(href: str) -> int:   
    search_result = episode_number_regex.search(href)
    episode_number = int(search_result.group('episode_number'))
    return episode_number


# %%
df = pl.read_ndjson(data_path / 'dataset.jsonl').sort('href')
df = df.with_columns(pl.col('release_date').str.to_date('%d.%m.%Y'))
df = df.with_columns((pl.col('href').apply(get_episode_num_from_href).alias('episode')))
df.sample(5)

# %%
episode = df.filter(pl.col('episode') == 120)
episode

# %%
episode_shownotes = episode['shownotes'][0]
print(episode_shownotes)


# %%
def get_shownotes_with_timestamps_for_episode(df: pl.DataFrame, episode_num: int) -> list[tuple]:
    shownotes_text = df.filter(pl.col('episode') == episode_num)['shownotes'][0]
    shownotes_source = get_shownotes_with_timestamps(shownotes_text)
    shownotes = [Shownotes(*x) for x in shownotes_source]
    return shownotes


# %%
get_shownotes_with_timestamps_for_episode(df, 120)

# %%
timestamps = [(x, y.timestamp, y.title) for x in df['episode'] for y in get_shownotes_with_timestamps_for_episode(df, x)]
timestamps_df = pl.from_records(timestamps, schema={'episode': int, 'timestamp': int, 'title': str})
timestamps_df.shape

# %%
timestamps_df.sample(10).sort('episode')

# %%
timestamps_df.write_csv('../data/timestamps.csv')


# %% [markdown]
# ## Add transcript to the timestamp datasets

# %%
def get_transcription_file(episode_number: int) -> Path:
    sizes = ['large', 'medium', 'small']

    for size in sizes:        
        if Path(data_path / 'episodes' / f'episode-{episode_number}.mp3-{size}.json').exists():
            return Path(data_path / 'episodes' / f'episode-{episode_number}.mp3-{size}.json'), size
        elif Path(data_path / 'episodes' / f'episode-0{episode_number}.mp3-{size}.json').exists():
            return Path(data_path / 'episodes' / f'episode-0{episode_number}.mp3-{size}.json'), size


# %%
episode_numbers = timestamps_df['episode'].unique()

def get_topic_texts_for_episode(df: pl.DataFrame, episode_number: int, debug: bool = False) -> tuple[list[Shownotes], str]:
    def _last_timestamp_bigger_than_last_segment_end() -> bool:
        return timestamps[-1].timestamp > transcription.segments[-1].end
    
    transcription_path, size = get_transcription_file(episode_number)
    with open(transcription_path, 'r', encoding='utf8') as f:
        transcription_data = json.load(f)
        segments = [Segment(**x) for x in transcription_data['segments']]
        transcription = Transcription(
            transcription_data['text'],
            segments,
            transcription_data['language'])
        
        text = transcription.text
        timestamps = get_shownotes_with_timestamps_for_episode(df, episode_number)

        if _last_timestamp_bigger_than_last_segment_end():
            print(f'WARN: episode={episode_number}\tlast timestamp={timestamps[-1].timestamp}\tlast segment\'s end={transcription.segments[-1].end}')
        
        topics = get_topic_texts(transcription, timestamps), size

    return topics


# %%
topics = []
failed_episodes = []
episode_numbers = timestamps_df['episode'].unique()

for episode_number in episode_numbers:
    try:
        human_shownotes = get_shownotes_with_timestamps_for_episode(df, episode_number)
        episode_shownotes, size = get_topic_texts_for_episode(df, episode_number)
        topics += [(episode_number, sn.timestamp, sn.title, hsn.timestamp, hsn.title) for sn, hsn in zip(episode_shownotes, human_shownotes)]
    except Exception as e:
        failed_episodes.append(episode_number)
        raise

pd.DataFrame(topics)

# %%
human_and_whispertimestamps_with_topics = pd.DataFrame(topics, columns=['episode', 'whisper_timestamp', 'topic_text', 'human_timestamp', 'human_title'])
human_and_whispertimestamps_with_topics

# %%
human_and_whispertimestamps_with_topics.to_csv('../data/human_and_whispertimestamps_with_topics.csv', index=None)

# %%
topics[1]

# %% [markdown]
# # Build topics dataset with splitted by sentences using stanza

# %%
from stanza import Pipeline

# %%
pipeline = Pipeline(lang='ru', processors='tokenize')

# %%
episodes = human_and_whispertimestamps_with_topics['episode'].unique()


# %%
def get_transcription_for_episode(episode_num: int) -> Transcription:
    transcription_path, size = get_transcription_file(episode_num)
    with open(transcription_path, 'r', encoding='utf8') as f:
        transcription_data = json.load(f)
        segments = [Segment(**x) for x in transcription_data['segments']]
        transcription = Transcription(
            transcription_data['text'],
            segments,
            transcription_data['language'])
    return transcription, size


# %%
def get_sentences(text: str) -> list[str]:
    return [x.text for x in pipeline(text).sentences]


# %%
sentences_timestamps = pickle.load(open('../data/sentences_with_timestamps.pkl', 'rb'))

# %%
sentences_with_timestamps = get_sentences_with_timestamps_by_letter(get_transcription_for_episode(207)[0], get_sentences, verbose=2)
sentences_with_timestamps[-5:]
# sentences_timestamps[episode_num] = sentences_with_timestamps

# %%
sentences_by_letter_timestamps_df = None

if Path('../data/sentences_by_letter_timestamps.csv').exists:
    sentences_by_letter_timestamps_df = pd.read_csv('../data/sentences_by_letter_timestamps.csv')
    sentences_by_letter_timestamps_df.head()

# %%
if sentences_by_letter_timestamps_df is None:
    sentences_by_letter_timestamps = {}
    for i, episode_num in tqdm(enumerate(episodes), total=len(episodes)):
        if episode_num not in sentences_by_letter_timestamps:
            transcription, size = get_transcription_for_episode(episode_num)
            sentences_with_timestamps = get_sentences_with_timestamps_by_letter(transcription, get_sentences)
            sentences_by_letter_timestamps[episode_num] = [(size, episode_num, *x) for x in sentences_with_timestamps]
            
    sentences_by_letter_timestamps_df = pd.DataFrame(sum(sentences_by_letter_timestamps.values(), []), columns=['size', 'episode', 'start', 'end', 'sentence'])
    sentences_by_letter_timestamps_df.head()

# %%
if not Path('../data/sentences_by_letter_timestamps.csv').exists:
    sentences_by_letter_timestamps_df.to_csv('../data/sentences_by_letter_timestamps.csv', index=None)

# %%
sentences_by_letter_timestamps_df.groupby('episode').agg(count=('size', 'count')).sort_values('count', ascending=False)

# %%
sentences_by_letter_timestamps_df[sentences_by_letter_timestamps_df['episode']==358]

# %%
sentences_by_letter_timestamps_df = pl.DataFrame(sentences_by_letter_timestamps_df)
sentences_by_letter_timestamps_df

# %%
sentences_by_letter_timestamps_df = sentences_by_letter_timestamps_df.with_columns((pl.col('sentence').apply(lambda x: len(x)).alias('sentence_length')))
sentences_by_letter_timestamps_df

# %%
np.percentile(sentences_by_letter_timestamps_df['sentence_length'], [1, 5, 25, 50, 75, 95])

# %%
sentences_by_letter_timestamps_df.filter(pl.col('sentence_length') == sentences_by_letter_timestamps_df['sentence_length'].max())

# %%
sentences_by_letter_timestamps_df.filter(pl.col('episode') == 358)

# %%
sentences_by_letter_timestamps_df.filter(pl.col('sentence_length') > 489)[44, 'sentence']

# %%
transcription_358, size = get_transcription_for_episode(358)
transcription_358_sentences = get_sentences(transcription_358.text)

# %%
transcription_358_sentences[-1][-200:]

# %%
sentences_by_letter_timestamps_df[207030,'sentence'][-200:]

# %%
sentences_by_letter_timestamps_df.groupby(['size', 'episode']).agg(pl.col('sentence').count())

# %%
sentences_by_letter_timestamps_df.groupby(pl.col('episode')).agg(pl.col('sentence_length').max())['sentence_length'].describe()

# %%
# episode with appropriate max sentence length
episodes_with_appropriate_max_sentences = sentences_by_letter_timestamps_df.groupby(pl.col('episode')).agg(pl.col('sentence_length').max()).filter(pl.col('sentence_length') <= 1280)['episode']
sentences_by_letter_timestamps_df.filter(pl.col('episode').is_in(episodes_with_appropriate_max_sentences))

# %%
len(sentences_by_letter_timestamps_df.filter(pl.col('episode').is_in(episodes_with_appropriate_max_sentences))['episode'].unique())

# %% [markdown]
# ## Build topics by sentences

# %%
from dataclasses import dataclass

@dataclass
class Sentence:
    size: str
    episode: int
    start: float
    end: float
    text: str
    length: int
    
    def __init__(self, size: str, episode: int, start: float, end: float, text: str, length: int) -> None:
        self.size = size
        self.episode = episode
        self.start = start
        self.end = end
        # get_topic_texts relies on that segments ends with whitespace
        self.text = f'{text} '
        self.length = length


def sentence_to_segment(row: Sentence) -> Segment:    
    start = row.start
    end = row.end
    text = row.text
    segment = Segment(
        start=start,
        end=end,
        text=text,
        id=0,
        avg_logprob=0,
        compression_ratio=0,
        no_speech_prob=0,
        seek=0,
        temperature=0,
        tokens=[])
    return segment


# %%
topics_from_sentences = {}
episodes_with_ok_max_length_sentences = sentences_by_letter_timestamps_df.filter(pl.col('episode').is_in(episodes_with_appropriate_max_sentences))['episode'].unique()
for episode in episodes_with_ok_max_length_sentences:
    segments = [sentence_to_segment(Sentence(*row)) for row in sentences_by_letter_timestamps_df.filter(pl.col('episode') == episode).iter_rows()]
    transcription = Transcription(' '.join([s.text for s in segments]), segments, 'ru')
    shownotes = get_shownotes_with_timestamps_for_episode(df, episode)
    topics_from_sentences[episode] = get_topic_texts(transcription, shownotes)

# %%
episode = 121
shownotes_for_episode = get_shownotes_with_timestamps_for_episode(df, episode)
for i in range(len(shownotes_for_episode)):
    print(f'Title: {shownotes_for_episode[i].title}')
    print(f'Transcript: {topics_from_sentences[episode][i].title[:1000]}\n')

# %% [markdown]
# # Reference segmentation

# %%
episode = 412
shownotes_for_episode = get_shownotes_with_timestamps_for_episode(df, episode)
for i in range(len(shownotes_for_episode)):
    print(f'Title: {shownotes_for_episode[i].title}')
    print(f'Transcript: {topics_from_sentences[episode][i].title[:3000]}\n')

# %%
sentences_by_letter_timestamps_df.filter(pl.col('episode') == episode).to_pandas().to_csv('../data/412_ep_reference.csv', index=None)

# %%
get_shownotes_with_timestamps_for_episode(df, episode_num=episode)

# %%
sss = sentences_by_letter_timestamps_df.filter(pl.col('episode') == episode).clone()
sss

# %%
shn = get_shownotes_with_timestamps_for_episode(df, episode_num=episode)
sss = sss.with_columns((pl.col('end').apply(lambda r: next((i+1 for i, x in enumerate(shn) if x.timestamp >= r), len(shn)))).alias('topic_num'))
sss.filter(pl.col('topic_num') == 2)

# %%
sss.groupby(pl.col('topic_num')).agg(pl.col('end').count()).sort('topic_num')

# %%
shownotes_by_episode_num = {}
def get_topic_num(episode_num: int, end: float) -> int:
    if episode_num in shownotes_by_episode_num:
        sn = shownotes_by_episode_num[episode_num]
    else:
        sn = get_shownotes_with_timestamps_for_episode(df, episode_num=episode_num)

    timestamps = list(zip([x.timestamp for i, x in enumerate(sn)], map(lambda x: x if x > 0 else 1_000_000, np.roll([x.timestamp for i, x in enumerate(sn)], -1))))
    return next((i+1 for i, (topic_start, topic_end) in enumerate(timestamps) if topic_end >= end), len(timestamps))


sentences_by_letter_timestamps_df = sentences_by_letter_timestamps_df.with_columns((pl.struct(['episode', 'end']).apply(lambda x: get_topic_num(x['episode'], x['end']))).alias('topic_num'))
sentences_by_letter_timestamps_df

# %%
sentences_by_letter_timestamps_df.write_csv('../data/sentences_by_letter_timestamps_topics.csv', has_header=True)

# %%
sentences_by_letter_timestamps_df.filter(pl.col('episode') == 415).write_csv('../data/415_ep_reference.csv', has_header=True)

# %%
[f'{x.timestamp},{x.title}' for x in get_shownotes_with_timestamps_for_episode(df, 415)]
