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
import json
from datetime import datetime
import re
from pathlib import Path
from itertools import groupby
from shownotes import get_topic_texts, get_shownotes_with_timestamps
from transcription import Shownotes, Segment, Transcription

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
episode_numbers = timestamps_df['episode'].unique()

def get_topic_texts_for_episode(df: pl.DataFrame, episode_number: int, debug: bool = False) -> tuple[list[Shownotes], str]:
    def _last_timestamp_bigger_than_last_segment_end() -> bool:
        return timestamps[-1].timestamp > transcription.segments[-1].end
    
    def _get_transcription_file(episode_number: int) -> Path:
        sizes = ['large', 'medium', 'small']
        
        for size in sizes:        
            if Path(data_path / 'episodes' / f'episode-{episode_number}.mp3-{size}.json').exists():
                return Path(data_path / 'episodes' / f'episode-{episode_number}.mp3-{size}.json'), size
            elif Path(data_path / 'episodes' / f'episode-0{episode_number}.mp3-{size}.json').exists():
                return Path(data_path / 'episodes' / f'episode-0{episode_number}.mp3-{size}.json'), size  
    
    transcription_path, size = _get_transcription_file(episode_number)
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
topics[0]
