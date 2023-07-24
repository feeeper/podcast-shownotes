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
import pandas as pd
import numpy as np
import polars as pl
import json
from datetime import datetime
import re
from pathlib import Path

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
df.sample(10)

# %%
episode = df.filter(pl.col('episode') == 120)
episode

# %%
episode_shownotes = episode['shownotes'][0]
print(episode_shownotes)


# %%
def timestamp_to_seconds(hour: str, minutes: str, seconds: str|None) -> int:
    if seconds is None:
        seconds = 0
    return 60 * 60 * int(hour) + 60 * int(minutes) + int(seconds)


# %%
timestamp_regexp = re.compile('\[(?P<ts_hour>\d\d):(?P<ts_min>\d\d):?(?P<ts_sec>\d\d)?\]\s+(?P<title>.*)$')
def get_shownotes_with_timestamps(shownotes: str) -> list[tuple]:
    timestamps = [(timestamp_to_seconds(x.group('ts_hour'), x.group('ts_min'), x.group('ts_sec')), x.group('title')) for x in [timestamp_regexp.search(sn) for sn in shownotes.split('\n')] if x is not None]
    # some episodes have multiple topics related to the same timestamp.
    grouped_timestamps = []
    for k, group in groupby(timestamps, key=lambda x: x[0]):
        grouped_timestamps.append((k, '. '.join([x[1] for x in list(group)])))
    return grouped_timestamps


# %%
def get_shownotes_with_timestamps_for_episode(episode_num: int) -> list[tuple]:
    shownotes = df.filter(pl.col('episode') == episode_num)['shownotes'][0]
    return get_shownotes_with_timestamps(shownotes)


# %%
get_shownotes_with_timestamps_for_episode(120)

# %%
timestamps = [(x, *y) for x in df['episode'] for y in get_shownotes_with_timestamps_for_episode(x)]
timestamps_df = pl.from_records(timestamps, schema={'episode': int, 'timestamp': int, 'title': str})
timestamps_df.shape

# %%
timestamps_df.sample(10).sort('episode')

# %%
timestamps_df.write_csv('../data/timestamps.csv')

# %% [markdown]
# ## Add transcript to the timestamp datasets

# %%
with open(data_path / 'episodes' / 'episode-0120.mp3-large.json', 'r', encoding='utf8') as f:
    transcript = json.load(f)

# %%
text = transcript['text']
text[:200]

# %%
transcript['segments'][0]

# %%
timestamps = get_shownotes_with_timestamps_for_episode(120)
print(f'{len(timestamps)=}')
timestamps

# %%
topics = []

current_topic_text = ''
next_topic_timestamp_idx = 1
for segment in transcript['segments']:
    current_topic_text += segment['text']
    if next_topic_timestamp_idx < len(timestamps) - 1 and segment['end'] > timestamps[next_topic_timestamp_idx][0]:        
        topics.append(current_topic_text)
        current_topic_text = ''
        next_topic_timestamp_idx += 1
        
topics.append(current_topic_text)

# %%
len(topics)

# %%
topics[0]
