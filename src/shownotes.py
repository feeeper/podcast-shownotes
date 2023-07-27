import sys
sys.path.append('../src')

import json
import re
from itertools import groupby
from pathlib import Path

import polars as pl

from transcription import Transcription, Shownotes

data_path = Path('/mnt/d/Projects/podcast-shownotes')

timestamp_regexp = re.compile('\[(?P<ts_hour>\d\d):(?P<ts_min>\d\d):?(?P<ts_sec>\d\d)?\]\s+(?P<title>.*)$')


def timestamp_to_seconds(hour: str, minutes: str, seconds: str|None) -> int:
    if seconds is None: seconds = 0
    return 60 * 60 * int(hour) + 60 * int(minutes) + int(seconds)


def get_shownotes_with_timestamps(shownotes: str) -> list[tuple]:
    timestamps = [(timestamp_to_seconds(x.group('ts_hour'), x.group('ts_min'), x.group('ts_sec')), x.group('title')) for x in (timestamp_regexp.search(sn) for sn in shownotes.split('\n')) if x is not None]
    # some episodes have multiple topics related to the same timestamp.
    grouped_timestamps = []
    for k, group in groupby(timestamps, key=lambda x: x[0]):
        grouped_timestamps.append((k, '. '.join([x[1] for x in list(group)])))
    return grouped_timestamps


def get_topic_texts(transcription: Transcription, shownotes: list[Shownotes]) -> list[tuple[float, str]]:
    topics = []
    current_topic_text = ''
    current_topic_index = 0
    next_topic_index = 1
    current_topic_end = shownotes[next_topic_index].timestamp
    for segment in transcription.segments:
        current_topic_text += segment.text
        if segment.end >= current_topic_end:
            topics.append((segment.end, current_topic_text))
            next_topic_index += 1
            if next_topic_index >= len(shownotes):
                break

            current_topic_index += 1
            current_topic_text = ''
            current_topic_end = shownotes[next_topic_index].timestamp

    topics.append((shownotes[current_topic_index].timestamp, current_topic_text))

    return topics
