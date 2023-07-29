import sys
sys.path.append('../src')

import typing as t
import re
from itertools import groupby

from models import Transcription, Shownotes


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


def get_topic_texts(transcription: Transcription, shownotes: list[Shownotes]) -> list[Shownotes]:
    if shownotes[-1].timestamp > transcription.segments[-1].end:
        return []

    topics = []
    current_topic_text = ''
    current_topic_index = 0
    next_topic_index = 1
    current_shownote_start = shownotes[current_topic_index].timestamp
    current_shownote_end = 1_000_000 if len(shownotes) == 1 else shownotes[next_topic_index].timestamp
    current_topic_start = transcription.segments[0].start

    for segment in transcription.segments:
        current_topic_text += segment.text
        if segment.end >= current_shownote_end:
            topics.append(Shownotes(current_topic_start, current_topic_text))
            next_topic_index += 1
            current_topic_index += 1
            current_topic_text = ''
            current_topic_start = segment.end
            current_shownote_start = shownotes[current_topic_index].timestamp
            current_shownote_end = 1_000_000 if next_topic_index >= len(shownotes) else shownotes[next_topic_index].timestamp

    topics.append(Shownotes(current_shownote_start, current_topic_text))
    return topics


def get_sentences_with_timestamps(
        transcription: Transcription,
        get_sentences_callback: t.Callable[[str], list[str]]
) -> list[tuple[float, float, str]]:
    text = transcription.text
    sentences = get_sentences_callback(text)

    sentence_timestamps = []
    start_from_idx = 0
    start_from_in_segment_idx = 0

    segment_sentences_dict = {}
    for sentence in sentences:
        first_segment = transcription.segments[start_from_idx]
        sentence_timestamp = first_segment.start
        test_sentence = first_segment.text.strip()

        # segment.text contains whole sentence
        if sentence == test_sentence.strip():
            start_from_idx += 1
            sentence_timestamps.append((sentence_timestamp, first_segment.end, sentence))
            continue

        test_sentence = ''
        for i, segment in enumerate(transcription.segments[start_from_idx:]):
            break_outer_loop = False
            if segment.text not in segment_sentences_dict:
                # segment text can contain more than one sentence or
                # even one sentence and a beginning of a next sentence
                segment_sentences_dict[segment.text] = get_sentences_callback(segment.text)

            segment_sentences = segment_sentences_dict[segment.text]
            least_segment_sentences = segment_sentences[start_from_in_segment_idx:]
            for seg_sentence_idx, seg_sentence in enumerate(least_segment_sentences):
                test_sentence += ' ' + seg_sentence
                if sentence == test_sentence.strip():
                    if seg_sentence_idx == len(least_segment_sentences) - 1:
                        start_from_idx += i + 1
                        start_from_in_segment_idx = 0
                    else:
                        start_from_in_segment_idx += seg_sentence_idx + 1

                    sentence_timestamps.append((sentence_timestamp, segment.end, sentence))
                    break_outer_loop = True
                    break
            if break_outer_loop:
                break
            else:
                start_from_in_segment_idx = 0

    return sentence_timestamps
