import sys
sys.path.append('../src')

import typing as t
import re
from itertools import groupby
import time

from tqdm import tqdm
import editdistance

from models import Transcription, Shownotes


timestamp_regexp = re.compile('\[(?P<ts_hour>\d\d):(?P<ts_min>\d\d):?(?P<ts_sec>\d\d)?\]\s+(?P<title>.*)$')


class bcolors:
    OKBLUE = '\033[94m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'

class Log:
    @staticmethod
    def error(message: str) -> None:
        print(bcolors.FAIL + message + bcolors.ENDC)

    @staticmethod
    def warn(message) -> None:
        print(bcolors.WARNING + message + bcolors.ENDC)

    @staticmethod
    def info(message) -> None:
        print(bcolors.OKBLUE + message + bcolors.ENDC)


def timestamp_to_seconds(hour: str, minutes: str, seconds: str|None) -> int:
    if seconds is None: seconds = 0
    return 60 * 60 * int(hour) + 60 * int(minutes) + int(seconds)


def get_shownotes_with_timestamps(shownotes: str) -> list[tuple]:
    shownotes = re.sub('\\\\n', '\\n', shownotes)
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
        get_sentences_callback: t.Callable[[str], list[str]],
        verbose: int = 0
) -> list[tuple[float, float, str]]:
    def _replace_dots_without_space(text_to_process: str) -> str:
        return rx.sub(r'.\n\n\1', text_to_process)

    st = time.time()
    rx = re.compile('\.\.\.\s*(\w?)')
    text = _replace_dots_without_space(transcription.text)
    sentences = get_sentences_callback(text)

    if verbose:
        Log.info(f'total sentences: {len(sentences)}')
        Log.info(f'total segments: {len(transcription.segments)}')

    sentence_timestamps = []
    start_from_idx = 0
    start_from_in_segment_idx = 0

    segment_sentences_dict = {}
    for sentence in tqdm(sentences, disable=(verbose < 2)):
        sentence_wo_spaces = sentence.replace(' ', '')
        first_segment = transcription.segments[start_from_idx]
        sentence_timestamp = first_segment.start
        test_sentence =  _replace_dots_without_space(first_segment.text.strip())

        # segment.text contains whole sentence
        if sentence_wo_spaces == test_sentence.strip().replace(' ', ''):
            start_from_idx += 1
            sentence_timestamps.append((sentence_timestamp, first_segment.end, sentence))
            continue

        test_sentence = ''
        for i, segment in enumerate(transcription.segments[start_from_idx:]):
            break_outer_loop = False
            if segment.text not in segment_sentences_dict:
                # segment text can contain more than one sentence or
                # even one sentence and a beginning of a next sentence
                segment_sentences_dict[segment.text] = get_sentences_callback(_replace_dots_without_space(segment.text))

            segment_sentences = segment_sentences_dict[segment.text]
            least_segment_sentences = segment_sentences[start_from_in_segment_idx:]
            for seg_sentence_idx, seg_sentence in enumerate(least_segment_sentences):
                test_sentence += seg_sentence.replace(' ', '')
                # test_get_sentences_with_timestamps_segment_could_not_split_into_sentences_correctly
                if editdistance.distance(sentence_wo_spaces, test_sentence) <= 2:
                    if seg_sentence_idx == len(least_segment_sentences) - 1:
                        start_from_idx += i + 1
                        start_from_in_segment_idx = 0
                    else:
                        start_from_in_segment_idx += seg_sentence_idx + 1
                        start_from_idx += i

                    if len(sentence_timestamps) == 160:
                        print(160)
                    sentence_timestamps.append((sentence_timestamp, segment.end, sentence))
                    break_outer_loop = True
                    break
            if break_outer_loop:
                break
            else:
                start_from_in_segment_idx = 0

    if verbose:
        if len(sentences) == len(sentence_timestamps):
            Log.info(f'expected timestamps: {len(sentences)}\tactual timestamps: {len(sentence_timestamps)}\tprocessed in: {time.time() - st}')
        else:
            Log.warn(f'Expected timestamps: {len(sentences)}. Actual timestamps: {len(sentence_timestamps)}')
            if len(sentence_timestamps) > 0:
                Log.warn(f'Last timestamp: {sentence_timestamps[-1]}')

    return sentence_timestamps


def get_sentences_with_timestamps_by_letter(
        transcription: Transcription,
        get_sentences_callback: t.Callable[[str], list[str]],
        verbose: int = 0
) -> list[tuple[float, float, str]]:
    st = time.time()

    text = transcription.text
    sentences = get_sentences_callback(text)

    if verbose:
        Log.info(f'total sentences: {len(sentences)}')
        Log.info(f'total segments: {len(transcription.segments)}')

    sentence_timestamps = []
    start_from_idx = 0
    start_from_in_segment_idx = 0

    for sentence in tqdm(sentences, disable=(verbose < 2)):
        break_outer_loop = False
        search_sentence = sentence.replace(' ', '')
        start_sentence = transcription.segments[start_from_idx].start
        test_sentence = transcription.segments[start_from_idx].text.replace(' ', '')[start_from_in_segment_idx:]

        if search_sentence == test_sentence:
            sentence_timestamps.append((start_sentence, transcription.segments[start_from_idx].end, sentence))
            start_from_idx += 1
            start_from_in_segment_idx = 0
            continue

        # start_from_idx += 1
        test_sentence = ''
        for segment_idx, segment in enumerate(transcription.segments[start_from_idx:]):
            # test_sentence += segment.text
            segment_text = segment.text.replace(' ', '')[start_from_in_segment_idx:]
            if search_sentence == test_sentence + segment_text:
                sentence_timestamps.append((start_sentence, segment.end, sentence))
                test_sentence = ''
                start_from_in_segment_idx = 0
                start_from_idx += segment_idx + 1
                break

            if search_sentence.startswith(test_sentence + segment_text):
                test_sentence += segment_text
                start_from_in_segment_idx = 0
                continue
            else:
                for i, letter in enumerate(segment_text):
                    if search_sentence.startswith(test_sentence + letter):
                        test_sentence += letter
                    else:
                        sentence_timestamps.append((start_sentence, segment.end, sentence))
                        # test_sentence = segment.text[i:]
                        start_from_in_segment_idx += i
                        start_from_idx += segment_idx
                        break_outer_loop = True
                        break

            if break_outer_loop:
                break

    if verbose > 0:
        if len(sentences) == len(sentence_timestamps):
            Log.info(f'expected timestamps: {len(sentences)}\tactual timestamps: {len(sentence_timestamps)}\tprocessed in: {time.time() - st}')
        else:
            Log.warn(f'Expected timestamps: {len(sentences)}. Actual timestamps: {len(sentence_timestamps)}')
            if len(sentence_timestamps) > 0:
                Log.warn(f'Last timestamp: {sentence_timestamps[-1]}')

    return sentence_timestamps
