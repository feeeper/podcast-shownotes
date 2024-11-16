from pathlib import Path
import json

import stanza

from shared.shownotes import (
    get_sentences_with_timestamps_by_letter,
    get_deepgram_sentences,
)
from shared.models import (
    Segment,
    Transcription,
    DeepgramTranscription,
    DeepgramSegment
)


class Sentence:
    def __init__(self, text: str, start: float, end: float, speaker: int = None):
        self.text = text
        self.start = start
        self.end = end
        self.speaker = speaker

    def __str__(self):
        return f'{self.text} ({self.start} - {self.end})'

    def __repr__(self):
        return str(self)


def get_sentences(path_to_transcript: Path, verbose: bool = False) -> list[Sentence]:
    if 'deepgram' in str(path_to_transcript).lower():
        with open(path_to_transcript, 'r') as f:
            data = json.load(f)

        transcription_data = data['results']['channels'][0]['alternatives'][0]
        segments = []
        for p in transcription_data['paragraphs']['paragraphs']:
            for s in p['sentences']:
                segments.append(DeepgramSegment(
                    start=s['start'],
                    end=s['end'],
                    text=s['text'],
                    speaker=p['speaker']
                ))

        transcript = DeepgramTranscription(
            text=transcription_data['transcript'],
            segments=segments
        )
        if verbose:
            print(f'[Deepgram] Total segments count = {len(transcript.segments)}')

        sentences = [Sentence(x[2], x[0], x[1], x[3]) for x in get_deepgram_sentences(transcript, verbose=verbose)]
    else:
        with open(path_to_transcript, 'r') as f:
            data = json.load(f)

        pipeline = stanza.Pipeline('ru', processors='tokenize')
        segments = [Segment(**x) for x in data['segments']]
        transcription = Transcription(data['text'], segments, data['language'])

        if verbose:
            print(f'Total segments count = {len(segments)}')

        sentences = [Sentence(x[2], x[0], x[1]) for x in get_sentences_with_timestamps_by_letter(
            transcription,
            get_sentences_callback=lambda text: [x.text for x in pipeline(text).sentences],
            verbose=verbose)]

    return sentences
