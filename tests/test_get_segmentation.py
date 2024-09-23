from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

import json
from src.components.segmentation.semantic_text_segmentation import SemanticTextSegmentationMultilingual
from src.components.segmentation.sentences import Sentence
from src.components.segmentation.segmentation_builder import (
    SegmentationBuilder,
    SegmentationResult,
    Segment,
    SegmentSentence
)
from stanza import Pipeline


pipeline = Pipeline('ru', processors=['tokenize'])


def test_get_segments():
    p = 'transcription-deepgram.json'
    with open(p, 'r') as f:
        data = json.load(f)
    transcription_data = data['results']['channels'][0]['alternatives'][0]
    segments = []
    for p in transcription_data['paragraphs']['paragraphs']:
        for s in p['sentences']:
            segments.append({
                'start': s['start'],
                'end': s['end'],
                'text': s['text'],
                'speaker': p['speaker']
            })
    segmenter = SemanticTextSegmentationMultilingual(
        [Sentence(**x) for x in segments],
        model='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

    segments = segmenter.get_segments(0.8)
    assert len(segments) == 9
    assert isinstance(segments[0], str)


def test_get_segments_as_sentences():
    p = 'transcription-deepgram.json'
    with open(p, 'r') as f:
        data = json.load(f)
    transcription_data = data['results']['channels'][0]['alternatives'][0]
    segments = []
    for p in transcription_data['paragraphs']['paragraphs']:
        for s in p['sentences']:
            segments.append({
                'start': s['start'],
                'end': s['end'],
                'text': s['text'],
                'speaker': p['speaker']
            })
    segmenter = SemanticTextSegmentationMultilingual(
        [Sentence(**x) for x in segments],
        model='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

    segments = segmenter.get_segments(0.8, as_sentences=True)

    assert len(segments) == 9
    assert len(segments[0]) == 69
    assert isinstance(segments[0][0], Sentence)


def test_get_segmentation_result():
    segmentation_builder = SegmentationBuilder(Path('data'))
    item_path = Path(__file__).resolve().parent / '472/transcription-deepgram.json'
    result = segmentation_builder.get_segments(item_path)

    assert isinstance(result, SegmentationResult)
    assert result.item == item_path
    assert len(result.segments) == 9
    assert len(result.sentences_by_segment) == 9

    for i, s in enumerate(result.segments):
        assert s.episode == 472
        assert isinstance(s, Segment)
        assert all([x.segment_num == s.num for x in result.sentences_by_segment[i]])

    for s in result.sentences_by_segment:
        assert all([isinstance(x, SegmentSentence) for x in s])
