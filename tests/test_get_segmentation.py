import json

from src.components.segmentation.semantic_text_segmentation import SemanticTextSegmentationMultilingual
from src.components.segmentation.sentences import Sentence

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
