import pandas as pd
import json

from src.components.segmentation.semantic_text_segmentation import SemanticTextSegmentationMultilingual

from stanza import Pipeline

pipeline = Pipeline('ru', processors=['tokenize'])


def test_basic():
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
    data = pd.DataFrame(segments)
    segmenter = SemanticTextSegmentationMultilingual(
        data,
        'text',
        model='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

    segments = segmenter.get_segments(0.8)
    assert len(segments) == 9
