from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.components.segmentation.sentences import get_sentences


def test_get_deepgram_sentences():
    sentences = get_sentences(Path('transcription-deepgram.json'))
    assert len(sentences) == 754


def test_get_non_deepgram_sentences():
    sentences = get_sentences(Path('episode-0122.mp3-large.json'))
    assert len(sentences) == 1259
