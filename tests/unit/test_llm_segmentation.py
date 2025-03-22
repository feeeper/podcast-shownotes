import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import json
import pytest
from unittest.mock import Mock, patch
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.completion_usage import CompletionUsage


from src.components.segmentation.llm_segmetation import LlmTextSegmentation
from src.components.segmentation.sentences import Sentence


def get_mock_openai_response():
    content = open('tests/unit/assets/llm_response.json').read()
    response = ChatCompletion(
        id="mock-id",
        model="mock-model",
        object="chat.completion",
        created=1234567890,
        choices=[
            Choice(
                finish_reason="stop",
                index=0,
                message=ChatCompletionMessage(
                    content=content,
                    role="assistant"
                )
            )
        ],
        usage=CompletionUsage(
            completion_tokens=10,
            prompt_tokens=20,
            total_tokens=30
        )
    )
    return response

@pytest.mark.parametrize(
        'delimiters,expected', 
        [
            ([0, 1], [['a'], ['b', 'c', 'd']]),
            ([0, 3], [['a','b', 'c'], ['d']]),
            ([0], [['a','b', 'c', 'd']])
        ])
def test_get_segments(
    delimiters,
    expected
):
    sentences=['a', 'b', 'c', 'd']
    segmentation = LlmTextSegmentation(
        sentences=sentences,
        api_key='api_key',
        base_url='https://base_url'
    )
    segments = segmentation._get_segments(sentences, delimiters)
    assert segments == expected, f'Expected: {expected}, got: {segments} for delimiters: {delimiters}'


def test_llm_call():
    mock_openai_response = get_mock_openai_response()
    mock_client = Mock()
    mock_client.chat.completions.create.return_value = mock_openai_response

    sentences = [Sentence(**x) for x in json.load(open('tests/unit/assets/sentences.json'))]
    segmentation = LlmTextSegmentation(
        sentences=sentences,
        api_key='mock_api_key',
        base_url='https://mock_base_url'
    )

    with patch.object(segmentation, '_client', mock_client):
        segments = segmentation.get_segments(as_sentences=True)
        expected_segments = json.load(open('tests/unit/assets/401-segments.json'))
        expected_segments = [[Sentence(**y) for y in x] for x in expected_segments]
        assert len(segments) == 54
        assert segments == expected_segments
