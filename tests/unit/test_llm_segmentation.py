import pytest

from src.components.segmentation.llm_segmetation import LlmTextSegmentation


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
        base_url='base_url'
    )
    segments = segmentation.get_segments(sentences, delimiters)
    assert segments == expected, f'Expected: {expected}, got: {segments} for delimiters: {delimiters}'
