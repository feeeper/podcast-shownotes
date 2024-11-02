from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pytest
from src.components.segmentation.segmentation_builder import SegmentationResult, Segment, SegmentSentence
from src.components.segmentation.segmentation_repository import SegmentationRepository


@pytest.skip('requires database connection')
def test_insert_segmentation():
    segmentation_repository = SegmentationRepository()
    item_path = Path(__file__).resolve().parent / '472'
    segmentation = SegmentationResult(
        item_path,
        [
            Segment(
                text='This is a test',
                start_at=0.0,
                end_at=1.0,
                episode=1,
                num=1
            )
        ],
        [
            [
                SegmentSentence(
                    text='This is a test',
                    start_at=0.0,
                    end_at=1.0,
                    speaker_id=None,
                    segment_num=1,
                    num=1
                )
            ]
        ]
    )
    result = segmentation_repository.save(segmentation)
    assert result is not None

    delete_result = segmentation_repository.delete(result)
    assert delete_result
