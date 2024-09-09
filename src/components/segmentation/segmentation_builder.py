from __future__ import annotations

from logging import getLogger
from pathlib import Path
from .sentences import get_sentences
from .semantic_text_segmentation import SemanticTextSegmentationMultilingual


logger = getLogger('segmentation_builder')


class SegmentationResult:
    def __init__(
            self,
            item: Path,
            segments_text: list[str],
            segments: list[list[str]]
    ) -> None:
        self.item = item
        self.segment_text = segments_text
        self.segments = segments


class SegmentationBuilder:
    def __init__(self, storage_dir: Path) -> None:
        self._storage_dir = storage_dir

    def pick_episodes(self) -> list[Path]:
        def _is_transcription_exists(directory_: Path) -> bool:
            for file in directory_.iterdir():
                if file.name.startswith('transcription-'):
                    logger.info(f'Transcription exists: {directory_}')
                    return True
            return False

        def _is_segmentation_completed(directory_: Path) -> bool:
            for file in directory_.iterdir():
                if file.name.startswith('segmentation_completed'):
                    logger.info(f'Segmentation exists: {directory_}')
                    return True
            return False

        def _is_in_progress(directory_: Path) -> bool:
            for file in directory_.iterdir():
                if file.name == 'segmentation_in_progress':
                    logger.info(f'In progress: {directory_}')
                    return True
            return False

        to_process: list[Path] = []
        for directory in self._storage_dir.iterdir():
            if not directory.is_dir():
                continue

            if not _is_transcription_exists(directory):
                logger.warning(f'Transcription not found: {directory}')
                continue

            if _is_segmentation_completed(directory) or _is_in_progress(directory):
                continue

            to_process.append(directory)

        return to_process

    @staticmethod
    def get_segments(item: Path) -> list[list[str]]:
        sentences = get_sentences(item)
        _segmentation = SemanticTextSegmentationMultilingual(sentences)
        segments_sentences = _segmentation.get_segments(as_sentences=True)
        return segments_sentences
