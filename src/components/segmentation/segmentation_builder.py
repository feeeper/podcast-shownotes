from __future__ import annotations
import dataclasses
from logging import getLogger
from pathlib import Path
from .sentences import get_sentences
from .semantic_text_segmentation import SemanticTextSegmentationMultilingual
from .sentences import Sentence


logger = getLogger('segmentation_builder')


@dataclasses.dataclass
class Segment:
    text: str  # plain text
    start_at: float  # timestamp
    end_at: float  # timestamp
    episode: int  # episode that contains this segment
    num: int  # number of the segment in the episode


@dataclasses.dataclass
class SegmentSentence:
    text: str  # plain text
    start_at: float  # timestamp
    end_at: float  # timestamp
    speaker_id: int | None  # speaker id (could differ for different episodes) if exists
    segment_num: int  # number of the segment in the episode
    num: int  # number of the sentence in the segment


class SegmentationResult:
    item: Path
    segments: list[Segment]
    sentences_by_segment: list[list[SegmentSentence]]

    def __init__(
            self,
            item: Path,
            segments: list[Segment],
            sentences_by_segment: list[list[SegmentSentence]]
    ) -> None:
        self.item = item
        self.segments = segments
        self.sentences_by_segment = sentences_by_segment


class SegmentationBuilder:
    def __init__(self, storage_dir: Path) -> None:
        self._storage_dir = storage_dir

    def pick_episodes(self) -> list[Path]:
        def _is_transcription_exists(directory_: Path) -> bool:
            for file in directory_.iterdir():
                if file.name.startswith('transcription-'):
                    logger.debug(f'Transcription exists: {directory_}')
                    return True
            return False

        def _is_segmentation_completed(directory_: Path) -> bool:
            for file in directory_.iterdir():
                if file.name.startswith('segmentation_completed'):
                    logger.debug(f'Segmentation exists: {directory_}')
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

    def get_segments(self, item: Path) -> SegmentationResult:
        transcript_file = self._get_transcript_file(item)
        sentences = get_sentences(transcript_file)
        _segmentation = SemanticTextSegmentationMultilingual(sentences)
        segments_sentences: list[list[Sentence]] = _segmentation.get_segments(threshold=0.8, as_sentences=True)
        segments = [Segment(
            text=' '.join([s.text for s in ss]),
            start_at=ss[0].start,
            end_at=ss[-1].end,
            episode=int(item.name),
            num=n
        ) for (n, ss) in enumerate(segments_sentences)]

        sentences_by_segment = [[SegmentSentence(
            text=s.text,
            start_at=s.start,
            end_at=s.end,
            speaker_id=s.speaker,
            segment_num=n,
            num=-1
        ) for s in ss] for (n, ss) in enumerate(segments_sentences)]

        i = 0
        for n, ss in enumerate(sentences_by_segment):
            for sn, s in enumerate(ss):
                s.num = i
                i += 1

        result = SegmentationResult(
            item=item,
            segments=segments,
            sentences_by_segment=sentences_by_segment)

        return result

    @staticmethod
    def _get_transcript_file(item: Path) -> Path:
        for file in item.iterdir():
            if file.name.startswith('transcription-'):
                return file
        raise FileNotFoundError(f'Transcription file not found: {item}')