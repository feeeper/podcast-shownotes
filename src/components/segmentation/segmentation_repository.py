from __future__ import annotations

from src.components.segmentation.segmentation_builder import SegmentationResult

from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from pymilvus.exceptions import SchemaNotReadyException


class SegmentationRepository:
    def __init__(self):
        ...

    def save(self, segmentation_result: SegmentationResult) -> None:
        ...

    def find(self, item: str) -> SegmentationResult:
        ...