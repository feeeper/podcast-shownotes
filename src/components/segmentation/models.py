import datetime
from pydantic import BaseModel
from uuid import UUID


class SearchResult(BaseModel):
    episode: int
    sentence: str
    segment: str
    distance: float
    starts_at: float
    ends_at: float


class SearchResults(BaseModel):
    results: list[SearchResult]


class Episode(BaseModel):
    id: UUID
    num: int
    title: str
    shownotes: str
    hosts: list[dict[str, str]]
    release_date: datetime.datetime
