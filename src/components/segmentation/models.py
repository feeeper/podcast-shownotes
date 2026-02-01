import datetime
from pydantic import BaseModel
from uuid import UUID


class BaseModelWithConfig(BaseModel):
    class Config:
        json_encoders = {
            UUID: str,
            datetime.datetime: lambda dt: dt.isoformat(),
        }


class EpisodeDto(BaseModelWithConfig):
    id: UUID
    podcast_slug: str
    num: int | None = None
    title: str
    shownotes: str
    hosts: list[str]
    release_date: datetime.datetime
    link: str


class SearchResultDto(BaseModelWithConfig):
    episode: int | None = None
    podcast_slug: str = ''
    sentence: str
    segment: str
    distance: float
    starts_at: float
    ends_at: float


class SearchResultComplexDto(BaseModelWithConfig):
    episode: EpisodeDto
    sentence: str
    segment: str
    distance: float
    starts_at: float
    ends_at: float


class SearchResult(SearchResultDto):
    id: UUID


class SearchResults(BaseModelWithConfig):
    results: list[SearchResultDto]


class Episode(BaseModelWithConfig):
    id: UUID
    podcast_slug: str
    num: int | None = None
    title: str
    shownotes: str
    hosts: list[str]
    release_date: datetime.datetime
    link: str
