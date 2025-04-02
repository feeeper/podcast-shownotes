import datetime
from pydantic import BaseModel, computed_field
from uuid import UUID


class BaseModelWithConfig(BaseModel):
    class Config:
        json_encoders = {
            UUID: str,
            datetime.datetime: lambda dt: dt.isoformat()
        }


class EpisodeDto(BaseModelWithConfig):
    id: UUID
    num: int
    title: str
    shownotes: str
    hosts: list[dict[str, str]]
    release_date: datetime.datetime

    @computed_field
    @property
    def link(self) -> str:
        if self.num <= 406:
            return f'https://devzen.ru/episode-{self.num:0>4}'
        else:
            return f'https://devzen.ru/episode-{self.num}'


class SearchResultDto(BaseModelWithConfig):
    episode: int
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
    num: int
    title: str
    shownotes: str
    hosts: list[dict[str, str]]
    release_date: datetime.datetime
    
    @computed_field
    @property
    def link(self) -> str:
        if self.num <= 406:
            return f'https://devzen.ru/episode-{self.num:0>4}'
        else:
            return f'https://devzen.ru/episode-{self.num}'
