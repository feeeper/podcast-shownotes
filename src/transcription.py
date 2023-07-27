class Segment:
    def __init__(self,
                 id: int,
                 seek: int,
                 start: float,
                 end: float,
                 text: str,
                 tokens: list[int],
                 temperature: float,
                 avg_logprob: float,
                 compression_ratio: float,
                 no_speech_prob: float):
        self.id = id
        self.seek = int(seek)
        self.start = float(start)
        self.end = float(end)
        self.text = text
        self.tokens = list(map(int, tokens))
        self.temperature = float(temperature)
        self.avg_logprob = float(avg_logprob)
        self.compression_ratio = float(compression_ratio)
        self.no_speech_prob = float(no_speech_prob)

    def __str__(self):
        return f'Segment({self.id=}, {self.start=}, {self.end=}, {self.text=})'


class Transcription:
    def __init__(self, text: str, segments: list[Segment], language: str):
        self.text = text
        self.segments = segments
        self.language = language


class Shownotes:
    def __init__(self, timestamp: float, title: str):
        self.timestamp = float(timestamp)
        self.title = title

    def __str__(self):
        return f'Shownotes({self.timestamp=}, "{self.title=}")'

    def __repr__(self):
        return str(self)
