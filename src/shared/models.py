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
                 no_speech_prob: float,
                 **kwargs):
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


class DeepgramSegment:
    def __init__(
            self,
            start: float,
            end: float,
            text: str,
            speaker: int
    ) -> None:
        self.start = start
        self.end = end
        self.text = text
        self.speaker = speaker

    def __str__(self):
        return f'DeepgramSegment(start={self.start}, end={self.end}, text="{self.text}", speaker={self.speaker})'

    def __repr__(self):
        return str(self)


class DeepgramTranscription:
    def __init__(
            self,
            text: str,
            segments: list[DeepgramSegment]
    ):
        self.text = text
        self.segments = segments

    def __str__(self):
        return f'DeepgramTranscript(text="{self.text}", segments={self.segments})'

    def __repr__(self):
        return str(self)
