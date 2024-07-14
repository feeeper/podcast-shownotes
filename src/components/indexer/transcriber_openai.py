import time
from pathlib import Path
from typing import Any

import openai
from .transcriber_base import TranscriberBase


class TranscriberOpenAi(TranscriberBase):
    def __init__(
            self,
            storage_dir: Path,
            api_key: str = None,
            debug: bool = False
    ) -> None:
        super().__init__(storage_dir, api_key, debug)

    def get_client(self, api_key: str) -> Any:
        return openai.OpenAI(api_key=api_key)

    def transcribe(self, item: Path) -> None:
        if self.debug:
            print(f'Transcribing: {item}')
            time.sleep(3)
        else:
            with open(item / 'episode.mp3', 'rb') as file:
                # mark as processing episode
                Path(item / 'in_progress').touch()

                response = self.client.audio.transcriptions.create(
                    file=file,
                    model='whisper-1',
                    response_format='verbose_json',
                    language='ru',
                )

                with open(item / 'transcription-openai.json', 'w', encoding='utf-8') as transcription_file:  # noqa E501
                    transcription_file.write(response.json())
