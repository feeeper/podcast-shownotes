import time
from pathlib import Path
from typing import Any

from deepgram import (
    DeepgramClient,
    PrerecordedOptions,
    FileSource,
    PrerecordedResponse,
)

from .transcriber_base import TranscriberBase


class TranscriberDeepgram(TranscriberBase):
    def __init__(
            self,
            storage_dir: Path,
            api_key: str = None,
            debug: bool = False
    ) -> None:
        super().__init__(storage_dir, api_key, debug)

    def get_client(self, api_key: str) -> Any:
        client = DeepgramClient(api_key)
        return client

    def transcribe(self, item: Path) -> None:
        if self.debug:
            print(f'Transcribing: {item}')
            time.sleep(3)
        else:
            try:
                options = PrerecordedOptions(
                    model="nova-2",
                    language="ru",
                    paragraphs=True,
                    diarize=True,
                )

                with open(item / 'episode.mp3', 'rb') as file:
                    buffer_data = file.read()

                payload: FileSource = {
                    'buffer': buffer_data,
                }

                # mark as processing episode
                Path(item / 'in_progress').touch()

                response: PrerecordedResponse = self.client.listen.prerecorded.v("1").transcribe_file(
                    source=payload,
                    options=options
                )

                with open(item / 'transcription-deepgram.json', 'w', encoding='utf-8') as transcription_file:  # noqa E501
                    transcription_file.write(response.to_json(ensure_ascii=False))

            except Exception as e:
                print(f'Exception: {e}')
