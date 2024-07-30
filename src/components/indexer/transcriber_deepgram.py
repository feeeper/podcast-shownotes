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
            debug: bool = False,
            verbose: int = 0,
    ) -> None:
        self.verbose = verbose
        if debug:
            self.verbose = 2

        if self.verbose:
            print('TranscriberDeepgram: __init__')

        super().__init__(storage_dir, api_key, debug)

    def get_client(self, api_key: str) -> Any:
        if self.verbose:
            print('TranscriberDeepgram: get_client')
        client = DeepgramClient(api_key)
        return client

    def transcribe(self, item: Path) -> None:
        if self.debug:
            print(f'Transcribing: {item}')
            time.sleep(3)
        else:
            try:
                if self.verbose:
                    print(f'Transcribing: {item}')
                options = PrerecordedOptions(
                    model="nova-2",
                    language="ru",
                    paragraphs=True,
                    diarize=True,
                )

                if self.verbose:
                    print(f'Processing: {item / "episode.mp3"}')
                with open(item / 'episode.mp3', 'rb') as file:
                    if self.verbose:
                        print(f'Reading: {item / "episode.mp3"}')
                    buffer_data = file.read()
                    if self.verbose:
                        print(f'Size: {len(buffer_data)} bytes')

                payload: FileSource = {
                    'buffer': buffer_data,
                }

                # mark as processing episode
                if self.verbose:
                    print(f'Marking as in_progress: {item}')
                Path(item / 'in_progress').touch()

                if self.verbose:
                    print(f'Sending request to Deepgram')
                response: PrerecordedResponse = self.client.listen.prerecorded.v("1").transcribe_file(
                    source=payload,
                    options=options
                )

                if self.verbose:
                    print(f'Saving response to: {item / "transcription-deepgram.json"}')
                with open(item / 'transcription-deepgram.json', 'w', encoding='utf-8') as transcription_file:  # noqa E501
                    transcription_file.write(response.to_json(ensure_ascii=False))

            except Exception as e:
                print(f'Exception: {e}', e)
                Path(item / 'in_progress').unlink()
