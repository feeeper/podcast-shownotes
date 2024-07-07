import time
from pathlib import Path
import openai


class Transcriber:
    def __init__(self,
                 storage_dir: Path,
                 api_key: str = None,
                 debug: bool = False
    ) -> None:
        self.debug = debug
        self._storage_dir = storage_dir

        if self.debug:
            ...
        else:
            self.client = openai.OpenAI(api_key=api_key)

    def transcribe(self, item: Path) -> None:
        if self.debug:
            print(f'Transcribing: {item}')
            time.sleep(3)
        else:
            with open(item / 'episode.mp3', 'rb') as file:
                response = self.client.audio.transcriptions.create(
                    file=file,
                    model='whisper-1',
                    response_format='verbose_json',
                    language='ru',
                )

                with open(item / 'transcription-openai.json', 'w', encoding='utf-8') as transcription_file:  # noqa E501
                    transcription_file.write(response.json())

    def pick_episodes(self) -> list[Path]:
        def _transcription_exists(directory_: Path) -> bool:
            for file in directory_.iterdir():
                if file.name.startswith('transcription-'):
                    return True
            return False

        to_process: list[Path] = []
        for directory in self._storage_dir.iterdir():
            if not directory.is_dir():
                continue

            if not (directory / 'episode.mp3').exists():
                continue

            if _transcription_exists(directory):
                continue

            to_process.append(directory)

        return to_process
