from pathlib import Path
from typing import Any


class TranscriberBase:
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
            self.client = self.get_client(api_key=api_key)

    def get_client(self, api_key: str) -> Any:  # todo: extract class
        pass

    def transcribe(self, item: Path) -> None:
        pass

    def pick_episodes(self) -> list[Path]:
        def _transcription_exists(directory_: Path) -> bool:
            for file in directory_.iterdir():
                if file.name.startswith('transcription-'):
                    return True
            return False

        def _in_progress(directory_: Path) -> bool:
            for file in directory_.iterdir():
                if file.name == 'in_progress':
                    return True
            return False

        to_process: list[Path] = []
        for directory in self._storage_dir.iterdir():
            if not directory.is_dir():
                continue

            if not (directory / 'episode.mp3').exists():
                continue

            if _transcription_exists(directory) or _in_progress(directory):
                continue

            to_process.append(directory)

        return to_process
