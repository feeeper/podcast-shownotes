from __future__ import annotations

import logging
from pathlib import Path
from datetime import datetime, timedelta, date
from io import TextIOWrapper


class FileHandler(logging.FileHandler):
    def __init__(
        self, log_dir: Path, name_pattern: str, log_rotation_num_days: int
    ):
        assert log_rotation_num_days >= 0
        self._log_dir = log_dir
        self._name_pattern = name_pattern
        self._log_rotation_num_days = log_rotation_num_days
        self._target_file = self._get_target_file()
        super(FileHandler, self).__init__(
            str(self._target_file), encoding='utf-8'
        )

    def emit(self, record: logging.LogRecord) -> None:
        self._maybe_handle_date_change()
        super().emit(record)

    def _open(self) -> TextIOWrapper:
        self._log_dir.mkdir(parents=True, exist_ok=True)
        # noinspection PyProtectedMember
        return super()._open()

    def _maybe_handle_date_change(self) -> None:
        target_file = self._get_target_file()
        if target_file != self._target_file:
            self._target_file = target_file
            self.baseFilename = str(self._target_file)
            self.setStream(self._open())
            self._drop_old_logs()

    def _get_target_file(self) -> Path:
        date_str = self._today().isoformat()
        return self._log_dir / self._name_pattern.replace('{date}', date_str)

    def _drop_old_logs(self) -> None:
        if '{date}' not in self._name_pattern:
            return

        min_date = self._today() - timedelta(days=self._log_rotation_num_days)
        prefix_len = self._name_pattern.index('{date}')
        glob_pattern = '????-??-??'
        for entry in self._log_dir.glob(
            self._name_pattern.replace('{date}', glob_pattern)
        ):
            if not entry.is_file():
                continue
            date_str = entry.stem[prefix_len : prefix_len + len(glob_pattern)]
            try:
                date_proper = date.fromisoformat(date_str)
            except ValueError:
                continue
            if date_proper < min_date:
                entry.unlink()

    def _today(self) -> date:
        return datetime.today().date()


__all__ = ['FileHandler']
