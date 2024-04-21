from __future__ import annotations

import asyncio

from asyncio.subprocess import Process
from enum import Enum
from logging import getLogger
import signal
import sys
from pathlib import Path


logger = getLogger('watcher_daemon')


class DaemonStatus(Enum):
    STARTED = 'started'
    STOPPED = 'stopped'
    HEALTHY = 'healthy'
    DEAD = 'dead'


class DaemonWrapper:
    _process: Process = None

    def __init__(
            self,
            *,
            module_name: str,
            args: list[str],
            pidfile: Path
    ):
        self._module_name = module_name
        self._args = args
        self._pidfile = pidfile

    async def start(self) -> None:
        logger.info('Starting daemon')

        self._process = await asyncio.create_subprocess_exec(
            sys.executable,
            '-m',
            self._module_name,
            *self._args
        )
        await self._wait_daemon_startup()
        logger.info('Daemon started')

    async def _wait_daemon_startup(self):
        check_interval = 0.1
        timeout = 10
        wait_counter = 0.0
        pidfile = self._pidfile

        while not pidfile.exists() or pidfile.read_text() != str(self._process.pid):
            print(f'{pidfile.exists()=}\t{self._process.pid=}')
            if self._process.returncode is not None:
                raise ChildProcessError(f'Daemon process has died: {self._process.returncode}')
            if wait_counter >= timeout:
                raise TimeoutError(f'Daemon startup timeout: {timeout}s')
            wait_counter += check_interval
            await asyncio.sleep(check_interval)

    async def shutdown(self) -> None:
        logger.info('Stopping daemon')
        self._process.send_signal(signal.SIGINT)
        while self.status is DaemonStatus.HEALTHY:
            await asyncio.sleep(0.1)
        logger.info('Daemon stopped')

    @property
    def status(self) -> DaemonStatus:
        if self._process is None:
            return DaemonStatus.STOPPED
        if self._process.returncode is None:
            return DaemonStatus.HEALTHY
        elif self._process.returncode in (0, -2):
            return DaemonStatus.STOPPED
        else:
            return DaemonStatus.DEAD


__all__ = ['DaemonWrapper', 'DaemonStatus']
