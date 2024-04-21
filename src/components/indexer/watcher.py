import datetime
import logging
import typing
from dataclasses import dataclass, field
from pathlib import Path

from aiohttp import web
import asyncio

from shared.args import IndexerServerArgs, DaemonArgs
from infrastructure.logging.setup import setup_logging
from shared.daemon_wrapper import DaemonWrapper


@dataclass
class ShutdownState:
    is_shutdown_requested: bool = field(init=False, default=False)


logger = logging.getLogger('watcher')


def main():
    try:
        index_server_args = IndexerServerArgs.parse(description='Watcher')
        logging_args = index_server_args.logging
        setup_logging(logging_args)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        daemon_args = DaemonArgs(logging=logging_args, storage=index_server_args.storage)
        daemon_wrapper = DaemonWrapper(
            module_name='src.components.indexer.daemon',
            args=daemon_args.forward(),
            pidfile=Path('./daemon')
        )
        shutdown_state = ShutdownState()
        try:
            loop.run_until_complete(
                _run_server(
                    shutdown_state=shutdown_state,
                    daemon=daemon_wrapper
                )
            )
        except asyncio.CancelledError:
            pass
    finally:
        logger.info('Server stopped')


@asyncio.coroutine
def watch() -> None:
    while True:
        print(f'periodic: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        yield from asyncio.sleep(1)


async def _run_server(
        *,
        shutdown_state: ShutdownState,
        daemon: DaemonWrapper
) -> None:
    routes = web.RouteTableDef()

    @routes.get('/ping')
    async def get_ping(request: web.Request) -> web.Response:
        return web.json_response(data={
            'pong': True,
            'utcnow': datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        })

    @routes.post('/shutdown')
    async def handle_shutdown(_) -> web.Response:
        await daemon.shutdown()
        shutdown_state.is_shutdown_requested = True
        return web.Response(status=200)

    app = web.Application(logger=logger)
    app.add_routes(routes)
    app.cleanup_ctx.append(_daemon_context(daemon))

    runner = web.AppRunner(
        app,
        access_log_format='%a %t "%r" %s %b "%{Referer}i" "%{User-Agent}i"'
    )
    await runner.setup()
    site = web.TCPSite(runner, 'localhost', 8080)
    await site.start()
    logger.info(f'Listening {site.name}')

    while not shutdown_state.is_shutdown_requested:
        await asyncio.sleep(0.1)

    await runner.cleanup()


def _daemon_context(
        daemon: DaemonWrapper
) -> typing.Callable[[web.Application], typing.AsyncIterator[None]]:
    async def _context(_: web.Application):
        await daemon.start()
        yield
        await daemon.shutdown()

    return _context


if __name__ == '__main__':
    main()
