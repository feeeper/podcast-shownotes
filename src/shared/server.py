from __future__ import annotations

import asyncio
from logging import Logger
from typing import AsyncIterator, Callable
from aiohttp import web
from aiohttp.web_app import Application

from infrastructure.logging.setup import flush_logs

print('TEST 3')

async def _run_server(
        daemon,
        routes: web.RouteTableDef,
        shutdown_state,
        logger: Logger,
        *,
        port: int,
        periodic_callback: Callable[[], None] | None = None,
) -> None:
    print('TEST 2')
    try:
        app = web.Application()
        app.add_routes(routes)

        runner = web.AppRunner(
            app,
            access_log_format='%a %t "%r" %s %b "%{Referer}i" "%{User-Agent}i"'
        )
        await runner.setup()
        site = web.TCPSite(runner, port=port)
        await site.start()
        logger.info('Listening on port %d', port)

        while not shutdown_state.shutdown:
            await asyncio.sleep(1)
            if periodic_callback is not None:
                periodic_callback()
        await runner.cleanup()
    except:
        logger.exception('Unhandled exception', exc_info=True)
        raise


def run_server_loop(
    daemon,
    routes: web.RouteTableDef,
    shutdown_state,
    logger: Logger,
    *,
    port: int,
    periodic_callback: Callable[[], None] | None = None,
    shutdown_callback: Callable[[], None] | None = None,
) -> None:
    print('TEST')
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(
            _run_server(
                daemon,
                routes,
                shutdown_state,
                logger,
                port=port,
                periodic_callback=periodic_callback,
            )
        )
    finally:
        if shutdown_callback is not None:
            shutdown_callback()
        daemon.send_interrupt_signal()
        logger.info('Server stopped')
        flush_logs()


__all__ = ['run_server_loop']
