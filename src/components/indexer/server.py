from __future__ import annotations
from logging import getLogger
from aiohttp import web
from .routes import setup_routes
from shared.shutdown import ShutdownState
from shared.server import run_server_loop

logger = getLogger(__spec__.name)  # type: ignore[name-defined]

def main() -> None:
    app = web.Application()
    routes = web.RouteTableDef()
    setup_routes(routes, service_name='Podcast Indexer')
    shutdown_state = ShutdownState()
    run_server_loop(
        app,
        routes,
        logger,
        port=8080)


if __name__ == '__main__':
    main()
