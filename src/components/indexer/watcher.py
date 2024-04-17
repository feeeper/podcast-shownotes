import datetime
import logging

from aiohttp import web

from shared.args import LoggingArgs
from infrastructure.logging.setup import setup_logging

logger = logging.getLogger('watcher')


def main():
    try:
        logging_args = LoggingArgs.parse(description='Watcher')
        print(logging_args)
        setup_logging(logging_args)
        _run_server()
    finally:
        logger.info('Server stopped')


def _run_server():
    routes = web.RouteTableDef()

    @routes.get('/ping')
    async def get_ping(request: web.Request) -> web.Response:
        return web.json_response(data={
            'pong': True,
            'utcnow': datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        })

    app = web.Application(logger=logger)
    app.add_routes(routes)

    logger.info('Server started')
    web.run_app(app, host='localhost', port=8080)


if __name__ == '__main__':
    main()
