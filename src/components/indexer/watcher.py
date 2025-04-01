import datetime
import logging
import typing
from dataclasses import dataclass, field
from pathlib import Path

from aiohttp import web
import asyncio

from shared.args import (
    IndexerServerArgs,
    DaemonArgs,
    DbConnectionArgs,
)
from infrastructure.logging.setup import setup_logging
from shared.daemon_wrapper import DaemonWrapper
from components.segmentation.embedding_builder import EmbeddingBuilder

from components.segmentation.pgvector_repository import DB


@dataclass
class ShutdownState:
    is_shutdown_requested: bool = field(init=False, default=False)


logger = logging.getLogger('watcher')
embedding_builder = EmbeddingBuilder()


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
            pidfile=Path(daemon_args.storage.directory) / 'indexer.pid'
        )

        transcribe_daemon_wrapper = DaemonWrapper(
            module_name='src.components.indexer.transcriber_daemon',
            args=index_server_args.forward(),
            pidfile=Path(daemon_args.storage.directory) / 'transcriber.pid')

        segmentation_daemon_wrapper = DaemonWrapper(
            module_name='src.components.segmentation.daemon',
            args=index_server_args.forward(),
            pidfile=Path(daemon_args.storage.directory) / 'segmentation.pid'
        )

        db_connection_args: DbConnectionArgs = index_server_args.database_connection
        repository = DB(
            host=db_connection_args.host,
            port=db_connection_args.port,
            dbname=db_connection_args.dbname,
            user=db_connection_args.user,
            password=db_connection_args.password,
        )

        shutdown_state = ShutdownState()
        try:
            loop.run_until_complete(
                _run_server(
                    shutdown_state=shutdown_state,
                    daemon=daemon_wrapper,
                    transcribe_daemon=transcribe_daemon_wrapper,
                    segmentation_daemon=segmentation_daemon_wrapper,
                    repository=repository,
                )
            )
        except asyncio.CancelledError:
            pass
    finally:
        logger.info('Server stopped')


async def _run_server(
        *,
        shutdown_state: ShutdownState,
        daemon: DaemonWrapper,
        transcribe_daemon: DaemonWrapper,
        segmentation_daemon: DaemonWrapper,
        repository: DB
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

    @routes.post('/search')
    async def handle_search(request: web.Request) -> web.Response:
        try:
            jdata = await request.json()
            text = jdata.get('query', None)
        except Exception as e:
            logger.error(e)
            return web.Response(status=400, reason='Bad Request')

        if text is None:
            return web.Response(status=400, reason='Empty query')

        limit = jdata.get('limit', 10)
        offset = jdata.get('offset', 0)
        results = repository.find_similar(text, limit=limit, offset=offset)
        return web.json_response(data=[x.model_dump() for x in results.results])
    
    @routes.get('/v2/search')
    async def handle_search_v2(request: web.Request) -> web.Response:
        try:
            text = request.query.get('query', None)
            if text is None:
                return web.Response(status=400, reason='Empty query')

            limit = int(request.query.get('limit', 10))
            offset = int(request.query.get('offset', 0))
        except Exception as e:
            logger.error(e)
            return web.Response(status=400, reason='Bad Request')
        
        if text is None:
            return web.Response(status=400, reason='Empty query')
        
        results = repository.find_similar_complex(text, limit=limit, offset=offset)
        return web.json_response(data=[x.model_dump() for x in results.results])

    @routes.get('/episodes/{episode_num}')
    async def handle_episodes(request: web.Request) -> web.Response:
        try:
            episode_num = request.match_info.get('episode_num', None)
            if episode_num is None:
                return web.Response(status=400, reason='Bad Request')
        except Exception as e:
            logger.error(e)
            return web.Response(status=400, reason='Bad Request')

        episode = repository.find_episode(episode_num)
        if episode is None:
            logger.warn(f'Episode {episode_num} not found')
            return web.json_response(status=404, reason=f'Episode {episode_num} not found')

        return web.json_response(data=episode, dumps=lambda e: e.json())

    app = web.Application(logger=logger)
    app.add_routes(routes)
    
    # Add error handling for daemon startup
    try:
        app.cleanup_ctx.append(_daemon_context(daemon))
        app.cleanup_ctx.append(_daemon_context(transcribe_daemon))
        app.cleanup_ctx.append(_daemon_context(segmentation_daemon))
    except Exception as e:
        logger.error(f"Failed to initialize daemon contexts: {str(e)}")
        raise

    runner = web.AppRunner(
        app,
        access_log_format='%a %t "%r" %s %b "%{Referer}i" "%{User-Agent}i"'
    )
    
    try:
        await runner.setup()
        site = web.TCPSite(runner, '0.0.0.0', 8080)
        await site.start()
        logger.info(f'Listening {site.name}')
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}")
        await runner.cleanup()
        raise

    while not shutdown_state.is_shutdown_requested:
        await asyncio.sleep(0.1)

    await runner.cleanup()


def _daemon_context(
        daemon: DaemonWrapper
) -> typing.Callable[[web.Application], typing.AsyncIterator[None]]:
    async def _context(app: web.Application):
        try:
            logger.info(f"Starting daemon: {daemon._module_name}")
            await daemon.start()
            logger.info(f"Daemon started successfully: {daemon._module_name}")
            yield
        except TimeoutError as e:
            logger.error(f"Timeout starting daemon {daemon._module_name}: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error in daemon {daemon._module_name}: {str(e)}")
            raise
        finally:
            try:
                await daemon.shutdown()
                logger.info(f"Daemon shut down: {daemon._module_name}")
            except Exception as e:
                logger.error(f"Error shutting down daemon {daemon._module_name}: {str(e)}")

    return _context


if __name__ == '__main__':
    main()
