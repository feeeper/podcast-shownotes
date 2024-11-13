import datetime
import logging
import typing
from dataclasses import dataclass, field
import dataclasses
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
import psycopg2
import psycopg2.extras
from pgvector.psycopg2 import register_vector
import numpy as np


@dataclass
class ShutdownState:
    is_shutdown_requested: bool = field(init=False, default=False)


@dataclass
class SearchResult:
    episode: int
    sentence: str
    segment: str
    distance: float
    starts_at: float
    ends_at: float


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
        conn = psycopg2.connect(
            host=db_connection_args.host,
            port=db_connection_args.port,
            dbname=db_connection_args.dbname,
            user=db_connection_args.user,
            password=db_connection_args.password
        )
        register_vector(conn)

        shutdown_state = ShutdownState()
        try:
            loop.run_until_complete(
                _run_server(
                    shutdown_state=shutdown_state,
                    daemon=daemon_wrapper,
                    transcribe_daemon=transcribe_daemon_wrapper,
                    segmentation_daemon=segmentation_daemon_wrapper,
                    connection=conn,
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
        connection
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
        jdata = await request.json()
        text = jdata['query']
        cursor = connection.cursor(cursor_factory=psycopg2.extras.DictCursor)
        embedding = embedding_builder.get_embeddings(text)
        cursor.execute(
            f'''SELECT 
                	e.episode_number as episode,
                	s.text as sentence,
                	s.start_at as starts_at,
                	s.end_at as ends_at,
                	seg.text as segment,
                	s.sentence_embedding <=> %s as distance
                FROM sentences s
                left join segments seg on s.segment_id = seg.id
                left join episodes e on e.id = seg.episode_id 
                ORDER by
                	s.sentence_embedding <=> %s
                LIMIT 5''',
            (np.array(embedding.tolist()), np.array(embedding.tolist()),)
        )
        records = cursor.fetchall()
        results = [SearchResult(**x) for x in records]
        return web.json_response(data=[dataclasses.asdict(r) for r in results])

    app = web.Application(logger=logger)
    app.add_routes(routes)
    app.cleanup_ctx.append(_daemon_context(daemon))
    app.cleanup_ctx.append(_daemon_context(transcribe_daemon))
    app.cleanup_ctx.append(_daemon_context(segmentation_daemon))

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
