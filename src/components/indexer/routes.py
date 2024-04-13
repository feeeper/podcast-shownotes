from aiohttp import web


def setup_routes(
        routes: web.RouteTableDef,
        *,
        service_name: str,
) -> None:
    @routes.get('/ping')
    async def hande_ping(request: web.Request) -> web.Response:
        return web.json_response(data={
            'service': service_name,
            'status': 'ok',
        })


__all__ = ['setup_routes']
