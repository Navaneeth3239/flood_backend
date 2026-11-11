import asyncio
from unittest.mock import AsyncMock

import httpx

from services import osm_tile_loader as module
from services.osm_tile_loader import OSMTileLoader, Tile


def test_overpass_timeout_payload_is_not_cached(monkeypatch):
    loader = OSMTileLoader()
    writes = []
    payloads = iter([
        {"remark": "runtime error: Query timed out", "elements": []},
        {"elements": [{"type": "way", "id": 42, "tags": {"highway": "path"}}]},
    ])

    class Client:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def post(self, endpoint, **kwargs):
            return httpx.Response(200, json=next(payloads), request=httpx.Request("POST", endpoint))

    monkeypatch.setattr(module.httpx, "AsyncClient", Client)
    monkeypatch.setattr(loader, "_write_cache", lambda dataset, tile, elements: writes.append(elements))
    result, _ = asyncio.run(loader._fetch_tile_from_endpoints("roads", Tile(1, 0, 1, 0), "query"))
    assert result[0]["id"] == 42
    assert writes == [result]


def test_tile_merge_preserves_tags_from_full_way(monkeypatch):
    loader = OSMTileLoader()
    tiles = [Tile(1, 0, 1, 0), Tile(1, 0, 2, 1)]
    monkeypatch.setattr(loader, "tiles_for_bbox", lambda *args: tiles)
    monkeypatch.setattr(loader, "_request_tile", AsyncMock(side_effect=[
        ([{"type": "way", "id": 42, "nodes": [1, 2], "tags": {"waterway": "river"}}], False),
        ([{"type": "way", "id": 42, "nodes": [1, 2]}], False),
    ]))
    elements, status = asyncio.run(loader.load("waterways", 1, 0, 2, 0, 'way["waterway"]{bbox}'))
    assert len(elements) == 1
    assert elements[0]["tags"]["waterway"] == "river"
    assert status["complete"]


def test_only_buildings_receive_longer_query_deadline(monkeypatch):
    loader = OSMTileLoader()
    request = AsyncMock(return_value=([], False))
    monkeypatch.setattr(loader, "_request_tile", request)
    monkeypatch.setattr(loader, "tiles_for_bbox", lambda *args: [Tile(1, 0, 1, 0)])
    for dataset, timeout in [("buildings", 60), ("roads", 20), ("waterways", 20)]:
        asyncio.run(loader.load(dataset, 1, 0, 1, 0, 'way["building"]{bbox}'))
        assert f"[timeout:{timeout}]" in request.call_args.args[2]


def test_small_area_uses_real_map_fallback_and_caches_it(monkeypatch):
    loader = OSMTileLoader()
    elements = [{"type":"node","id":1,"lat":10,"lon":77}]
    fallback = AsyncMock(return_value=elements)
    monkeypatch.setattr(loader, '_request_map_tile', fallback)
    writes=[]
    monkeypatch.setattr(loader, '_write_cache', lambda dataset,tile,data: writes.append((dataset,data)))
    class Client:
        def __init__(self, **kwargs): pass
        async def __aenter__(self): return self
        async def __aexit__(self,*args): pass
        async def post(self,endpoint,**kwargs):
            raise httpx.ConnectError('offline')
    monkeypatch.setattr(module.httpx,'AsyncClient',Client)
    result,_=asyncio.run(loader._fetch_tile_from_endpoints('roads',Tile(10.01,10,77.01,77),'query'))
    assert result==elements
    fallback.assert_awaited_once()
    assert writes==[('roads',elements)]


def test_map_fallback_is_shared_across_concurrent_layers(monkeypatch):
    loader=OSMTileLoader()
    monkeypatch.setattr(loader,'_read_cache',lambda *args:None)
    fetch=AsyncMock(return_value=[])
    monkeypatch.setattr(loader,'_fetch_map_tile',fetch)
    async def run():
        tile=Tile(10.01,10,77.01,77)
        return await asyncio.gather(*(loader._request_map_tile(tile) for _ in range(3)))
    assert asyncio.run(run())==[[],[],[]]
    fetch.assert_awaited_once()


def test_map_fallback_completes_relation_geometry(monkeypatch):
    loader=OSMTileLoader()
    writes=[]
    monkeypatch.setattr(loader,'_write_cache',lambda *args:writes.append(args))
    class Client:
        def __init__(self,**kwargs): pass
        async def __aenter__(self): return self
        async def __aexit__(self,*args): pass
        async def get(self,url,**kwargs):
            elements = [{'type':'relation','id':8,'tags':{'building':'yes'},'members':[{'type':'way','ref':2}]}] if 'map.json' in url else [
                {'type':'way','id':2,'nodes':[1,1]}, {'type':'node','id':1,'lat':10,'lon':77}]
            return httpx.Response(200,json={'elements':elements},request=httpx.Request('GET',url))
    monkeypatch.setattr(module.httpx,'AsyncClient',Client)
    result=asyncio.run(loader._fetch_map_tile(Tile(10.01,10,77.01,77)))
    assert {(e['type'],e['id']) for e in result}=={('relation',8),('way',2),('node',1)}
    assert len(writes)==1
