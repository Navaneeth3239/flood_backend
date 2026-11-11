"""Regression coverage for complete per-area layers and Supabase cascade deletion."""
import asyncio
import os
import uuid
from unittest.mock import AsyncMock

import networkx as nx
import pytest
from fastapi import HTTPException
from routers import routing_and_rivers as api
from services import area_map_store

BOUNDARY = [[10.,77.],[10.,77.01],[10.01,77.01],[10.01,77.]]
EMPTY = {"type":"FeatureCollection", "features":[], "metadata":{"building_version":3}}


def test_partial_network_reuses_saved_layer_and_never_synthesizes(monkeypatch):
    stored = {"roads": EMPTY}
    monkeypatch.setattr(area_map_store, 'load_layers', lambda *args: stored.copy())
    monkeypatch.setattr(area_map_store, 'save_layer', lambda aid,key,name,fc: stored.update({name:fc}))
    roads = AsyncMock(side_effect=AssertionError('Roads already saved'))
    rivers = AsyncMock(side_effect=[RuntimeError('timeout'), (nx.DiGraph(),EMPTY)])
    monkeypatch.setattr(api.road_service,'get_road_network',roads)
    monkeypatch.setattr(api.river_service,'get_river_network',rivers)
    payload=api.LocationRequest(polygon=BOUNDARY,area_id=str(uuid.uuid4()))
    first=asyncio.run(api.extract_networks(payload))
    assert first['osm_loading']['failed_layers']==['rivers']
    assert first['rivers']['geojson']['features']==[]
    assert 'rivers' not in stored
    second=asyncio.run(api.extract_networks(payload))
    assert second['osm_loading']['complete']
    assert 'rivers' in stored
    roads.assert_not_awaited()


def test_building_failure_is_not_saved_as_empty(monkeypatch):
    monkeypatch.setattr(area_map_store,'load_layers',lambda *args: {})
    saves=[]
    monkeypatch.setattr(area_map_store,'save_layer',lambda *args: saves.append(args))
    monkeypatch.setattr(api.building_service,'get_buildings',AsyncMock(side_effect=RuntimeError('timeout')))
    with pytest.raises(HTTPException) as error:
        asyncio.run(api.extract_buildings(api.LocationRequest(polygon=BOUNDARY)))
    assert error.value.status_code==503
    assert saves==[]


def test_successful_empty_buildings_are_loaded_from_cache(monkeypatch):
    monkeypatch.setattr(area_map_store,'load_layers',lambda *args:{'buildings':EMPTY})
    provider=AsyncMock(side_effect=AssertionError('Empty is a complete result'))
    monkeypatch.setattr(api.building_service,'get_buildings',provider)
    result=asyncio.run(api.extract_buildings(api.LocationRequest(polygon=BOUNDARY)))
    assert result['osm_loading']['complete']
    assert result['buildings']['total_features']==0
    provider.assert_not_awaited()


def test_failed_save_is_visible_and_not_reported_complete(monkeypatch):
    monkeypatch.setattr(area_map_store,'load_layers',lambda *args: {})
    def failed(*args): raise RuntimeError('Database unavailable')
    monkeypatch.setattr(area_map_store,'save_layer',failed)
    monkeypatch.setattr(api.building_service,'get_buildings',AsyncMock(return_value=(EMPTY,{'complete':True})))
    with pytest.raises(HTTPException):
        asyncio.run(api.extract_buildings(api.LocationRequest(polygon=BOUNDARY)))


def test_boundary_identity_does_not_match_nearby_or_resized_areas():
    assert area_map_store.boundary_key(BOUNDARY,{}) != area_map_store.boundary_key([[10.,77.],[10.,77.02],[10.01,77.01],[10.01,77.]],{})


@pytest.mark.skipif(os.environ.get('RUN_SUPABASE_INTEGRATION')!='1',reason='Opt in to disposable Supabase integration test')
def test_supabase_roundtrip_all_layers_and_atomic_area_deletion():
    from lib.db import supabase
    aid=str(uuid.uuid4())
    other=str(uuid.uuid4())
    key=area_map_store.boundary_key(BOUNDARY,{})
    # >1000 features verifies we do not silently truncate at the REST row limit.
    fc={'type':'FeatureCollection','features':[{'type':'Feature','id':str(i),'geometry':{'type':'LineString','coordinates':[[77,10],[77.01,10.01]]},'properties':{}} for i in range(1205)]}
    try:
        for id in [aid,other]:
            supabase.table('custom_areas').insert({'id':id,'name':'Disposable layer regression','lat':10,'lng':77,'shape':'polygon','description':'Keep this description'}).execute()
        for layer in ['roads','rivers','buildings']:
            area_map_store.save_layer(aid,key,layer,fc if layer!='rivers' else EMPTY)
        area_map_store.save_layer(other,key,'roads',EMPTY)
        loaded=area_map_store.load_layers(aid,key)
        assert len(loaded['roads']['features'])==1205
        assert len(loaded['buildings']['features'])==1205
        assert loaded['rivers']==EMPTY
        assert area_map_store.load_layers(aid,'different-boundary')=={}
        desc=supabase.table('custom_areas').select('description').eq('id',aid).execute().data[0]['description']
        assert desc=='Keep this description'
        for legacy in [aid,'dt-area-'+aid]:
            supabase.table('digital_twin_network_features').insert({'id':legacy+'_test','area_key':legacy,'feature_id':'test','feature_type':'path','geometry':{'type':'LineString','coordinates':[[77,10],[77.01,10.01]]},'properties':{},'bbox':[77,10,77.01,10.01]}).execute()
        supabase.table('custom_areas').delete().eq('id',aid).execute()
        assert area_map_store.load_layers(aid,key)=={}
        assert area_map_store.load_layers(other,key)=={'roads':EMPTY}
        assert not supabase.table('digital_twin_network_features').select('id').in_('area_key',[aid,'dt-area-'+aid]).execute().data
        with pytest.raises(Exception):
            area_map_store.save_layer(aid,key,'buildings',fc)
    finally:
        for id in [aid,other]:
            supabase.table('custom_areas').delete().eq('id',id).execute()
