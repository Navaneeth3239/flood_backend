import gzip,json
from services import complete_buildings as module

BOX={'south':10,'west':77,'north':10.01,'east':77.01}
POLY=[[10,77],[10,77.01],[10.01,77.01],[10.01,77]]
def feature(x=77.001,y=10.001):
 return {'type':'Feature','properties':{},'geometry':{'type':'Polygon','coordinates':[[[x,y],[x+.00001,y],[x+.00001,y+.00001],[x,y+.00001],[x,y]]]}}
def tile(monkeypatch,tmp_path,features):
 monkeypatch.setattr(module,'TILES_DIR',tmp_path)
 monkeypatch.setattr(module,'bbox_to_quadkeys',lambda *args:['test'])
 monkeypatch.setattr(module,'tile_parts',lambda key:[tmp_path/'test.csv.gz'])
 with gzip.open(tmp_path/'test.csv.gz','wt') as stream:
  for f in features: stream.write(json.dumps(f)+'\n')
def test_no_2500_building_cap(monkeypatch,tmp_path):
 tile(monkeypatch,tmp_path,[feature(77.001+(i%60)*.0001,10.001+(i//60)*.0001) for i in range(3001)])
 result=module.enrich_buildings({'features':[]},BOX,POLY)
 assert len(result['features'])==3001
 assert len({f['id'] for f in result['features']})==3001

def test_merge_preserves_osm_and_boundary_crossing_houses(monkeypatch,tmp_path):
 mapped=feature()
 crossing=feature(76.999995,10.005)
 tile(monkeypatch,tmp_path,[mapped,crossing,feature(78,10)])
 result=module.enrich_buildings({'features':[mapped]},BOX,POLY)
 assert len(result['features'])==2
 assert result['features'][0]==mapped
 assert result['features'][1]['geometry']==crossing['geometry']

def test_border_quadkey_keeps_every_country_partition(monkeypatch,tmp_path):
 import io
 monkeypatch.setattr(module,'TILES_DIR',tmp_path)
 monkeypatch.setattr(module,'_parts_index',{'border':[{'url':'https://example.com/india'},{'url':'https://example.com/nepal'}]})
 monkeypatch.setattr(module,'load_quadkey_index',lambda:{})
 monkeypatch.setattr(module.urllib.request,'urlopen',lambda url,**kwargs:io.BytesIO(url.encode()))
 paths=module.tile_parts('border')
 assert len(paths)==2 and paths[0]!=paths[1]
 assert [p.read_text() for p in paths]==['https://example.com/india','https://example.com/nepal']
