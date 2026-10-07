"""Zones for ais-vessels: NOAA ENC anchorages + maintained channels (dissolved by fairway) + hand-drawn approximate polygons."""
import json, os, urllib.parse, urllib.request
import numpy as np
from shapely.geometry import shape, box, mapping
from shapely.ops import unary_union
HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
RAW = os.environ.get('SHOWCASE_RAW', '/private/tmp/claude-501/-Users-robertchristie-Documents-GitHub-deck-gl-master/85599621-aabc-441b-ab0f-35747be6f781/scratchpad/showcase/raw') + '/ais-vessels'
OUT = os.path.join(APP, 'public/data/ais-zones'); os.makedirs(OUT, exist_ok=True)
B = 'https://encdirect.noaa.gov/arcgis/rest/services/NavigationChartData'
ENV = '-74.30,40.45,-73.75,40.80'
def fetch(layer, name):
    p = f'{RAW}/{name}.geojson'
    if not os.path.exists(p):
        q = urllib.parse.urlencode(dict(geometry=ENV, geometryType='esriGeometryEnvelope', inSR=4326, outSR=4326, where='1=1', outFields='*', f='geojson'))
        urllib.request.urlretrieve(f'{B}/{layer}/query?{q}', p)
    return json.load(open(p))
anch = fetch('Anchorage_Areas/MapServer/0', 'anch'); chan = fetch('MarineTransportation/MapServer/1', 'chan')
feats = []
for f in anch['features']:
    inf = f['properties'].get('INFORM') or ''; nm = (f['properties'].get('OBJNAM') or '').strip()
    if '110.155' not in inf: continue  # general anchorages (not special small-craft anchorages)
    g = shape(f['geometry'])
    if g.area < 1e-6: continue
    feats.append(dict(name=' '.join(nm.split()) or 'Sandy Hook Bay Anchorage', kind='anchorage', source='NOAA ENC Anchorage Areas (ENCDirect)', geom=g.simplify(0.0002)))
groups = {}
for f in chan['features']:
    fw = (f['properties'].get('FAIRWAY') or '').strip()
    if not fw or f['properties'].get('THEMELAYER') != 'Dredged Area': continue
    groups.setdefault(' '.join(fw.split()), []).append(shape(f['geometry']))
for fw, gs in groups.items():
    g = unary_union([x.buffer(0.00005) for x in gs]).simplify(0.0002)
    if g.area < 2e-5: continue
    feats.append(dict(name=fw + ' (maintained channel)', kind='channel', source='NOAA ENC Coastal Maintained Channels (ENCDirect)', geom=g))
approx = [
 ('Verrazzano-Narrows Bridge span (approx.)', 'gate', box(-74.062, 40.602, -74.032, 40.611)),
 ('Port Newark-Elizabeth terminals (approx.)', 'terminal', box(-74.175, 40.665, -74.135, 40.705)),
 ('Red Hook container terminal (approx.)', 'terminal', box(-74.022, 40.672, -74.008, 40.684)),
 ('Bayonne-Port Jersey terminals (approx.)', 'terminal', box(-74.085, 40.645, -74.052, 40.680)),
 ('Statue of Liberty / Ellis Island tour area (approx.)', 'tourist', box(-74.057, 40.682, -74.032, 40.696)),
 ('Staten Island Ferry lane, St George-Whitehall (approx.)', 'ferry', box(-74.075, 40.640, -74.005, 40.705).intersection(box(-74.075, 40.640, -74.005, 40.705))),
]
for nm, k, g in approx:
    if k == 'ferry':
        from shapely.geometry import LineString
        g = LineString([(-74.073, 40.643), (-74.012, 40.700)]).buffer(0.003)
    feats.append(dict(name=nm, kind=k, source='hand-drawn, approximate (not an official boundary)', geom=g))
KINDS = ['anchorage', 'channel', 'terminal', 'gate', 'tourist', 'ferry']
fc = {'type': 'FeatureCollection', 'features': []}
for i, f in enumerate(feats):
    fc['features'].append({'type': 'Feature', 'properties': dict(id=i, name=f['name'], kind=f['kind'], source=f['source']), 'geometry': json.loads(json.dumps(mapping(f['geom']), default=list))})
def q(o):
    if isinstance(o, (list, tuple)): return [q(x) for x in o]
    return round(o, 6)
for f in fc['features']: f['geometry']['coordinates'] = q(f['geometry']['coordinates'])
json.dump(fc, open(f'{OUT}/zones.geojson', 'w'), separators=(',', ':'))
allg = unary_union([f['geom'] for f in feats]); b = allg.bounds
cnt = {k: sum(1 for f in feats if f['kind'] == k) for k in KINDS}
m = {'id': 'ais-zones', 'version': 1, 'kind': 'polygons', 'count': len(feats), 'bbox': list(b), 'crs': 'EPSG:4326',
     'geometry': {'type': 'polygons', 'file': 'zones.geojson'},
     'properties': {'kinds': KINDS, 'kindCounts': cnt, 'companion': 'ais-vessels',
       'attribution': 'NOAA Coast Survey ENCDirect (public domain) for anchorages and channels; terminal/gate/ferry/tourist polygons are hand-drawn approximations',
       'storyNotes': [f'{cnt["anchorage"]} official general anchorages (33 CFR 110.155) in the Upper/Lower Bay, Hudson and Raritan Bay.',
         f'{cnt["channel"]} maintained channels dissolved by fairway name (Ambrose, Anchorage, Kill Van Kull, Arthur Kill, Newark Bay ...).',
         'Terminal, bridge, ferry-lane and tourist-area polygons are approximate and flagged "(approx.)" in their names.']}}
json.dump(m, open(f'{OUT}/manifest.json', 'w'), indent=1)
print(cnt, os.path.getsize(f'{OUT}/zones.geojson'), [f['name'] for f in feats if f['kind']=='channel'])
