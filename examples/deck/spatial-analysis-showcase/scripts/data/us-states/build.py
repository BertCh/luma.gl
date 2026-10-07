#!/usr/bin/env python
"""Build `us-states`: contiguous-US state outlines dissolved from the us-counties geometry (so edges match exactly).
Run build us-counties first."""
import json, os
import numpy as np, geopandas as gpd
from shapely.geometry import mapping, Polygon, MultiPolygon
from shapely.ops import unary_union

SC = '/private/tmp/claude-501/-Users-robertchristie-Documents-GitHub-deck-gl-master/85599621-aabc-441b-ab0f-35747be6f781/scratchpad/showcase'
APP = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
OUT = f'{APP}/public/data/us-states'; CTY = f'{APP}/public/data/us-counties'
os.makedirs(OUT, exist_ok=True)
g = gpd.read_file(f'{SC}/raw/us-counties/shp20/cb_2022_us_county_20m.shp')
g = g[~g.STATEFP.isin(['02', '15', '60', '66', '69', '72', '78'])]
rows = []
for sf, h in g.groupby('STATEFP'):
    u = unary_union([x.buffer(0) for x in h.geometry])
    rows.append((sf, h.STATE_NAME.iloc[0], h.STUSPS.iloc[0], u))
rows.sort()
def rnd(geom):
    ps = list(geom.geoms) if isinstance(geom, MultiPolygon) else [geom]
    out = [Polygon(np.round(np.asarray(p.exterior.coords), 4), [np.round(np.asarray(i.coords), 4) for i in p.interiors]) for p in ps]
    return MultiPolygon(out) if len(out) > 1 else out[0]
feats = [{'type': 'Feature', 'id': int(sf), 'properties': {'fips': sf, 'name': nm, 'abbr': ab}, 'geometry': mapping(rnd(u))} for sf, nm, ab, u in rows]
open(f'{OUT}/states.geojson', 'w').write(json.dumps({'type': 'FeatureCollection', 'features': feats}, separators=(',', ':')))
cen = gpd.GeoSeries([r[3] for r in rows], crs=4326).to_crs(5070).centroid.to_crs(4326)
np.column_stack([cen.x, cen.y]).astype(np.float32).tofile(f'{OUT}/centroid.bin')
np.array([int(r[0]) for r in rows], dtype=np.uint8).tofile(f'{OUT}/fips.bin')
b = gpd.GeoSeries([r[3] for r in rows]).total_bounds
n = len(rows)
manifest = {'id': 'us-states', 'version': 1, 'kind': 'polygons', 'count': n, 'crs': 'EPSG:4326',
  'bbox': [round(float(x), 4) for x in b],
  'columns': {'fips': {'file': 'fips.bin', 'dtype': 'uint8', 'components': 1, 'length': n, 'description': 'State FIPS'},
              'centroid': {'file': 'centroid.bin', 'dtype': 'float32', 'components': 2, 'length': n}},
  'geometry': {'type': 'polygons', 'file': 'states.geojson'},
  'properties': {'coverage': '48 contiguous states + DC (AK and HI dropped)', 'order': 'sorted by state FIPS',
                 'derivation': 'dissolved from Census cb_2022_us_county_20m (1:20,000,000) counties, 4-decimal coordinates',
                 'names': [r[1] for r in rows], 'abbr': [r[2] for r in rows],
                 'storyNotes': ['Outlines match us-counties edges exactly, so they overlay county choropleths without slivers.',
                                'Use as a thin outline layer; 49 features.']}}
json.dump(manifest, open(f'{OUT}/manifest.json', 'w'), indent=1)
print(n, 'states', os.path.getsize(f'{OUT}/states.geojson') / 1e3, 'KB')
