#!/usr/bin/env python3
"""Build public/data/naturalearth-atlantic-coast from Natural Earth 1:50m coastline (public domain).

Usage: python3 -I build.py <ne_50m_coastline.geojson> <out-dir>
Keeps coastline parts that intersect the Atlantic hurricane window and clips them to it.
"""
import json, sys, os, array, math

SRC, OUT = sys.argv[1], sys.argv[2]
W, S, E, N = -105.0, 5.0, 15.0, 62.0
data = json.load(open(SRC))


def inside(p):
    return W <= p[0] <= E and S <= p[1] <= N


parts = []
for feature in data['features']:
    geom = feature['geometry']
    lines = geom['coordinates'] if geom['type'] == 'MultiLineString' else [geom['coordinates']]
    for line in lines:
        run = []
        for p in line:
            if inside(p):
                run.append(p)
            elif len(run) > 1:
                parts.append(run); run = []
            else:
                run = []
        if len(run) > 1:
            parts.append(run)

offsets, vertices = array.array('I', [0]), array.array('f')
for run in parts:
    for p in run:
        vertices.extend([p[0], p[1]])
    offsets.append(len(vertices) // 2)

os.makedirs(OUT, exist_ok=True)
for name, arr in (('pathOffsets', offsets), ('vertices', vertices)):
    with open(os.path.join(OUT, name + '.bin'), 'wb') as h: arr.tofile(h)
lons, lats = vertices[0::2], vertices[1::2]
manifest = {
    'id': 'naturalearth-atlantic-coast', 'version': 1, 'kind': 'lines', 'count': len(parts),
    'bbox': [min(lons), min(lats), max(lons), max(lats)], 'crs': 'EPSG:4326',
    'columns': {
        'pathOffsets': {'file': 'pathOffsets.bin', 'dtype': 'uint32', 'components': 1, 'length': len(offsets)},
        'vertices': {'file': 'vertices.bin', 'dtype': 'float32', 'components': 2, 'length': len(vertices) // 2}},
    'properties': {'source': 'Natural Earth ne_50m_coastline (1:50,000,000)', 'window': [W, S, E, N],
                   'note': 'Generalised coastline: islands and small bays are simplified, so distances under about 20 km are approximate.'}}
json.dump(manifest, open(os.path.join(OUT, 'manifest.json'), 'w'), indent=1)
print(len(parts), 'parts', len(vertices) // 2, 'vertices')
