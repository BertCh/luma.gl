import json, os, numpy as np, shapely
from shapely.geometry import shape
from shapely import points
APP = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
D = APP + '/public/data/ais-vessels/'; m = json.load(open(D + 'manifest.json'))
rd = lambda k: np.fromfile(D + m['columns'][k]['file'], dtype=m['columns'][k]['dtype']).reshape(-1, m['columns'][k]['components'])
off = rd('pathOffsets')[:, 0]; v = rd('vertices'); t = rd('timestamp')[:, 0]
for k, c in m['columns'].items(): assert rd(k).shape[0] == c['length'], k
assert off[-1] == len(v) == len(t) and len(off) == m['count'] + 1
assert np.isfinite(v).all() and not np.isnan(rd('speed')).any() and rd('heading').max() < 360
for i in range(m['count']): assert (np.diff(t[off[i]:off[i+1]].astype(int)) > 0).all()
print('trajectories ok', m['count'], len(v), v.min(0), v.max(0), t.max())
z = json.load(open(APP + '/public/data/ais-zones/zones.geojson'))
P = points(v[:, 0].astype(float), v[:, 1].astype(float)); pid = np.repeat(np.arange(m['count']), np.diff(off))
for f in z['features']:
    inside = shapely.contains_xy(shape(f['geometry']), v[:, 0], v[:, 1])
    if f['properties']['kind'] in ('gate', 'terminal', 'tourist', 'ferry') or inside.sum() > 800:
        print(f['properties']['name'][:50], 'pts', int(inside.sum()), 'paths', len(set(pid[inside])))
