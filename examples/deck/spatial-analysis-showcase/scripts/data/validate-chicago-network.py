"""Read back every manifest of the chicago-network datasets and assert shapes, bounds and index ranges.
usage: validate-chicago-network.py PUBLIC_DATA_DIR"""
import json, sys, pathlib, numpy as np
P = pathlib.Path(sys.argv[1])
def load(ds):
    d = P / ds; m = json.load(open(d / 'manifest.json')); a = {}
    for k, c in m['columns'].items():
        x = np.fromfile(d / c['file'], np.dtype(c['dtype']))
        assert x.size == c['length'] * c['components'], (ds, k, x.size, c)
        assert not (np.issubdtype(x.dtype, np.floating) and np.isnan(x).any()), (ds, k, 'NaN')
        a[k] = x.reshape(-1, c['components']) if c['components'] > 1 else x
    return m, a
for ds in ['chicago-roads', 'chicago-walk']:
    m, a = load(ds); n = len(a['nodes']); E = len(a['edgeSource'])
    assert a['edgeSource'].max() < n and a['edgeTarget'].max() < n and (np.diff(a['edgeSource'].astype(int)) >= 0).all()
    assert a['edgePathOffsets'][-1] == len(a['edgeVertices']) and a['nodeEdgeOffsets'][-1] == E and len(a['edgePathOffsets']) == E + 1
    assert (a['edgeLength'] > 0).all(); b = m['bbox']; assert -88.0 < b[0] < b[2] < -87.5 and 41.6 < b[1] < b[3] < 42.1
    print(ds, 'ok', n, E, 'bytes', sum(f.stat().st_size for f in (P / ds).iterdir()))
    if ds == 'chicago-roads':
        r = a['edgeReverse']; ok = r != 0xFFFFFFFF
        assert (a['edgeSource'][r[ok]] == a['edgeTarget'][ok]).all(); assert ((a['edgeOneway'] == 1) == ~ok).mean() > 0.97
        R = a
m, a = load('chicago-crashes'); e = a['edgeIndex']; ok = e != 0xFFFFFFFF
assert (a['edgeOffset'][ok] <= R['edgeLength'][e[ok]] + 1e-3).all() and a['timestamp'].max() < 365 * 86400; print('crashes ok', len(e), ok.mean())
m, a = load('chicago-gps-traces'); assert a['pathOffsets'][-1] == len(a['vertices']) and a['truthEdge'].max() < len(R['edgeSource'])
re = a['routeEdges'].astype(int); ro = a['routeOffsets']
for i in range(len(ro) - 1):
    s = re[ro[i]:ro[i + 1]]; assert (R['edgeTarget'][s[:-1]] == R['edgeSource'][s[1:]]).all()
print('gps ok', len(ro) - 1)
m, a = load('chicago-taxi-od'); assert a['origin'].max() < 77 and a['hourlyHour'].max() == 23 and a['dailyPickups'].size == 365 * 77
assert a['count'].sum() == a['hourlyCount'].sum(); print('taxi ok', a['count'].sum())
m, a = load('cta-transit'); ns = len(a['stopPosition']); assert a['hopSource'].max() < ns and a['hopTarget'].max() < ns and a['shapePathOffsets'][-1] == len(a['shapeVertices']) and a['stopRoutes'].max() < len(m['properties']['routes']); print('cta ok', ns)
