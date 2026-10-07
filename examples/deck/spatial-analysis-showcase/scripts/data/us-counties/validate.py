"""Read back us-counties bins, assert shapes, report NaNs, queen contiguity islands and Moran's I."""
import json, os, numpy as np
from collections import defaultdict
D = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../public/data/us-counties'))
m = json.load(open(f'{D}/manifest.json')); n = m['count']
DT = {'float32': np.float32, 'uint32': np.uint32, 'uint8': np.uint8}
def col(name):
    c = m['columns'][name]; a = np.fromfile(f"{D}/{c['file']}", dtype=DT[c['dtype']])
    assert len(a) == c['length'] * c['components'], name
    return a.reshape(-1, c['components']) if c['components'] > 1 else a
for k in m['columns']: col(k)
fips = col('fips'); assert len(set(fips)) == n and (np.diff(fips.astype(np.int64)) > 0).all()
nan = {k: int(np.isnan(col(k)).sum()) for k, c in m['columns'].items() if c['dtype'] == 'float32' and c['components'] == 1}
print('NaN counts (nonzero):', {k: v for k, v in nan.items() if v})
v = col('vertices'); assert np.isfinite(v).all(); b = m['bbox']
assert v[:, 0].min() >= b[0] - 1e-3 and v[:, 0].max() <= b[2] + 1e-3 and v[:, 1].min() >= b[1] - 1e-3 and v[:, 1].max() <= b[3] + 1e-3
cpo, pro, ro = col('countyPolygonOffsets'), col('polygonRingOffsets'), col('ringOffsets')
# vertex -> counties (exact match), queen contiguity
vx = defaultdict(set)
for i in range(n):
    for p in range(cpo[i], cpo[i + 1]):
        for r in range(pro[p], pro[p + 1]):  # exterior + holes (VA independent cities are holes)
            for xy in map(tuple, v[ro[r]:ro[r + 1] - 1]): vx[xy].add(i)
nb = defaultdict(set)
for s in vx.values():
    if len(s) > 1:
        for a in s:
            nb[a] |= s - {a}
deg = np.array([len(nb[i]) for i in range(n)])
print('queen: mean degree %.2f, min %d max %d, islands %d' % (deg.mean(), deg.min(), deg.max(), (deg == 0).sum()))
print('islands:', [m['properties'] and json.load(open(f'{D}/names.json'))['name'][i] for i in np.where(deg == 0)[0]])
try:
    import esda, libpysal
    from libpysal.weights import W
    w = W({i: sorted(nb[i]) for i in range(n)}, silence_warnings=True); w.transform = 'r'
    for k in ['places_diabetes_ageAdj', 'medianHouseholdIncome', 'sviOverall', 'minorityShare']:
        y = col(k).astype(float); ok = ~np.isnan(y); y = np.where(ok, y, np.nanmean(y))
        print(k, "Moran's I = %.3f" % esda.Moran(y, w, permutations=0).I)
except Exception as e: print('pysal skipped', e)
print('validate OK')
