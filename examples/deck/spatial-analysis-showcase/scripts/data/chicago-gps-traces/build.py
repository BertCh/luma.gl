"""Build chicago-gps-traces: SIMULATED noisy GPS traces from random shortest paths on chicago-roads.
usage: build.py ROADS_CACHE_PKL OUT_DIR [N_TRACES=200] [SEED=7]"""
import json, sys, pathlib, pickle
import numpy as np, geopandas as gpd, shapely
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
pkl, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
N = int(sys.argv[3]) if len(sys.argv) > 3 else 200; rng = np.random.default_rng(int(sys.argv[4]) if len(sys.argv) > 4 else 7)
R = pickle.load(open(pkl, 'rb'))
src, tgt, tt, kph = R['src'].astype(np.int64), R['tgt'].astype(np.int64), R['tt'], R['kph']
nn = len(R['nodes']); m = len(src)
# fastest parallel edge per (s,t)
order = np.lexsort((tt, tgt, src)); first = np.ones(m, bool); first[1:] = (src[order][1:] != src[order][:-1]) | (tgt[order][1:] != tgt[order][:-1])
keep = order[first]
W = csr_matrix((tt[keep].astype(np.float64), (src[keep], tgt[keep])), shape=(nn, nn))
eid = {(int(src[i]), int(tgt[i])): int(i) for i in keep}
# node -> UTM
utm = gpd.GeoSeries(gpd.points_from_xy(R['nodes'][:, 0], R['nodes'][:, 1]), crs=4326).to_crs(26916)
nxy = np.column_stack([utm.x, utm.y])
to_ll = lambda xy: gpd.GeoSeries(gpd.points_from_xy(xy[:, 0], xy[:, 1]), crs=26916).to_crs(4326)
eline_cache = {}
def eline(i):
    if i not in eline_cache:
        eline_cache[i] = gpd.GeoSeries([R['geoms'][i]], crs=4326).to_crs(26916).iloc[0]
    return eline_cache[i]
T0 = 1685951400  # unused marker; times are seconds since 2023-06-05T00:00:00 local
traces = []
tries = 0
while len(traces) < N and tries < N * 20:
    tries += 1
    o = int(rng.integers(nn)); d, pred = dijkstra(W, indices=o, return_predecessors=True)
    cand = np.where((d > 420) & (d < 1500))[0]        # 7-25 min free-flow
    if len(cand) == 0: continue
    t = int(rng.choice(cand)); path = [t]
    while path[-1] != o: path.append(int(pred[path[-1]]))
    path = path[::-1]
    edges = [eid[(a, b)] for a, b in zip(path[:-1], path[1:])]
    if len(edges) < 6: continue
    lines = [eline(i) for i in edges]; glen = np.array([l.length for l in lines]); cum = np.concatenate([[0], np.cumsum(glen)])
    speed = (kph[edges].astype(float) / 3.6) * rng.uniform(0.55, 0.85)      # realistic fraction of limit (lights, traffic)
    etime = glen / speed; tcum = np.concatenate([[0], np.cumsum(etime)])
    dt = int(rng.integers(5, 16)); sigma = float(rng.uniform(10, 25)); start = int(rng.uniform(7, 19) * 3600)
    ts = np.arange(0, tcum[-1], dt, dtype=float)
    # dropouts: 0-3 gaps of 30-90 s
    keepm = np.ones(len(ts), bool)
    for _ in range(int(rng.integers(0, 4))):
        g0 = rng.uniform(0, tcum[-1]); keepm &= ~((ts >= g0) & (ts < g0 + rng.uniform(30, 90)))
    ts = ts[keepm]
    if len(ts) < 20: continue
    ei = np.clip(np.searchsorted(tcum, ts, side='right') - 1, 0, len(edges) - 1)
    fr = (ts - tcum[ei]) / etime[ei]
    true_xy = np.array([eline(edges[e]).interpolate(f, normalized=True).coords[0] for e, f in zip(ei, np.clip(fr, 0, 1))])
    noisy = true_xy + rng.normal(0, sigma, true_xy.shape)
    traces.append(dict(ts=(start + ts).astype('<u4'), true=true_xy, noisy=noisy, edge=np.array(edges)[ei], off=(fr * R['length'][np.array(edges)[ei]]), route=edges, sigma=sigma, dt=dt))
print('traces', len(traces), 'tries', tries, flush=True)
cnt = np.array([len(x['ts']) for x in traces]); offs = np.concatenate([[0], np.cumsum(cnt)]).astype('<u4')
cat = lambda k: np.concatenate([x[k] for x in traces])
noisy_ll = to_ll(cat('noisy')); true_ll = to_ll(cat('true'))
rcnt = np.array([len(x['route']) for x in traces]); roffs = np.concatenate([[0], np.cumsum(rcnt)]).astype('<u4')
cols = {}
def add(name, arr, dtype, comp=1, **extra):
    arr = np.ascontiguousarray(np.asarray(arr).astype(dtype)); arr.tofile(out / f'{name}.bin')
    cols[name] = {'file': f'{name}.bin', 'dtype': np.dtype(dtype).name, 'components': comp, 'length': int(arr.size // comp), **extra}
add('pathOffsets', offs, '<u4', note='trace i = vertices pathOffsets[i]..pathOffsets[i+1]-1')
add('vertices', np.column_stack([noisy_ll.x, noisy_ll.y]), '<f4', 2, note='SIMULATED noisy GPS fixes lon/lat')
add('timestamp', cat('ts'), '<u4', unit='seconds since 2023-06-05T00:00:00 (simulated Monday, local time)')
add('truthPosition', np.column_stack([true_ll.x, true_ll.y]), '<f4', 2, note='noise-free position on the true route, for validation')
add('truthEdge', cat('edge'), '<u4', note='chicago-roads edge index the vehicle was on at each fix')
add('truthEdgeOffset', cat('off'), '<f4', unit='m', note='distance along the true edge from its source')
add('routeOffsets', roffs, '<u4', note='ground-truth edge sequence of trace i = routeEdges[routeOffsets[i]..routeOffsets[i+1]-1]')
add('routeEdges', np.concatenate([x['route'] for x in traces]), '<u4', note='chicago-roads directed edge indices in travel order')
add('noiseSigma', [x['sigma'] for x in traces], '<f4', unit='m', note='per-trace Gaussian noise std dev (isotropic)')
add('sampleInterval', [x['dt'] for x in traces], '<u1', unit='s', note='nominal per-trace sampling interval')
err = np.hypot(*(cat('noisy') - cat('true')).T)
allv = np.column_stack([noisy_ll.x, noisy_ll.y])
story = [
    f"SIMULATED data: {len(traces)} traces, {int(cnt.sum()):,} fixes, produced by routing random origin-destination pairs (7-25 min free-flow) on chicago-roads; not real vehicles.",
    f"Noise: per-trace Gaussian sigma 10-25 m per axis (radial RMS error {np.sqrt((err**2).mean()):.1f} m), sampling every 5-15 s, 0-3 dropouts of 30-90 s per trace; speed is 55-85% of the posted limit.",
    f"Median trace has {int(np.median(cnt))} fixes over {np.median([x['ts'][-1]-x['ts'][0] for x in traces])/60:.0f} min; routes cross {int(rcnt.sum()):,} edges in total (median {int(np.median(rcnt))} per trace).",
    "Ground truth: truthEdge per fix and routeEdges per trace let a map-matching tool report edge accuracy; noise on dense downtown blocks (~100 m) is comparable to block spacing, so it is genuinely ambiguous there.",
]
man = {'id': 'chicago-gps-traces', 'version': 1, 'kind': 'trajectories', 'count': len(traces), 'bbox': [round(float(allv[:, 0].min()), 5), round(float(allv[:, 1].min()), 5), round(float(allv[:, 0].max()), 5), round(float(allv[:, 1].max()), 5)], 'crs': 'EPSG:4326', 'columns': cols,
  'properties': {'description': 'SIMULATED (synthetic) GPS traces for map matching: random shortest paths on chicago-roads, sampled every 5-15 s with 10-25 m Gaussian noise and occasional dropouts. Ground-truth edge sequences are included. Not real vehicle data.',
    'simulated': True, 'roadsDataset': 'chicago-roads', 'licence': 'Synthetic, derived from OpenStreetMap data (ODbL 1.0)', 'attribution': '© OpenStreetMap contributors', 'storyNotes': story}}
(out / 'manifest.json').write_text(json.dumps(man, indent=1))
print('\n'.join(story)); print('bytes', sum(p.stat().st_size for p in out.iterdir()))
