"""Build chicago-roads: simplified OSM drive network of the City of Chicago, largest strongly connected component.
usage: build.py PBF COMMUNITY_AREAS_GEOJSON OUT_DIR [CACHE_PICKLE]
Also writes OUT_DIR/../../../../scripts cache? no: pass CACHE_PICKLE to dump the arrays for downstream builders (crash snapping, GPS traces)."""
import json, sys, pathlib, pickle, re
import numpy as np, pandas as pd, geopandas as gpd, shapely, networkx as nx, osmnx as ox, pyrosm
from shapely.geometry import LineString
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
pbf, ca_path, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3]); out.mkdir(parents=True, exist_ok=True)
cache = sys.argv[4] if len(sys.argv) > 4 else None
city = gpd.read_file(ca_path).to_crs(4326).union_all()
osm = pyrosm.OSM(str(pbf))
nodes, edges = osm.get_network('driving', nodes=True)
print('raw segments', len(edges), flush=True)
edges = edges[~edges.highway.isin(['busway', 'emergency_bay', 'closed'])].copy()
if 'access' in edges: edges = edges[~edges.access.isin(['no', 'private'])].copy()
# clip to city: both end nodes inside the boundary
nx_ = nodes.set_index('id')
inside = pd.Series(shapely.contains_xy(city, nx_.lon.values, nx_.lat.values), index=nx_.index)
edges = edges[inside.reindex(edges.u).values & inside.reindex(edges.v).values].copy()
print('segments in city', len(edges), flush=True)
CLASSES = ['motorway', 'trunk', 'primary', 'secondary', 'tertiary', 'residential', 'service/other']
def cls(h):
    h = h.replace('_link', '')
    return CLASSES.index(h) if h in CLASSES[:6] else 6
DEFAULT_KPH = [88, 64, 48, 48, 40, 40, 32]   # Chicago: 55 mph expressway, 30 mph arterial, 25-30 residential
def parse_speed(s, c):
    if isinstance(s, str):
        m = re.match(r'\s*(\d+(?:\.\d+)?)\s*(mph)?', s)
        if m: v = float(m.group(1)); return v * 1.609344 if m.group(2) else v
    return DEFAULT_KPH[c]
edges['cls'] = edges.highway.map(cls)
edges['kph'] = [parse_speed(s, c) for s, c in zip(edges.maxspeed, edges.cls)]
ow = edges.oneway.astype(str).str.lower()
implied = edges.highway.isin(['motorway', 'motorway_link']) | edges.junction.isin(['roundabout', 'circular'])
fwd = ow.isin(['yes', 'true', '1']) | (implied & (ow != '-1') & (ow != 'no'))
rev = ow == '-1'
edges['oneway_flag'] = (fwd | rev).astype(int)
# build directed MultiDiGraph
G = nx.MultiDiGraph(crs='EPSG:4326')
G.add_nodes_from((int(i), {'x': float(x), 'y': float(y)}) for i, x, y in zip(nx_.index, nx_.lon, nx_.lat) if True)
for u, v, g, c, k, o, f, r, ln in zip(edges.u, edges.v, edges.geometry, edges.cls, edges.kph, edges.oneway_flag, fwd, rev, edges['length']):
    attr = dict(cls=int(c), kph=round(float(k)), oneway=int(o), length=float(ln))
    if not r: G.add_edge(int(u), int(v), geometry=g, **attr)
    if not f: G.add_edge(int(v), int(u), geometry=LineString(list(g.coords)[::-1]), **attr)
G.remove_nodes_from([n for n in list(G) if G.degree(n) == 0])
print('graph', len(G), G.number_of_edges(), flush=True)
G = ox.simplification.simplify_graph(G, edge_attrs_differ=['cls', 'kph', 'oneway'], remove_rings=False)
print('simplified', len(G), G.number_of_edges(), flush=True)
# largest SCC
comp = max(nx.strongly_connected_components(G), key=len)
G = G.subgraph(comp).copy()
ids = sorted(G.nodes); idx = {n: i for i, n in enumerate(ids)}
rows = []
for u, v, k, d in G.edges(keys=True, data=True):
    g = d['geometry']
    def one(x): return x[0] if isinstance(x, list) else x
    rows.append((idx[u], idx[v], float(d['length']) if not isinstance(d['length'], list) else sum(d['length']), int(one(d['cls'])), int(one(d['kph'])), int(one(d['oneway'])), g))
E = pd.DataFrame(rows, columns=['s', 't', 'len', 'cls', 'kph', 'ow', 'geom']).sort_values(['s', 't', 'len'], kind='stable').reset_index(drop=True)
n = len(ids); m = len(E)
# reverse edge lookup (pair with same endpoints swapped and closest length)
key = {}
for i, (s, t, l) in enumerate(zip(E.s, E.t, E.len)): key.setdefault((s, t), []).append((l, i))
rev_idx = np.full(m, 0xFFFFFFFF, np.uint32)
for i, (s, t, l) in enumerate(zip(E.s, E.t, E.len)):
    c = key.get((t, s))
    if c: rev_idx[i] = min(c, key=lambda a: abs(a[0] - l))[1]
# geometry: Douglas-Peucker ~1 m, ensure endpoints equal node coords
geoms = shapely.simplify(np.array(E.geom.values, dtype=object), 1e-5)
ncoords = np.array([[G.nodes[i]['x'], G.nodes[i]['y']] for i in ids], '<f4')
offs = np.zeros(m + 1, '<u4'); verts = []
for i, g in enumerate(geoms):
    c = np.asarray(g.coords, '<f4').reshape(-1, 2); verts.append(c); offs[i + 1] = offs[i] + len(c)
verts = np.concatenate(verts).astype('<f4')
src = E.s.values.astype('<u4'); tgt = E.t.values.astype('<u4')
node_off = np.zeros(n + 1, '<u4'); np.add.at(node_off, src.astype(np.int64) + 1, 1); node_off = np.cumsum(node_off).astype('<u4')
length = E.len.values.astype('<f4'); kph = E.kph.values.astype('<u1'); cl = E.cls.values.astype('<u1')
tt = (length / (kph.astype('f4') / 3.6)).astype('<f4')
cols = {}
def add(name, arr, dtype, comp=1, **extra):
    arr = np.ascontiguousarray(arr.astype(dtype)); arr.tofile(out / f'{name}.bin')
    cols[name] = {'file': f'{name}.bin', 'dtype': np.dtype(dtype).name, 'components': comp, 'length': int(arr.size // comp), **extra}
add('nodes', ncoords, '<f4', 2, note='node lon/lat')
add('edgeSource', src, '<u4', note='edges sorted by (source, target); directed, two-way streets appear in both directions')
add('edgeTarget', tgt, '<u4')
add('edgeLength', length, '<f4', unit='m', note='true length of the (simplified) edge in metres')
add('edgeClass', cl, '<u1', categories=CLASSES)
add('edgeSpeed', kph, '<u1', unit='km/h', note='OSM maxspeed where tagged (mph converted), else class default [88,64,48,48,40,40,32]')
add('edgeTravelTime', tt, '<f4', unit='s', note='length / speed, free-flow, no intersection delay')
add('edgeOneway', E.ow.values, '<u1', note='1 if the OSM way is one-way (no reverse edge), 0 for two-way streets')
add('edgeReverse', rev_idx, '<u4', note='index of the opposite-direction edge, 4294967295 if none (one-way)')
add('nodeEdgeOffsets', node_off, '<u4', note='CSR: out-edges of node i are edges nodeEdgeOffsets[i]..nodeEdgeOffsets[i+1]-1')
add('edgePathOffsets', offs, '<u4', note='n+1 offsets into edgeVertices')
add('edgeVertices', verts, '<f4', 2, note='edge polyline lon/lat, oriented source -> target')
b = [float(verts[:, 0].min()), float(verts[:, 1].min()), float(verts[:, 0].max()), float(verts[:, 1].max())]
cnt = np.bincount(cl, minlength=7); km = np.bincount(cl, weights=length, minlength=7) / 1000
deg = np.diff(node_off.astype(np.int64))
story = [
    f"{n:,} intersections and {m:,} directed edges ({int((E.ow==0).sum()//2):,} two-way street pairs plus {int((E.ow==1).sum()):,} one-way edges); {km.sum():,.0f} directed km of road in the largest strongly connected component.",
    'Directed km by class: ' + ', '.join(f"{c} {k:,.0f}" for c, k in zip(CLASSES, km)) + '.',
    f"The Chicago grid shows: {int((deg==4).sum()):,} four-way intersections ({(deg==4).mean():.0%} of nodes with 4 out-edges).",
    "Expressways (I-90/94 Dan Ryan and Kennedy, I-55 Stevenson, I-290 Eisenhower) are modelled as motorway edges and are the fast corridors for isochrones.",
    f"Median edge length {np.median(length):.0f} m, longest {length.max():,.0f} m.",
]
manifest = {'id': 'chicago-roads', 'version': 1, 'kind': 'network', 'count': m, 'nodeCount': n, 'bbox': [round(x, 5) for x in b], 'crs': 'EPSG:4326', 'columns': cols,
  'properties': {'description': 'OpenStreetMap driveable street network clipped to the City of Chicago, simplified (degree-2 nodes merged), largest strongly connected component. Directed edges, CSR-sorted by source.',
    'licence': 'ODbL 1.0', 'attribution': '© OpenStreetMap contributors', 'osmExtract': 'BBBike Chicago extract, ' + pbf.name, 'storyNotes': story}}
(out / 'manifest.json').write_text(json.dumps(manifest, indent=1))
print('\n'.join(story)); print('bytes', sum(p.stat().st_size for p in out.iterdir()))
if cache:
    pickle.dump(dict(nodes=ncoords, src=src, tgt=tgt, length=length, kph=kph, cls=cl, tt=tt, ow=E.ow.values, rev=rev_idx, geoms=list(geoms)), open(cache, 'wb'))
