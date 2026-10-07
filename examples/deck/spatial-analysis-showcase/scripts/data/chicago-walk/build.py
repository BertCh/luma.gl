"""Build chicago-walk: OSM pedestrian network (Loop + near neighbourhoods, ~5 x 5 km), both directions, largest component.
usage: build.py PBF OUT_DIR"""
import json, sys, pathlib
import numpy as np, pandas as pd, networkx as nx, osmnx as ox, pyrosm, shapely
from shapely.geometry import LineString
pbf, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
BBOX = [-87.6652, 41.8575, -87.5998, 41.9025]     # ~5.4 km x 5.0 km centred on the Loop / Near North
nodes, edges = pyrosm.OSM(str(pbf), bounding_box=BBOX).get_network('walking', nodes=True)
edges = edges[~edges.highway.isin(['corridor', 'elevator', 'busway', 'emergency_bay'])].copy()
if 'access' in edges: edges = edges[~edges.access.isin(['no', 'private'])]
if 'foot' in edges: edges = edges[edges.foot != 'no']
if 'service' in edges: edges = edges[~((edges.highway == 'service') & edges.service.isin(['parking_aisle', 'driveway', 'drive-through']))]
CLASSES = ['sidewalk', 'crossing', 'path/pedestrian', 'street', 'steps']
def cls(h, f):
    if h == 'steps': return 4
    if f == 'crossing': return 1
    if f == 'sidewalk': return 0
    if h in ('footway', 'path', 'pedestrian'): return 2
    return 3
edges['cls'] = [cls(h, f) for h, f in zip(edges.highway, edges.footway if 'footway' in edges else [None] * len(edges))]
nxy = nodes.set_index('id')
G = nx.MultiDiGraph(crs='EPSG:4326')
used = set(edges.u) | set(edges.v)
G.add_nodes_from((int(i), {'x': float(nxy.lon[i]), 'y': float(nxy.lat[i])}) for i in used if i in nxy.index)
for u, v, g, c, ln in zip(edges.u, edges.v, edges.geometry, edges.cls, edges['length']):
    if int(u) not in G or int(v) not in G: continue
    G.add_edge(int(u), int(v), geometry=g, cls=int(c), length=float(ln))
    G.add_edge(int(v), int(u), geometry=LineString(list(g.coords)[::-1]), cls=int(c), length=float(ln))
G = ox.simplification.simplify_graph(G, edge_attrs_differ=['cls'], remove_rings=False)
G = G.subgraph(max(nx.strongly_connected_components(G), key=len)).copy()
ids = sorted(G.nodes); idx = {n: i for i, n in enumerate(ids)}
one = lambda x: x[0] if isinstance(x, list) else x
rows = [(idx[u], idx[v], sum(d['length']) if isinstance(d['length'], list) else d['length'], int(one(d['cls'])), d['geometry']) for u, v, d in G.edges(data=True)]
E = pd.DataFrame(rows, columns=['s', 't', 'len', 'cls', 'geom']).sort_values(['s', 't', 'len'], kind='stable').reset_index(drop=True)
n, m = len(ids), len(E)
geoms = shapely.simplify(np.array(E.geom.values, dtype=object), 1e-5)
offs = np.zeros(m + 1, '<u4'); vs = []
for i, g in enumerate(geoms): c = np.asarray(g.coords, '<f4').reshape(-1, 2); vs.append(c); offs[i + 1] = offs[i] + len(c)
verts = np.concatenate(vs); ncoords = np.array([[G.nodes[i]['x'], G.nodes[i]['y']] for i in ids], '<f4')
src, tgt = E.s.values.astype('<u4'), E.t.values.astype('<u4'); length = E.len.values.astype('<f4')
node_off = np.cumsum(np.concatenate([[0], np.bincount(src, minlength=n)])).astype('<u4')
tt = (length / (4.8 / 3.6)).astype('<f4')
cols = {}
def add(name, arr, dtype, comp=1, **extra):
    arr = np.ascontiguousarray(np.asarray(arr).astype(dtype)); arr.tofile(out / f'{name}.bin')
    cols[name] = {'file': f'{name}.bin', 'dtype': np.dtype(dtype).name, 'components': comp, 'length': int(arr.size // comp), **extra}
add('nodes', ncoords, '<f4', 2)
add('edgeSource', src, '<u4', note='directed; every walkway appears in both directions; sorted by source')
add('edgeTarget', tgt, '<u4'); add('edgeLength', length, '<f4', unit='m')
add('edgeClass', E.cls.values, '<u1', categories=CLASSES)
add('edgeTravelTime', tt, '<f4', unit='s', note='length / 4.8 km/h; crossings carry no signal delay')
add('nodeEdgeOffsets', node_off, '<u4'); add('edgePathOffsets', offs, '<u4'); add('edgeVertices', verts, '<f4', 2)
km = np.bincount(E.cls.values, weights=length, minlength=5) / 2000
story = [f"{n:,} nodes and {m:,} directed edges ({m//2:,} walkable links) in the Loop, Near North, Near West and Near South sides.",
         "Walkway km by class (one direction): " + ', '.join(f"{c} {k:.0f}" for c, k in zip(CLASSES, km)) + ".",
         "Sidewalks and crossings are mapped as separate ways, so walk isochrones can be compared with street-centreline isochrones from chicago-roads.",
         f"Median link {np.median(length):.0f} m; walking speed fixed at 4.8 km/h."]
b = [float(verts[:, 0].min()), float(verts[:, 1].min()), float(verts[:, 0].max()), float(verts[:, 1].max())]
man = {'id': 'chicago-walk', 'version': 1, 'kind': 'network', 'count': m, 'nodeCount': n, 'bbox': [round(x, 5) for x in b], 'crs': 'EPSG:4326', 'columns': cols,
  'properties': {'description': 'OpenStreetMap pedestrian network (sidewalks, crossings, paths, walkable streets) of the Loop and neighbouring areas, simplified, largest connected component, both directions.',
   'licence': 'ODbL 1.0', 'attribution': '© OpenStreetMap contributors', 'storyNotes': story}}
(out / 'manifest.json').write_text(json.dumps(man, indent=1)); print('\n'.join(story)); print('bytes', sum(p.stat().st_size for p in out.iterdir()))
