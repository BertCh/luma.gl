"""Build chicago-crashes: 2023 traffic crashes snapped onto chicago-roads.
usage: build.py CRASHES_CSV ROADS_CACHE_PKL ROADS_MANIFEST_DIR OUT_DIR"""
import json, sys, pathlib, pickle
import numpy as np, pandas as pd, geopandas as gpd, shapely
from shapely import STRtree
csv, pkl, roads_dir, out = map(pathlib.Path, sys.argv[1:5]); out.mkdir(parents=True, exist_ok=True)
R = pickle.load(open(pkl, 'rb'))
df = pd.read_csv(csv, low_memory=False)
n0 = len(df)
df = df.dropna(subset=['latitude', 'longitude']); df = df[(df.latitude != 0) & df.longitude.between(-88.1, -87.4) & df.latitude.between(41.5, 42.2)]
df['ts'] = pd.to_datetime(df.crash_date)
df = df[df.ts.dt.year == 2023].sort_values('ts').reset_index(drop=True)
SEV = ['NO INDICATION OF INJURY', 'REPORTED, NOT EVIDENT', 'NONINCAPACITATING INJURY', 'INCAPACITATING INJURY', 'FATAL']
df['sev'] = df.most_severe_injury.map({s: i for i, s in enumerate(SEV)}).fillna(0).astype(int)
# snap to roads in UTM 16N
proj = lambda x, y: gpd.GeoSeries(gpd.points_from_xy(x, y), crs=4326).to_crs(26916)
pts = proj(df.longitude.values, df.latitude.values).values
und = np.where((R['rev'] == 0xFFFFFFFF) | (np.arange(len(R['rev'])) < R['rev'].astype(np.int64)))[0]   # one edge per undirected street
lines = gpd.GeoSeries([R['geoms'][i] for i in und], crs=4326).to_crs(26916).values
tree = STRtree(lines)
li = tree.nearest(pts)    # index of the nearest street for each point
dist = shapely.distance(pts, lines[li])
frac = shapely.line_locate_point(lines[li], pts, normalized=True)
eidx = und[li].astype(np.int64)
MAXD = 60.0
ok = dist <= MAXD
off = frac * R['length'][eidx]
edge = np.where(ok, eidx, 0xFFFFFFFF).astype('<u4'); offset = np.where(ok, off, 0).astype('<f4'); sd = np.clip(np.round(dist), 0, 255).astype('<u1')
t0 = pd.Timestamp('2023-01-01'); sec = ((df.ts - t0).dt.total_seconds()).astype('<u4').values
cols = {}
def add(name, arr, dtype, comp=1, **extra):
    arr = np.ascontiguousarray(np.asarray(arr).astype(dtype)); arr.tofile(out / f'{name}.bin')
    cols[name] = {'file': f'{name}.bin', 'dtype': np.dtype(dtype).name, 'components': comp, 'length': int(arr.size // comp), **extra}
add('position', np.column_stack([df.longitude, df.latitude]), '<f4', 2)
add('timestamp', sec, '<u4', unit='seconds since 2023-01-01T00:00:00 (Chicago local time as published; not UTC)')
add('severity', df.sev, '<u1', categories=['no injury', 'reported, not evident', 'non-incapacitating', 'incapacitating', 'fatal'])
add('injuries', df.injuries_total.fillna(0).clip(0, 255), '<u1', note='total persons injured')
add('fatalities', df.injuries_fatal.fillna(0).clip(0, 255), '<u1')
add('edgeIndex', edge, '<u4', note='index into chicago-roads edges (one representative direction of the street; use edgeReverse for the other); 4294967295 = no edge within 60 m')
add('edgeOffset', offset, '<f4', unit='m', note='distance along the edge from its source node (fraction of snapped polyline x edgeLength)')
add('snapDistance', sd, '<u1', unit='m', note='distance from crash point to edge, rounded, capped at 255')
tot_inj = int(df.injuries_total.fillna(0).sum()); fat = int(df.injuries_fatal.fillna(0).sum())
d = df.assign(hour=df.ts.dt.hour, dow=df.ts.dt.day_name(), edge=np.where(ok, eidx, -1))
byedge = d[d.edge >= 0].groupby('edge').size().sort_values(ascending=False)
top_i = int(byedge.index[0]); nm = d[d.edge == top_i].street_name.mode().iloc[0]
cause = df.prim_contributory_cause.value_counts()
streets = ", ".join(f"{k.title()} ({v:,})" for k, v in df.street_name.str.strip().value_counts().head(5).items())
story = [
    f"{len(df):,} crashes with coordinates in 2023 ({n0 - len(df)} dropped for missing/invalid location); {tot_inj:,} people injured, {fat} killed in {int((df.injuries_fatal.fillna(0) > 0).sum())} fatal crashes.",
    f"{ok.mean():.1%} snap within 60 m of a drivable edge (median snap {np.median(dist):.1f} m); the rest have no drivable edge nearby (e.g. lots, alleys, edges outside the strongly connected component) and carry edgeIndex 4294967295.",
    f"Busiest edge: {byedge.iloc[0]} crashes on one block of {nm}. Rush hour: {int(d.hour.value_counts().idxmax()):02d}:00 is the peak hour; busiest weekday {d.dow.value_counts().idxmax()}.",
    f"Most common primary cause after 'unable to determine'/'not applicable': {cause.drop(['UNABLE TO DETERMINE', 'NOT APPLICABLE'], errors='ignore').index[0].title()} ({int(cause.drop(['UNABLE TO DETERMINE', 'NOT APPLICABLE'], errors='ignore').iloc[0]):,}).",
    f"Severity mix: {(df.sev==0).mean():.0%} no injury; {(df.sev>=3).sum():,} incapacitating or fatal. Streets with most crashes: {streets}. Crashes are confined to the street graph, so a planar K-function over-states clustering that a network K corrects.",
]
b = [float(df.longitude.min()), float(df.latitude.min()), float(df.longitude.max()), float(df.latitude.max())]
man = {'id': 'chicago-crashes', 'version': 1, 'kind': 'points', 'count': int(len(df)), 'bbox': [round(x, 5) for x in b], 'crs': 'EPSG:4326', 'columns': cols,
  'properties': {'description': 'City of Chicago traffic crashes in 2023 (CPD-reported, crash_date in 2023) with position, time, severity, and a precomputed snap to chicago-roads (edgeIndex + edgeOffset) for network K-function / linear referencing.',
    'roadsDataset': 'chicago-roads', 'snapMaxDistanceMeters': MAXD,
    'licence': 'City of Chicago Data Portal Terms of Use (open public data)', 'storyNotes': story}}
(out / 'manifest.json').write_text(json.dumps(man, indent=1))
print('\n'.join(story)); print('bytes', sum(p.stat().st_size for p in out.iterdir()))
