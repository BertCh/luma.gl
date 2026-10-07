"""LifeWatch/INBO lesser black-backed gulls (Zenodo 5068540, CC0): autumn 2015 migration, hourly -> trajectories (see README.md)."""
import json, os, urllib.request
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
RAW = os.environ.get('SHOWCASE_RAW', '/private/tmp/claude-501/-Users-robertchristie-Documents-GitHub-deck-gl-master/85599621-aabc-441b-ab0f-35747be6f781/scratchpad/showcase/raw') + '/gull-migration'
OUT = os.path.join(APP, 'public/data/gull-migration'); os.makedirs(OUT, exist_ok=True); os.makedirs(RAW, exist_ok=True)
Z = 'https://zenodo.org/api/records/5068540/files/'
for f in ('LBBG_ZEEBRUGGE-gps-2015.csv.gz', 'LBBG_ZEEBRUGGE-reference-data.csv'):
    if not os.path.exists(f'{RAW}/{f}'): urllib.request.urlretrieve(Z + f + '/content', f'{RAW}/{f}')
T0 = pd.Timestamp('2015-07-15T00:00:00'); T1 = pd.Timestamp('2015-12-01T00:00:00')
d = pd.read_csv(f'{RAW}/LBBG_ZEEBRUGGE-gps-2015.csv.gz', usecols=['timestamp', 'location-long', 'location-lat', 'individual-local-identifier', 'ground-speed', 'manually-marked-outlier', 'import-marked-outlier', 'visible'])
d = d.rename(columns={'timestamp': 'ts', 'location-long': 'lng', 'location-lat': 'lat', 'individual-local-identifier': 'bird', 'ground-speed': 'spd'})
d = d[(d['manually-marked-outlier'] == False) & (d['import-marked-outlier'] == False) & (d.visible == True)].dropna(subset=['lng', 'lat'])
d['ts'] = pd.to_datetime(d.ts); d = d[(d.ts >= T0) & (d.ts < T1)].copy()
d['sec'] = (d.ts - T0).dt.total_seconds().astype('int64'); d['hour'] = d.sec // 3600
ref = pd.read_csv(f'{RAW}/LBBG_ZEEBRUGGE-reference-data.csv').drop_duplicates('animal-id').set_index('animal-id')
SEX = ['f', 'm', 'u']
birds = []
for b, g in d.sort_values('sec').groupby('bird'):
    g = g.drop_duplicates('hour')  # first fix of each hour
    if len(g) < 500: continue
    birds.append((b, g))
birds.sort(key=lambda x: x[0])
n = len(birds)
offs = np.r_[0, np.cumsum([len(g) for _, g in birds])].astype('<u4')
cat = lambda c: np.concatenate([c(g) for _, g in birds])
vert = cat(lambda g: g[['lng', 'lat']].values).astype('<f4'); tim = cat(lambda g: g.sec.values).astype('<u4')
spd = cat(lambda g: g.spd.fillna(0).values).astype('<f4')
sex = np.array([SEX.index(ref['animal-sex'].get(b, 'u')) if ref['animal-sex'].get(b, 'u') in SEX else 2 for b, _ in birds], 'u1')
files = {'pathOffsets.bin': offs, 'vertices.bin': vert, 'timestamp.bin': tim, 'speed.bin': spd, 'sex.bin': sex}
for f, a in files.items(): a.tofile(f'{OUT}/{f}')
npts = len(tim)
def col(f, dt, comp, ln, **kw): return {'file': f, 'dtype': dt, 'components': comp, 'length': ln, **kw}
minlat = [(float(g.lat.min()), b) for b, g in birds]; far = sorted(minlat)[:3]
south = sum(1 for m, _ in minlat if m < 38.0); africa = sum(1 for m, _ in minlat if m < 30.0)
dep = []
for b, g in birds:
    s = g[g.lat < 45.0]
    if len(s): dep.append(s.ts.min())
dep = pd.Series(dep)
m = {'id': 'gull-migration', 'version': 1, 'kind': 'trajectories', 'count': n,
 'bbox': [float(vert[:, 0].min()), float(vert[:, 1].min()), float(vert[:, 0].max()), float(vert[:, 1].max())], 'crs': 'EPSG:4326',
 'columns': {
  'pathOffsets': col('pathOffsets.bin', 'uint32', 1, n + 1),
  'vertices': col('vertices.bin', 'float32', 2, npts),
  'timestamp': col('timestamp.bin', 'uint32', 1, npts, unit='seconds since 2015-07-15T00:00:00Z'),
  'speed': col('speed.bin', 'float32', 1, npts, unit='m/s (GPS ground speed, 0 where missing)'),
  'sex': col('sex.bin', 'uint8', 1, n, categories=SEX)},
 'properties': {'birdIds': [b for b, _ in birds], 'birdIdNote': 'bird id i = properties.birdIds[i], Movebank individual-local-identifier',
  'window': '2015-07-15 to 2015-11-30 (autumn migration)', 'sampling': 'first GPS fix of each UTC hour; marked outliers removed',
  'lifeStage': 'all tracked birds are adults', 'study': 'LBBG_ZEEBRUGGE (UvA-BiTS GPS tags), Belgium/Netherlands breeding colonies',
  'doi': 'https://doi.org/10.5281/zenodo.5068540',
  'attribution': 'Stienen E, Desmet P, Aelterman B, et al. LifeWatch GPS tracking of Lesser Black-backed gulls (LBBG_ZEEBRUGGE), INBO / Research Institute for Nature and Forest and Flanders Marine Institute, Zenodo, CC0 1.0',
  'storyNotes': [
    f'{n} adult gulls from the Zeebrugge/Belgian-Dutch North Sea coast colonies, hourly fixes, 15 Jul - 30 Nov 2015.',
    f'{south} birds reach south of 38 N (Iberia/Morocco) and {africa} cross south of 30 N into the Sahara/West Africa; furthest south ' + ', '.join(f'{b} at {la:.1f} N' for la, b in far) + '.',
    (f'Southbound departures (first fix south of 45 N) spread from {dep.min():%d %b} to {dep.max():%d %b}, median {dep.median():%d %b}.' if len(dep) else 'Departure dates unavailable.'),
    f'On 30 Nov, {sum(1 for _, g in birds if g.lat.iloc[-1] < 41)} of {n} birds are south of 41 N; {sum(1 for _, g in birds if 36.5 <= g.lat.iloc[-1] <= 37.5 and -8.0 <= g.lng.iloc[-1] <= -5.5)} sit in the Gulf of Cadiz / Guadalquivir area of SW Spain and {sum(1 for _, g in birds if g.lat.iloc[-1] > 49)} are still in the Channel / North Sea (resident or late migrants).',
    'All birds are adults (no juveniles) tagged at known colonies, so start points are shared and routes can be compared by similarity clustering.']}}
json.dump(m, open(f'{OUT}/manifest.json', 'w'), indent=1)
print(n, npts, {f: os.path.getsize(f'{OUT}/{f}') for f in os.listdir(OUT)}); print(m['properties']['storyNotes'])
