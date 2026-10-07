"""OpenFlights airports + routes -> undirected airport-pair flows (see README.md)."""
import csv, json, os, sys, urllib.request
import numpy as np
from collections import defaultdict
HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
RAW = os.environ.get('SHOWCASE_RAW', '/private/tmp/claude-501/-Users-robertchristie-Documents-GitHub-deck-gl-master/85599621-aabc-441b-ab0f-35747be6f781/scratchpad/showcase/raw') + '/openflights'
OUT = os.path.join(APP, 'public/data/openflights')
BASE = 'https://raw.githubusercontent.com/jpatokal/openflights/master/data/'
os.makedirs(RAW, exist_ok=True); os.makedirs(OUT, exist_ok=True)
for f in ('airports.dat', 'routes.dat'):
    if not os.path.exists(f'{RAW}/{f}'):
        urllib.request.urlretrieve(BASE + f, f'{RAW}/{f}')

airports = {}
for r in csv.reader(open(f'{RAW}/airports.dat', encoding='utf-8')):
    iata = r[4]
    if len(iata) != 3 or r[12] != 'airport': continue
    airports[iata] = dict(id=r[0], name=r[1], city=r[2], country=r[3], lat=float(r[6]), lng=float(r[7]), alt=float(r[8]))

pairs = defaultdict(lambda: {'n': 0, 'air': set(), 'ab': 0, 'ba': 0})
for r in csv.reader(open(f'{RAW}/routes.dat', encoding='utf-8')):
    a, b = r[2], r[4]
    if a == b or a not in airports or b not in airports: continue
    k = (a, b) if a < b else (b, a)
    p = pairs[k]; p['n'] += 1; p['air'].add(r[0])
    if a < b: p['ab'] += 1
    else: p['ba'] += 1

used = sorted({c for k in pairs for c in k})
idx = {c: i for i, c in enumerate(used)}
n = len(used)
pos = np.array([[airports[c]['lng'], airports[c]['lat']] for c in used], dtype='<f4')
deg = np.zeros(n, '<u2'); rts = np.zeros(n, '<u4')
keys = sorted(pairs)
org = np.array([idx[a] for a, b in keys], '<u4'); dst = np.array([idx[b] for a, b in keys], '<u4')
cnt = np.array([pairs[k]['n'] for k in keys], '<u2')
airc = np.array([len(pairs[k]['air']) for k in keys], '<u2')
dirab = np.array([pairs[k]['ab'] for k in keys], '<u2'); dirba = np.array([pairs[k]['ba'] for k in keys], '<u2')
for i, k in enumerate(keys):
    for c in k: deg[idx[c]] += 1; rts[idx[c]] += cnt[i]
# great-circle length km
def gc(a, b):
    la1, lo1, la2, lo2 = map(np.radians, (a[1], a[0], b[1], b[0]))
    h = np.sin((la2-la1)/2)**2 + np.cos(la1)*np.cos(la2)*np.sin((lo2-lo1)/2)**2
    return 2*6371.0*np.arcsin(np.sqrt(h))
dist = np.array([gc(pos[o], pos[d]) for o, d in zip(org, dst)], '<f4')

files = {'position.bin': pos, 'degree.bin': deg, 'routeCount.bin': rts, 'origin.bin': org, 'destination.bin': dst,
         'count.bin': cnt, 'airlineCount.bin': airc, 'directedForward.bin': dirab, 'directedBackward.bin': dirba, 'distanceKm.bin': dist}
for f, a in files.items(): a.tofile(f'{OUT}/{f}')
with open(f'{OUT}/airports.csv', 'w', newline='', encoding='utf-8') as fh:
    w = csv.writer(fh); w.writerow(['index', 'iata', 'name', 'city', 'country'])
    for i, c in enumerate(used): w.writerow([i, c, airports[c]['name'], airports[c]['city'], airports[c]['country']])
top = np.argsort(-deg)[:8]
def col(f, dt, comp, ln, **kw): return {'file': f, 'dtype': dt, 'components': comp, 'length': ln, **kw}
m = {
  'id': 'openflights', 'version': 1, 'kind': 'flows', 'count': len(keys),
  'bbox': [float(pos[:, 0].min()), float(pos[:, 1].min()), float(pos[:, 0].max()), float(pos[:, 1].max())], 'crs': 'EPSG:4326',
  'columns': {
    'locations': col('position.bin', 'float32', 2, n),
    'degree': col('degree.bin', 'uint16', 1, n, description='distinct connected airports'),
    'routeCount': col('routeCount.bin', {np.dtype('uint16'): 'uint16', np.dtype('uint32'): 'uint32'}[rts.dtype], 1, n, description='airline-route records touching the airport'),
    'origin': col('origin.bin', 'uint32', 1, len(keys), description='index into locations; origin IATA < destination IATA (undirected pair)'),
    'destination': col('destination.bin', 'uint32', 1, len(keys)),
    'count': col('count.bin', 'uint16', 1, len(keys), description='airline-route records on the pair, both directions'),
    'airlineCount': col('airlineCount.bin', 'uint16', 1, len(keys), description='distinct airlines serving the pair'),
    'directedForward': col('directedForward.bin', 'uint16', 1, len(keys), description='records origin->destination'),
    'directedBackward': col('directedBackward.bin', 'uint16', 1, len(keys), description='records destination->origin'),
    'distanceKm': col('distanceKm.bin', 'float32', 1, len(keys), description='great-circle length'),
  },
  'table': {'file': 'airports.csv', 'columns': ['index', 'iata', 'name', 'city', 'country'], 'note': 'row i describes locations[i]'},
  'properties': {
    'airportCount': n, 'edgeCount': len(keys),
    'dataVintage': 'OpenFlights data is community-maintained and largely frozen around June 2014',
    'attribution': 'OpenFlights.org (airports.dat, routes.dat), Open Database License 1.0',
    'storyNotes': [
      'Top hubs by distinct destinations: ' + ', '.join(f'{used[i]} ({deg[i]})' for i in top) + '.',
      f'{n} airports and {len(keys)} undirected pairs; the median airport has degree {int(np.median(deg))} while the busiest has {int(deg.max())} - a heavy-tailed hub-and-spoke network.',
      f'Longest pair: {used[org[dist.argmax()]]}-{used[dst[dist.argmax()]]} at {dist.max():.0f} km; {(dist<500).mean()*100:.0f}% of pairs are under 500 km.',
      f'{int((deg==1).sum())} airports ({(deg==1).mean()*100:.0f}%) are leaves with a single connection - the spokes that network coarsening collapses first.',
      'Data is frozen around 2014: it has no post-2014 routes or carriers, so treat it as a structural network, not current traffic.']
  }}
json.dump(m, open(f'{OUT}/manifest.json', 'w'), indent=1)
tot = sum(os.path.getsize(f'{OUT}/{f}') for f in os.listdir(OUT)); print('airports', n, 'pairs', len(keys), 'bytes', tot)
print(m['properties']['storyNotes'])
