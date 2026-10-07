"""BTS On-Time Reporting Carrier data, July 2023 -> directed US airport-pair flight counts (see README.md)."""
import csv, io, json, os, urllib.request, zipfile
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
ROOT = os.environ.get('SHOWCASE_RAW', '/private/tmp/claude-501/-Users-robertchristie-Documents-GitHub-deck-gl-master/85599621-aabc-441b-ab0f-35747be6f781/scratchpad/showcase/raw')
RAW = ROOT + '/us-airline-flows'; OUT = os.path.join(APP, 'public/data/us-airline-flows'); os.makedirs(OUT, exist_ok=True); os.makedirs(RAW, exist_ok=True)
ZIP = f'{RAW}/On_Time_Reporting_Carrier_On_Time_Performance_1987_present_2023_7.zip'
if not os.path.exists(ZIP): urllib.request.urlretrieve('https://www.transtats.bts.gov/PREZIP/' + os.path.basename(ZIP), ZIP)
zf = zipfile.ZipFile(ZIP); name = [n for n in zf.namelist() if n.endswith('.csv')][0]
d = pd.read_csv(zf.open(name), usecols=['Origin', 'Dest', 'Distance', 'DepDelay', 'Cancelled', 'DayOfWeek'], low_memory=False)
total = len(d)
apt = {}
for r in csv.reader(open(ROOT + '/openflights/airports.dat', encoding='utf-8')):
    if len(r[4]) == 3 and r[12] == 'airport': apt[r[4]] = (float(r[7]), float(r[6]), r[1], r[2], r[3])
d = d[d.Origin.isin(apt) & d.Dest.isin(apt)]
d['cx'] = d.Cancelled.astype(int); d['late15'] = (d.DepDelay.fillna(0) >= 15).astype(int)
g = d.groupby(['Origin', 'Dest']).agg(flights=('Origin', 'size'), cancelled=('cx', 'sum'), late=('late15', 'sum'), delay=('DepDelay', 'mean'), dist=('Distance', 'mean')).reset_index()
used = sorted(set(g.Origin) | set(g.Dest)); idx = {c: i for i, c in enumerate(used)}; n = len(used)
pos = np.array([apt[c][:2] for c in used], '<f4')
org = g.Origin.map(idx).values.astype('<u4'); dst = g.Dest.map(idx).values.astype('<u4')
cnt = g.flights.values.astype('<u4'); can = g.cancelled.values.astype('<u2'); late = g.late.values.astype('<u2')
dly = g['delay'].fillna(0).values.astype('<f4'); dist = (g.dist.values * 1.609344).astype('<f4')
outd = np.zeros(n, '<u4'); ind = np.zeros(n, '<u4'); np.add.at(outd, org, cnt); np.add.at(ind, dst, cnt)
tot = (outd + ind).astype('<u4')
files = {'position.bin': pos, 'origin.bin': org, 'destination.bin': dst, 'count.bin': cnt, 'cancelled.bin': can, 'lateDepartures.bin': late, 'meanDepDelay.bin': dly, 'distanceKm.bin': dist, 'airportFlights.bin': tot}
for f, a in files.items(): a.tofile(f'{OUT}/{f}')
with open(f'{OUT}/airports.csv', 'w', newline='', encoding='utf-8') as fh:
    w = csv.writer(fh); w.writerow(['index', 'iata', 'name', 'city', 'country'])
    for i, c in enumerate(used): w.writerow([i, c, apt[c][2], apt[c][3], apt[c][4]])
col = lambda f, dt, comp, ln, **kw: {'file': f, 'dtype': dt, 'components': comp, 'length': ln, **kw}
topP = g.sort_values('flights', ascending=False).head(5); topA = np.argsort(-tot)[:6]
# Hawaii/Alaska share
m = {'id': 'us-airline-flows', 'version': 1, 'kind': 'flows', 'count': len(g),
 'bbox': [float(pos[:, 0].min()), float(pos[:, 1].min()), float(pos[:, 0].max()), float(pos[:, 1].max())], 'crs': 'EPSG:4326',
 'columns': {'locations': col('position.bin', 'float32', 2, n),
  'origin': col('origin.bin', 'uint32', 1, len(g), description='directed flow: index into locations'), 'destination': col('destination.bin', 'uint32', 1, len(g)),
  'count': col('count.bin', 'uint32', 1, len(g), unit='scheduled flights in July 2023 (weight)'),
  'cancelled': col('cancelled.bin', 'uint16', 1, len(g), unit='cancelled flights (included in count)'),
  'lateDepartures': col('lateDepartures.bin', 'uint16', 1, len(g), unit='flights departing >= 15 min late'),
  'meanDepDelay': col('meanDepDelay.bin', 'float32', 1, len(g), unit='minutes'),
  'distanceKm': col('distanceKm.bin', 'float32', 1, len(g)),
  'airportFlights': col('airportFlights.bin', 'uint32', 1, n, description='in + out flights per airport')},
 'table': {'file': 'airports.csv', 'columns': ['index', 'iata', 'name', 'city', 'country'], 'note': 'row i describes locations[i]'},
 'properties': {'month': '2023-07', 'weightMeaning': 'flights (not passengers): BTS T-100 passenger tables need an interactive download, so this uses the On-Time Reporting Carrier file for the largest US carriers',
  'coverage': f'{len(d):,} of {total:,} flight records matched to airport coordinates (OpenFlights)',
  'attribution': 'U.S. Bureau of Transportation Statistics, On-Time Reporting Carrier On-Time Performance (public domain); airport coordinates and names from OpenFlights (ODbL 1.0)',
  'storyNotes': [f'{len(g):,} directed airport pairs, {n} airports, {len(d):,} flights in July 2023.',
   'Busiest pairs: ' + ', '.join(f'{r.Origin}->{r.Dest} ({r.flights})' for r in topP.itertuples()) + '.',
   'Busiest airports (in+out flights): ' + ', '.join(f'{used[i]} ({tot[i]:,})' for i in topA) + '.',
   f'Mean departure delay is {d.DepDelay.mean():.1f} min and {d.cx.mean()*100:.1f}% of flights were cancelled; worst mean delay among pairs with >=100 flights: ' + (lambda s: f'{s.Origin}->{s.Dest} ({s.delay:.0f} min)')(g[g.flights >= 100].sort_values('delay').iloc[-1]) + '.',
   f'Weights are skewed: the top 10% of pairs carry {g.flights.sort_values(ascending=False).head(len(g)//10).sum()/g.flights.sum()*100:.0f}% of flights, a classic case for flow aggregation and edge bundling.']}}
json.dump(m, open(f'{OUT}/manifest.json', 'w'), indent=1)
print(n, len(g), {f: os.path.getsize(f'{OUT}/{f}') for f in os.listdir(OUT)}); print(m['properties']['storyNotes'])
