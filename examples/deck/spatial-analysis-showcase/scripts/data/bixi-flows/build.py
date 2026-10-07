#!/usr/bin/env python3
"""Build public/data/bixi-flows from the BIXI Montreal 2024 open-data trip CSV (August 2024).

usage: python3 -I build.py <DonneesOuvertes2024.zip> <out dir>

Rides shorter than 60 s (failed docks) and rides with a missing station are dropped.
Times are converted to America/Montreal local time before bucketing by hour and weekday/weekend.
"""
import csv, collections, io, json, struct, sys, zipfile
from array import array
from datetime import datetime
from zoneinfo import ZoneInfo

TZ = ZoneInfo('America/Montreal')
PAIR_MIN = 3        # full-month pair table keeps pairs with at least this many rides
HOURLY_MIN = 10     # hourly rows keep pairs with at least this many rides; the rest become residual rows

zip_path, out_dir = sys.argv[1], sys.argv[2]
import os
os.makedirs(out_dir, exist_ok=True)

lo = datetime(2024, 8, 1, tzinfo=TZ).timestamp() * 1000
hi = datetime(2024, 9, 1, tzinfo=TZ).timestamp() * 1000

pairs = collections.Counter()            # (origin, destination, daytype, hour) -> rides
positions = collections.defaultdict(list)
boroughs = collections.defaultdict(collections.Counter)
month_hours = collections.Counter()
rows = kept = short = 0

with zipfile.ZipFile(zip_path) as z:
    name = z.namelist()[0]
    with z.open(name) as raw:
        reader = csv.reader(io.TextIOWrapper(raw, encoding='utf-8', newline=''))
        header = next(reader)
        ix = {h: i for i, h in enumerate(header)}
        for r in reader:
            rows += 1
            try:
                start = float(r[ix['STARTTIMEMS']]); end = float(r[ix['ENDTIMEMS']])
            except ValueError:
                continue
            if not (lo <= start < hi):
                continue
            if end - start < 60000:
                short += 1
                continue
            a, b = r[ix['STARTSTATIONNAME']], r[ix['ENDSTATIONNAME']]
            if not a or not b or not r[ix['STARTSTATIONARRONDISSEMENT']] or not r[ix['ENDSTATIONARRONDISSEMENT']]:
                continue
            kept += 1
            local = datetime.fromtimestamp(start / 1000, TZ)
            pairs[(a, b, 0 if local.weekday() < 5 else 1, local.hour)] += 1
            month_hours[int((start - lo) // 3600000)] += 1
            for nm, la, lg, bo in (
                (a, 'STARTSTATIONLATITUDE', 'STARTSTATIONLONGITUDE', 'STARTSTATIONARRONDISSEMENT'),
                (b, 'ENDSTATIONLATITUDE', 'ENDSTATIONLONGITUDE', 'ENDSTATIONARRONDISSEMENT'),
            ):
                try:
                    positions[nm].append((float(r[ix[lg]]), float(r[ix[la]])))
                except ValueError:
                    pass
                boroughs[nm][r[ix[bo]]] += 1
print('rows', rows, 'kept', kept, 'short', short, file=sys.stderr)

# stations: median position, modal borough, sorted by (borough, name)
stations = {}
for nm, pts in positions.items():
    xs = sorted(p[0] for p in pts); ys = sorted(p[1] for p in pts)
    stations[nm] = (xs[len(xs) // 2], ys[len(ys) // 2], boroughs[nm].most_common(1)[0][0])
names = sorted(stations, key=lambda n: (stations[n][2], n))
index = {n: i for i, n in enumerate(names)}
N = len(names)
borough_names = sorted({stations[n][2] for n in names})
borough_index = {b: i for i, b in enumerate(borough_names)}

pair_month = collections.Counter()
for (a, b, d, h), n in pairs.items():
    pair_month[(a, b)] += n
total_rides = sum(pairs.values())
self_rides = sum(n for (a, b), n in pair_month.items() if a == b)

# full-month pair table (no self pairs), most frequent first
month_pairs = sorted(((n, a, b) for (a, b), n in pair_month.items() if a != b and n >= PAIR_MIN), reverse=True)
origin = array('H', (index[a] for n, a, b in month_pairs))
destination = array('H', (index[b] for n, a, b in month_pairs))
count = array('H', (min(n, 65535) for n, a, b in month_pairs))

# hourly rows for frequent pairs; everything else is folded into residual rows to/from the sentinel zone N
keep = {k for k, n in pair_month.items() if n >= HOURLY_MIN and k[0] != k[1]}
h_origin, h_destination, h_hour, h_daytype, h_count = array('H'), array('H'), array('B'), array('B'), array('B')
resid_out = collections.Counter(); resid_in = collections.Counter()
for (a, b, d, h), n in sorted(pairs.items(), key=lambda kv: (index[kv[0][0]], index[kv[0][1]], kv[0][2], kv[0][3])):
    if a == b:
        continue
    if (a, b) in keep:
        h_origin.append(index[a]); h_destination.append(index[b]); h_hour.append(h); h_daytype.append(d); h_count.append(n)
    else:
        resid_out[(index[a], d, h)] += n
        resid_in[(index[b], d, h)] += n
r_origin, r_destination, r_hour, r_daytype, r_count = array('H'), array('H'), array('B'), array('B'), array('H')
for (s, d, h), n in sorted(resid_out.items()):
    r_origin.append(s); r_destination.append(N); r_hour.append(h); r_daytype.append(d); r_count.append(n)
for (s, d, h), n in sorted(resid_in.items()):
    r_origin.append(N); r_destination.append(s); r_hour.append(h); r_daytype.append(d); r_count.append(n)

departures = array('I', [0] * N); arrivals = array('I', [0] * N)
for (a, b, d, h), n in pairs.items():
    if a != b:
        departures[index[a]] += n; arrivals[index[b]] += n
locations = array('f')
for n in names:
    locations.extend((stations[n][0], stations[n][1]))
borough = array('B', (borough_index[stations[n][2]] for n in names))
month = array('I', (month_hours.get(i, 0) for i in range(744)))

columns = {}
def put(key, arr, dtype, components=1, **extra):
    fn = key + '.bin'
    with open(os.path.join(out_dir, fn), 'wb') as f:
        f.write(arr.tobytes())
    columns[key] = {'file': fn, 'dtype': dtype, 'components': components, 'length': len(arr) // components, **extra}
    return len(arr) * arr.itemsize

sizes = 0
sizes += put('locations', locations, 'float32', 2, note='station positions (median reported lng, lat); index = station index')
sizes += put('origin', origin, 'uint16', note='origin station index, pairs with at least %d rides in August 2024, most frequent first, no same-station trips' % PAIR_MIN)
sizes += put('destination', destination, 'uint16', note='destination station index')
sizes += put('count', count, 'uint16', note='rides in August 2024')
sizes += put('hourlyOrigin', h_origin, 'uint16', note='sliced rows for pairs with at least %d rides: one row per (origin, destination, daytype, hour)' % HOURLY_MIN)
sizes += put('hourlyDestination', h_destination, 'uint16')
sizes += put('hourlyHour', h_hour, 'uint8', note='start hour of day 0-23, America/Montreal local time')
sizes += put('hourlyDaytype', h_daytype, 'uint8', note='0 = weekday (Mon-Fri), 1 = weekend (Sat-Sun)')
sizes += put('hourlyCount', h_count, 'uint8', note='rides in that slice')
sizes += put('residualOrigin', r_origin, 'uint16', note='rides of rarer pairs, folded per station: (station -> sentinel zone N) departures and (sentinel -> station) arrivals; adding them to the hourly rows makes every station total exact')
sizes += put('residualDestination', r_destination, 'uint16')
sizes += put('residualHour', r_hour, 'uint8')
sizes += put('residualDaytype', r_daytype, 'uint8')
sizes += put('residualCount', r_count, 'uint16')
sizes += put('stationBorough', borough, 'uint8', categories=borough_names, note='borough or municipality of the station, as published by BIXI')
sizes += put('stationDepartures', departures, 'uint32', note='rides starting here (to another station)')
sizes += put('stationArrivals', arrivals, 'uint32', note='rides ending here (from another station)')
sizes += put('monthHourly', month, 'uint32', note='rides per hour, 744 hours from 2024-08-01 00:00 local')

lons = [stations[n][0] for n in names]; lats = [stations[n][1] for n in names]
manifest = {
    'id': 'bixi-flows',
    'version': 1,
    'kind': 'flows',
    'count': len(origin),
    'bbox': [min(lons), min(lats), max(lons), max(lats)],
    'crs': 'EPSG:4326',
    'columns': columns,
    'properties': {
        'description': 'BIXI Montreal trips, August 2024: station-to-station flows, sliced by weekday/weekend x start hour. Hourly rows hold pairs with at least %d rides; rarer pairs are folded into residual rows so every station total is exact.' % HOURLY_MIN,
        'stationNames': names,
        'stationCount': N,
        'sentinelZone': N,
        'totalRides': total_rides,
        'sameStationRides': self_rides,
        'droppedShortRides': short,
        'pairMin': PAIR_MIN,
        'hourlyMin': HOURLY_MIN,
        'monthStart': '2024-08-01T00:00:00-04:00',
        'timezone': 'America/Montreal',
        'source': {'url': 'https://bixi.com/en/open-data/', 'file': 'DonneesOuvertes2024_010203040506070809101112.zip'},
    },
}
with open(os.path.join(out_dir, 'manifest.json'), 'w') as f:
    json.dump(manifest, f, ensure_ascii=False, indent=1)
print(json.dumps({'stations': N, 'pairs': len(origin), 'hourlyRows': len(h_origin), 'residualRows': len(r_origin), 'boroughs': len(borough_names), 'bytes': sizes, 'rides': total_rides}))
