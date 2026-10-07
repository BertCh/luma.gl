"""Build chicago-taxi-od from the monthly Socrata aggregates fetched by fetch.py.
usage: build.py RAW_DIR COMMUNITY_AREAS_GEOJSON OUT_DIR"""
import json, sys, glob, pathlib
import numpy as np, pandas as pd, geopandas as gpd
raw, ca_path, out = map(pathlib.Path, sys.argv[1:4]); out.mkdir(parents=True, exist_ok=True)
ca = gpd.read_file(ca_path).to_crs(4326)
ca['num'] = ca['area_numbe'].astype(int); ca = ca.sort_values('num').reset_index(drop=True)
assert len(ca) == 77
cen = ca.to_crs(26916).geometry.centroid.to_crs(4326)
loc = np.column_stack([cen.x, cen.y]).astype('<f4')
names = ca['community'].str.title().tolist()
def w(name, arr):
    arr.tofile(out / name); return name
od = pd.concat([pd.read_json(f) for f in sorted(glob.glob(str(raw / 'od-*.json')))])
total_trips = int(od.n.sum())
od = od.dropna(subset=['p', 'd']); od = od[(od.p.between(1, 77)) & (od.d.between(1, 77))]
od['p'] = od.p.astype(int) - 1; od['d'] = od.d.astype(int) - 1
od['daytype'] = np.where(od.dow.isin([0, 6]), 1, 0)   # 0 weekday, 1 weekend (Sat/Sun)
for c in ('fare', 'secs', 'miles'): od[c + '_w'] = od[c] * od.n
kept = int(od.n.sum())
def agg(keys):
    g = od.groupby(keys, as_index=False)[['n', 'fare_w', 'secs_w', 'miles_w']].sum()
    g['fare'] = g.fare_w / g.n; g['secs'] = g.secs_w / g.n; g['miles'] = g.miles_w / g.n
    return g.sort_values(keys).reset_index(drop=True)
tot = agg(['p', 'd']); hr = agg(['p', 'd', 'daytype', 'h'])
cols = {}
def add(name, arr, dtype, comp=1, note=None):
    arr = np.ascontiguousarray(arr.astype(dtype)); fn = w(name + '.bin', arr)
    cols[name] = {'file': fn, 'dtype': np.dtype(dtype).name, 'components': comp, 'length': int(arr.size // comp)}
    if note: cols[name]['note'] = note
add('locations', loc, '<f4', 2, 'community-area centroids, index = community area number - 1 (see properties.areaNames)')
add('origin', tot.p.values, '<u4', note='pickup community area index (full year, all hours)')
add('destination', tot.d.values, '<u4', note='dropoff community area index')
add('count', tot.n.values, '<u4', note='trips in 2023')
add('meanFare', tot.fare.values, '<f4', note='mean fare, USD (excludes tips/extras)')
add('meanSeconds', tot.secs.values, '<f4', note='mean trip duration, s')
add('meanMiles', tot.miles.values, '<f4', note='mean trip distance, miles')
add('hourlyOrigin', hr.p.values, '<u4', note='sliced flows: one row per (origin, destination, daytype, hour)')
add('hourlyDestination', hr.d.values, '<u4')
add('hourlyHour', hr.h.values, '<u1', note='pickup hour of day 0-23 (Chicago local time, as published)')
add('hourlyDaytype', hr.daytype.values, '<u1', note='0 = weekday (Mon-Fri), 1 = weekend (Sat-Sun)')
add('hourlyCount', hr.n.values, '<u4')
add('hourlyMeanFare', hr.fare.values, '<f4')
add('hourlyMeanSeconds', hr.secs.values, '<f4')
# daily pickups per area
dd = pd.concat([pd.read_json(f) for f in sorted(glob.glob(str(raw / 'daily-*.json')))])
dd['day'] = pd.to_datetime(dd['day'].astype(str).str[:10]); dd = dd.dropna(subset=['p']); dd = dd[dd.p.between(1, 77)]
dd['doy'] = (dd.day - pd.Timestamp('2023-01-01')).dt.days; dd['p'] = dd.p.astype(int) - 1
assert dd.doy.min() == 0 and dd.doy.max() == 364
M = np.zeros((365, 77), '<u4'); F = np.zeros((365, 77), '<f4')
M[dd.doy.values, dd.p.values] = dd.n.values; F[dd.doy.values, dd.p.values] = dd.fare.values
add('dailyPickups', M.ravel(), '<u4', 77, 'row-major [dayOfYear 0..364][area 0..76]; day 0 = 2023-01-01 (Sunday); a column of 77 values per row, flatten length 365*77')
cols['dailyPickups']['length'] = 365
add('dailyMeanFare', F.ravel(), '<f4', 77, 'same layout as dailyPickups')
cols['dailyMeanFare']['length'] = 365
daily_tot = M.sum(1); area_tot = M.sum(0)
dates = pd.date_range('2023-01-01', periods=365)
bd = int(daily_tot.argmax()); ld = int(daily_tot.argmin())
hh = hr.groupby('h').n.sum(); hwk = hr[hr.daytype == 0].groupby('h').n.sum(); hwe = hr[hr.daytype == 1].groupby('h').n.sum()
top = tot.sort_values('n', ascending=False).head(3)
tops = [f"{names[int(r.p)]} -> {names[int(r.d)]}: {int(r.n):,} trips" for r in top.itertuples()]
big = tot[tot.n > 2000].sort_values("fare", ascending=False).iloc[0]
fare_top = f"{names[int(big.p)]} -> {names[int(big.d)]} (${big.fare:.0f}, {int(big.n):,} trips)"
selfshare = tot[tot.p == tot.d].n.sum() / tot.n.sum()
story = [
    f"{total_trips:,} taxi trips started in 2023 ({kept:,} / {kept/total_trips:.0%} have both pickup and dropoff community areas; the rest are suppressed by the city for privacy or outside Chicago).",
    f"Busiest pickup area: {names[int(area_tot.argmax())]} ({int(area_tot.max()):,} trips); the Near North Side/Loop/O'Hare trio dominates the OD matrix. Top flows: " + '; '.join(tops),
    f"Hourly rhythm: weekday peak at {int(hwk.idxmax()):02d}:00, weekend peak at {int(hwe.idxmax()):02d}:00; quietest hour is {int(hh.idxmin()):02d}:00.",
    f"Busiest day {dates[bd].date()} ({int(daily_tot[bd]):,} trips, {dates[bd].day_name()}); quietest {dates[ld].date()} ({int(daily_tot[ld]):,}, {dates[ld].day_name()}).",
    f"{selfshare:.0%} of mapped trips stay inside one community area. Highest mean fare among flows with >2,000 trips: {fare_top}.",
]
b = ca.total_bounds
manifest = {'id': 'chicago-taxi-od', 'version': 1, 'kind': 'flows', 'count': int(len(tot)),
    'bbox': [round(float(b[0]), 5), round(float(b[1]), 5), round(float(b[2]), 5), round(float(b[3]), 5)], 'crs': 'EPSG:4326', 'columns': cols,
    'properties': {'description': 'Chicago Taxi Trips 2023 (Socrata wrvz-psew) aggregated server-side then locally. Primary flow columns (origin/destination/count/mean*) are full-year OD between the 77 community areas; hourly* columns are the same OD sliced by weekday/weekend x pickup hour; dailyPickups/dailyMeanFare are 365 x 77 pickup matrices for temporal reduction / calendar buckets. Trips without both community areas are dropped (counts in storyNotes).',
        'areaNames': names, 'areaNumbers': ca.num.tolist(), 'hourlyCount_rows': int(len(hr)),
        'licence': 'City of Chicago Data Portal Terms of Use (public data, no redistribution restriction); data provided by taxi companies via the City of Chicago',
        'storyNotes': story}}
(out / 'manifest.json').write_text(json.dumps(manifest, indent=1))
print('flows', len(tot), 'hourly rows', len(hr), 'bytes', sum(p.stat().st_size for p in out.iterdir()))
print('\n'.join(story))
