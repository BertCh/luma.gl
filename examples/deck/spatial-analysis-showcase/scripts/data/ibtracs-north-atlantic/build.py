#!/usr/bin/env python3
"""Build public/data/ibtracs-north-atlantic from the raw NOAA IBTrACS North Atlantic CSV.

Usage: python3 -I build.py <ibtracs.NA.list.v04r01.csv> <out-dir>
Keeps seasons 1980-2025 (satellite era, complete seasons), 6-hourly synoptic fixes
(00/06/12/18 UTC) of the 'main' track of each storm.
"""
import csv, json, struct, sys, datetime, os, array

SRC, OUT = sys.argv[1], sys.argv[2]
FIRST_SEASON, LAST_SEASON = 1980, 2025
ORIGIN = datetime.datetime(1980, 1, 1)
NATURES = ['TS', 'ET', 'DS', 'SS', 'NR']


def num(text):
    text = text.strip()
    try:
        return float(text)
    except ValueError:
        return None


def category(wind):
    # Saffir-Simpson from 1-minute sustained wind in knots; 0 = tropical depression, 1 = tropical storm
    if wind < 34: return 0
    if wind < 64: return 1
    if wind < 83: return 2
    if wind < 96: return 3
    if wind < 113: return 4
    if wind < 137: return 5
    return 6


storms = {}
with open(SRC, newline='') as handle:
    reader = csv.DictReader(handle)
    next(reader)  # units row
    for row in reader:
        season = int(row['SEASON'])
        if season < FIRST_SEASON or season > LAST_SEASON or row['TRACK_TYPE'] != 'main':
            continue
        iso = row['ISO_TIME']
        if iso[14:16] != '00' or int(iso[11:13]) % 6:
            continue
        lat, lon = num(row['LAT']), num(row['LON'])
        if lat is None or lon is None:
            continue
        if lon < -105:
            continue  # the few storms that cross into the East Pacific are cut at 105 W
        wind = num(row['USA_WIND'])
        if wind is None:
            wind = num(row['WMO_WIND'])
        pressure = num(row['USA_PRES'])
        if pressure is None:
            pressure = num(row['WMO_PRES'])
        seconds = int((datetime.datetime.strptime(iso, '%Y-%m-%d %H:%M:%S') - ORIGIN).total_seconds())
        storms.setdefault(row['SID'], []).append({
            'season': season, 'name': row['NAME'], 'lat': lat, 'lon': lon, 'wind': wind,
            'pressure': pressure, 'time': seconds, 'nature': row['NATURE'],
            'dist': num(row['DIST2LAND'])})

ordered = sorted((v for v in storms.items() if len(v[1]) >= 4), key=lambda kv: kv[1][0]['time'])


def fill_wind(fixes):
    known = [i for i, f in enumerate(fixes) if f['wind'] is not None]
    if not known:
        for f in fixes: f['wind'] = 0.0
        return 0
    filled = 0
    for i, f in enumerate(fixes):
        if f['wind'] is not None: continue
        left = max((k for k in known if k < i), default=None)
        right = min((k for k in known if k > i), default=None)
        if left is None: f['wind'] = fixes[right]['wind']
        elif right is None: f['wind'] = fixes[left]['wind']
        else:
            t = (i - left) / (right - left)
            f['wind'] = fixes[left]['wind'] * (1 - t) + fixes[right]['wind'] * t
        filled += 1
    return filled


vertices, times, winds, pressures, cats, natures, dists = (array.array(c) for c in 'f I B H B B H'.split())
offsets = array.array('I', [0])
seasons, max_winds, max_cats, sids, names = array.array('H'), array.array('B'), array.array('B'), [], []
filled_total = 0
for sid, fixes in ordered:
    filled_total += fill_wind(fixes)
    for f in fixes:
        vertices.extend([f['lon'], f['lat']])
        times.append(f['time'])
        winds.append(int(round(f['wind'])))
        pressures.append(int(f['pressure']) if f['pressure'] else 0)
        cats.append(category(f['wind']))
        natures.append(NATURES.index(f['nature']) if f['nature'] in NATURES else 4)
        dists.append(min(65535, int(f['dist'])) if f['dist'] is not None else 65535)
    offsets.append(len(times))
    seasons.append(fixes[0]['season'])
    top = max(int(round(f['wind'])) for f in fixes)
    max_winds.append(top)
    max_cats.append(category(top))
    sids.append(sid)
    names.append(fixes[0]['name'].title() if fixes[0]['name'] != 'NOT_NAMED' else 'Unnamed')

os.makedirs(OUT, exist_ok=True)
columns = {}


def write(name, arr, dtype, components=1, **extra):
    arr = array.array(arr.typecode, arr)
    if sys.byteorder == 'big': arr.byteswap()
    with open(os.path.join(OUT, name + '.bin'), 'wb') as h: arr.tofile(h)
    columns[name] = {'file': name + '.bin', 'dtype': dtype, 'components': components,
                     'length': len(arr) // components, **extra}


write('pathOffsets', offsets, 'uint32')
write('vertices', vertices, 'float32', 2)
write('timestamp', times, 'uint32', unit='seconds since 1980-01-01T00:00:00Z (uint32, exact; scenes re-base per storm before using float32)')
write('wind', winds, 'uint8', unit='knots, 1-minute sustained (USA_WIND, else WMO_WIND; gaps interpolated)')
write('pressure', pressures, 'uint16', unit='hPa minimum central pressure, 0 where missing')
write('category', cats, 'uint8', categories=['TD', 'TS', 'Cat 1', 'Cat 2', 'Cat 3', 'Cat 4', 'Cat 5'],
      description='Saffir-Simpson class derived from wind at each fix (below 34 kt TD, 34-63 TS, 64-82 Cat 1, 83-95 Cat 2, 96-112 Cat 3, 113-136 Cat 4, 137+ Cat 5)')
write('nature', natures, 'uint8', categories=NATURES,
      description='IBTrACS NATURE: TS tropical, ET extratropical, DS disturbance, SS subtropical, NR not reported')
write('dist2land', dists, 'uint16', unit='km to nearest land (IBTrACS DIST2LAND), 65535 where missing')
write('season', seasons, 'uint16', description='per storm: IBTrACS season (year)')
write('maxWind', max_winds, 'uint8', unit='knots, per storm peak')
write('maxCategory', max_cats, 'uint8', categories=['TD', 'TS', 'Cat 1', 'Cat 2', 'Cat 3', 'Cat 4', 'Cat 5'])

lons = [vertices[i] for i in range(0, len(vertices), 2)]
lats = [vertices[i] for i in range(1, len(vertices), 2)]
manifest = {
    'id': 'ibtracs-north-atlantic', 'version': 1, 'kind': 'trajectories', 'count': len(ordered),
    'bbox': [min(lons), min(lats), max(lons), max(lats)], 'crs': 'EPSG:4326', 'columns': columns,
    'properties': {
        'timeOrigin': '1980-01-01T00:00:00Z', 'timeOriginEpochSeconds': 315532800,
        'stormIds': sids, 'names': names,
        'stormIdNote': 'storm i = properties.stormIds[i] (IBTrACS SID), names[i] its name; seasons per storm are in the season column',
        'window': 'seasons 1980-2025 (complete seasons; the 2026 season is in progress and omitted)',
        'sampling': '6-hourly synoptic fixes (00, 06, 12, 18 UTC) of the main track; interpolated landfall and 3-hourly fixes are dropped; storms with fewer than 4 fixes are dropped; fixes west of 105 W (East Pacific after a Central America crossing) are cut',
        'source': 'NOAA NCEI IBTrACS v04r01, North Atlantic list CSV',
        'doi': 'https://doi.org/10.25921/82ty-9e16',
        'attribution': 'Knapp, K. R., M. C. Kruk, D. H. Levinson, H. J. Diamond, C. J. Neumann (2010): The International Best Track Archive for Climate Stewardship (IBTrACS). Bulletin of the American Meteorological Society 91, 363-376. Public domain (NOAA).',
        'windGapsFilled': filled_total,
        'caveat': 'Best-track data are post-season analyses. Before the satellite era (about 1966, reliably 1980 onward) storms far from land were missed or undersampled, so trends before then are not comparable.'
    }}
with open(os.path.join(OUT, 'manifest.json'), 'w') as h: json.dump(manifest, h, indent=1)
print(len(ordered), 'storms', len(times), 'fixes', 'bbox', manifest['bbox'], 'wind gaps filled', filled_total)
