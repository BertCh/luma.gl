"""Build cta-transit from the CTA GTFS feed (stops, route counts, shapes, weekday hop graph).
usage: build.py GTFS_ZIP COMMUNITY_AREAS_GEOJSON OUT_DIR"""
import json, sys, pathlib, zipfile, datetime as dt
import numpy as np, pandas as pd, geopandas as gpd, shapely
zpath, ca_path, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3]); out.mkdir(parents=True, exist_ok=True)
z = zipfile.ZipFile(zpath)
rd = lambda n, **k: pd.read_csv(z.open(n), dtype=str, **k)
ca = gpd.read_file(ca_path).to_crs(4326); cb = ca.total_bounds; city = ca.union_all()
BUF = 0.01
box = (cb[0] - BUF, cb[1] - BUF, cb[2] + BUF, cb[3] + BUF)
routes = rd('routes.txt'); trips = rd('trips.txt'); cal = rd('calendar.txt'); cald = rd('calendar_dates.txt'); stops = rd('stops.txt')
# representative weekday: Wednesday inside the feed's validity window
cal['s'] = pd.to_datetime(cal.start_date); cal['e'] = pd.to_datetime(cal.end_date)
day = pd.Timestamp(cal.s.min()) + pd.Timedelta(days=(2 - pd.Timestamp(cal.s.min()).weekday()) % 7)
sid = set(cal[(cal.s <= day) & (cal.e >= day) & (cal.wednesday == '1')].service_id)
ex = cald[cald.date == day.strftime('%Y%m%d')]
sid = (sid - set(ex[ex.exception_type == '2'].service_id)) | set(ex[ex.exception_type == '1'].service_id)
trips = trips[trips.service_id.isin(sid)].copy(); print('weekday', day.date(), 'services', len(sid), 'trips', len(trips), flush=True)
st = pd.read_csv(z.open('stop_times.txt'), usecols=['trip_id', 'departure_time', 'stop_id', 'stop_sequence'], dtype=str)
st = st[st.trip_id.isin(set(trips.trip_id))].copy()
st['seq'] = st.stop_sequence.astype(int)
t = st.departure_time.str.split(':', expand=True).astype(int); st['sec'] = t[0] * 3600 + t[1] * 60 + t[2]; st['hour'] = (st.sec // 3600) % 24
st = st.merge(trips[['trip_id', 'route_id', 'direction_id', 'shape_id']], on='trip_id')
st = st.sort_values(['trip_id', 'seq']).reset_index(drop=True)
# collapse rail platforms to parent station
stops['lat'] = stops.stop_lat.astype(float); stops['lon'] = stops.stop_lon.astype(float)
parent = stops.set_index('stop_id').parent_station.fillna('')
st['sid'] = [parent.get(s, '') or s for s in st.stop_id]
# stations table (location_type 1 for rail parents and location_type 0 for bus stops)
rt = routes.set_index('route_id'); rlist = routes.reset_index(drop=True)
rindex = {r: i for i, r in enumerate(rlist.route_id)}
st['mode'] = st.route_id.map(lambda r: 1 if rt.route_type[r] == '1' else 0)
g = st.groupby('sid')
S = pd.DataFrame({'trips': g.trip_id.nunique(), 'routes': g.route_id.nunique(), 'rail': g['mode'].max(),
                  'peak': st[(st.hour >= 7) & (st.hour < 9)].groupby('sid').trip_id.nunique()}).fillna(0)
S = S.join(stops.set_index('stop_id')[['lat', 'lon', 'stop_name', 'wheelchair_boarding']], how='inner')
inbox = (S.lon >= box[0]) & (S.lon <= box[2]) & (S.lat >= box[1]) & (S.lat <= box[3])
S = S[inbox].copy(); S = S.sort_values(['rail', 'lat', 'lon'], ascending=[False, True, True])
sidx = {s: i for i, s in enumerate(S.index)}; ns = len(S)
S['wc'] = S.wheelchair_boarding.fillna('0').astype(int).clip(0, 2)
# stop -> routes CSR
sr = st[st.sid.isin(sidx)][['sid', 'route_id']].drop_duplicates()
sr['si'] = sr.sid.map(sidx); sr['ri'] = sr.route_id.map(rindex); sr = sr.sort_values(['si', 'ri'])
soff = np.zeros(ns + 1, '<u4'); np.add.at(soff, sr.si.values + 1, 1); soff = np.cumsum(soff).astype('<u4')
# hop graph: consecutive stops within a trip (stations), median scheduled seconds
st['next_sid'] = st.groupby('trip_id').sid.shift(-1); st['next_sec'] = st.groupby('trip_id').sec.shift(-1)
h = st.dropna(subset=['next_sid']).copy(); h['dt'] = h.next_sec - h.sec
h = h[h.sid.isin(sidx) & h.next_sid.isin(sidx) & (h.sid != h.next_sid) & (h.dt >= 0)]
H = h.groupby(['sid', 'next_sid']).agg(sec=('dt', 'median'), trips=('trip_id', 'nunique'), rail=('mode', 'max')).reset_index()
H['a'] = H.sid.map(sidx); H['b'] = H.next_sid.map(sidx); H = H.sort_values(['a', 'b']).reset_index(drop=True)
# shapes: most common shape per (route, direction)
sh = pd.read_csv(z.open('shapes.txt'), dtype={'shape_id': str}, usecols=['shape_id', 'shape_pt_lat', 'shape_pt_lon', 'shape_pt_sequence'])
pick = trips.groupby(['route_id', 'direction_id']).shape_id.agg(lambda x: x.value_counts().index[0]).reset_index()
sh = sh[sh.shape_id.isin(set(pick.shape_id))].sort_values(['shape_id', 'shape_pt_sequence'])
lines = {k: shapely.LineString(np.column_stack([v.shape_pt_lon, v.shape_pt_lat])) for k, v in sh.groupby('shape_id')}
paths, pr, pdir = [], [], []
for r in pick.itertuples():
    ln = shapely.simplify(lines[r.shape_id], 4e-5)   # ~4 m
    clip = ln.intersection(shapely.box(*box))
    parts = [clip] if clip.geom_type == 'LineString' else list(getattr(clip, 'geoms', []))
    for p in parts:
        if p.geom_type == 'LineString' and len(p.coords) > 1: paths.append(np.asarray(p.coords, '<f4')); pr.append(rindex[r.route_id]); pdir.append(int(r.direction_id))
poff = np.cumsum([0] + [len(p) for p in paths]).astype('<u4'); pv = np.concatenate(paths).astype('<f4')
cols = {}
def add(name, arr, dtype, comp=1, **extra):
    arr = np.ascontiguousarray(np.asarray(arr).astype(dtype)); arr.tofile(out / f'{name}.bin')
    cols[name] = {'file': f'{name}.bin', 'dtype': np.dtype(dtype).name, 'components': comp, 'length': int(arr.size // comp), **extra}
add('stopPosition', S[['lon', 'lat']].values, '<f4', 2, note='stops: bus stops + rail stations (platforms merged into parent station)')
add('stopMode', S.rail, '<u1', categories=['bus', 'rail'])
add('stopRouteCount', S.routes, '<u1', note='distinct routes serving the stop on the sample weekday')
add('stopWeekdayTrips', S.trips, '<u2', note='scheduled departures on the sample weekday')
add('stopPeakTrips', S.peak, '<u2', note='scheduled departures 07:00-08:59 on the sample weekday')
add('stopWheelchair', S.wc, '<u1', categories=['unknown', 'accessible', 'not accessible'], note='GTFS wheelchair_boarding')
add('stopRouteOffsets', soff, '<u4', note='CSR over stopRoutes: routes of stop i are stopRoutes[offsets[i]..offsets[i+1]-1]')
add('stopRoutes', sr.ri.values, '<u2', note='index into properties.routes')
add('hopSource', H.a, '<u4', note='transit hops: consecutive stops of a trip, weekday service; indices into stops')
add('hopTarget', H.b, '<u4')
add('hopSeconds', H.sec, '<f4', unit='s', note='median scheduled in-vehicle time')
add('hopTrips', H.trips, '<u2', note='weekday trips using the hop')
add('hopMode', H.rail, '<u1', categories=['bus', 'rail'])
add('shapePathOffsets', poff, '<u4', note='n+1 offsets into shapeVertices (route shapes clipped to the city bbox + 1 km)')
add('shapeVertices', pv, '<f4', 2)
add('shapeRoute', pr, '<u2', note='index into properties.routes')
add('shapeDirection', pdir, '<u1')
rl = [{'id': r.route_id, 'short': r.route_short_name if isinstance(r.route_short_name, str) else '', 'name': r.route_long_name, 'type': 'rail' if r.route_type == '1' else 'bus', 'color': '#' + r.route_color} for r in rlist.itertuples()]
rail = S[S.rail == 1]
cityS = S[shapely.contains_xy(city, S.lon.values, S.lat.values)]
top = S.sort_values('trips', ascending=False).head(3)
story = [
    f"{int((S.rail==0).sum()):,} bus stops and {len(rail)} rail stations inside the Chicago bbox; {len(rlist)} routes ({sum(r['type']=='rail' for r in rl)} 'L' lines).",
    f"Busiest stops by weekday departures: " + '; '.join(f"{r.stop_name} ({int(r.trips)})" for r in top.itertuples()) + '.',
    f"Stops/terminals with the most routes: " + ', '.join(f"{r.stop_name} ({int(r.routes)})" for r in S.sort_values('routes', ascending=False).head(3).itertuples()) + '; \'rail\' mode marks any stop where an L line calls (bus bays at rail terminals are merged into the station).',
    f"{len(H):,} weekday hops form a ready-made transit graph; median bus hop {H[H.rail==0].sec.median():.0f} s, median rail hop {H[H.rail==1].sec.median():.0f} s.",
    f"Sample service day is Wednesday {day.date()}; the published feed only covers {cal.s.min().date()} to {cal.e.max().date()}.",
    f"{len(cityS):,} of {ns:,} stops lie inside the city limits; the rest are inner suburbs within ~1 km of the boundary."]
manifest = {'id': 'cta-transit', 'version': 1, 'kind': 'network', 'count': int(ns), 'bbox': [round(float(x), 5) for x in (S.lon.min(), S.lat.min(), S.lon.max(), S.lat.max())], 'crs': 'EPSG:4326', 'columns': cols,
  'properties': {'description': 'CTA GTFS aggregates: bus stops and rail stations with route counts and weekday frequency, route shapes, and a weekday stop-to-stop hop graph with median scheduled times. Derived from the CTA GTFS feed; contains no full timetable.',
    'licence': 'CTA Developer License Agreement and Terms of Use (limited licence to use/distribute/derive for assisting transit riders or promoting public transportation; no stand-alone resale; credit optional)',
    'attribution': 'Data provided by Chicago Transit Authority', 'routes': rl, 'sampleDate': str(day.date()), 'storyNotes': story}}
(out / 'manifest.json').write_text(json.dumps(manifest, indent=1))
print('\n'.join(story)); print('bytes', sum(p.stat().st_size for p in out.iterdir()))
