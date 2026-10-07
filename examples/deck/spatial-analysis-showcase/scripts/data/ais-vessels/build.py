"""NOAA Marine Cadastre AIS (CC0) NY/NJ Harbor, one day -> per-vessel trajectories + zones (see README.md)."""
import json, os, urllib.request
import numpy as np, pandas as pd, duckdb
HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
RAW = os.environ.get('SHOWCASE_RAW', '/private/tmp/claude-501/-Users-robertchristie-Documents-GitHub-deck-gl-master/85599621-aabc-441b-ab0f-35747be6f781/scratchpad/showcase/raw') + '/ais-vessels'
OUT = os.path.join(APP, 'public/data/ais-vessels'); OUTZ = os.path.join(APP, 'public/data/ais-zones')
os.makedirs(RAW, exist_ok=True); os.makedirs(OUT, exist_ok=True); os.makedirs(OUTZ, exist_ok=True)
DAY = '2024-06-12'; T0 = pd.Timestamp(DAY + 'T00:00:00')
SRC = f'https://ocmgeodatastor1.blob.core.windows.net/marinecadastre/ais2024/ais-{DAY}.parquet'
PQ = f'{RAW}/ais-{DAY}.parquet'
if not os.path.exists(PQ): urllib.request.urlretrieve(SRC, PQ)
BBOX = (-74.30, 40.45, -73.75, 40.80)  # w s e n
CATS = ['cargo', 'tanker', 'passenger', 'tug', 'fishing', 'pleasure', 'other']
def category(t):
    if t is None or np.isnan(t): return 6
    t = int(t)
    if 70 <= t <= 79: return 0
    if 80 <= t <= 89: return 1
    if 60 <= t <= 69: return 2
    if t in (31, 32, 52): return 3
    if t == 30: return 4
    if t in (36, 37): return 5
    return 6

con = duckdb.connect(); con.sql('install spatial; load spatial')
df = con.sql(f"""select mmsi, base_date_time t, sog, cog, heading, vessel_type, length, status, ST_X(geometry) lng, ST_Y(geometry) lat
 from read_parquet('{PQ}') where ST_X(geometry) between {BBOX[0]} and {BBOX[2]} and ST_Y(geometry) between {BBOX[1]} and {BBOX[3]}""").df()
print('raw points', len(df), 'vessels', df.mmsi.nunique())
df = df[(df.sog.isna()) | (df.sog < 45)].copy()  # drop implausible SOG (AIS sentinel/garbage)
df['sec'] = ((df.t - T0).dt.total_seconds()).astype('int64')
df = df.sort_values(['mmsi', 'sec']).drop_duplicates(['mmsi', 'sec'])
# vessel-level attributes: modal type, max length
vt = df.dropna(subset=['vessel_type']).groupby('mmsi').vessel_type.agg(lambda s: s.mode().iloc[0])
vl = df.groupby('mmsi').length.max()
M2 = lambda lat: 111320.0 * np.cos(np.radians(lat))
paths = []  # (mmsi, arrays)
for mmsi, g in df.groupby('mmsi', sort=True):
    sec = g.sec.values; lng = g.lng.values; lat = g.lat.values; sog = g.sog.fillna(0).values; cog = g.cog.fillna(0).values; hd = g.heading.values.astype(float)
    # drop position spikes (> 50 kn implied)
    keep = [0]
    for i in range(1, len(sec)):
        j = keep[-1]; dt = max(sec[i] - sec[j], 1)
        d = np.hypot((lng[i]-lng[j]) * M2(lat[i]), (lat[i]-lat[j]) * 110574.0)
        if d / dt > 25.7: continue
        keep.append(i)
    keep = np.array(keep)
    sec, lng, lat, sog, cog, hd = sec[keep], lng[keep], lat[keep], sog[keep], cog[keep], hd[keep]
    # split at gaps > 20 min
    cut = np.where(np.diff(sec) > 1200)[0] + 1
    for s, e in zip(np.r_[0, cut], np.r_[cut, len(sec)]):
        # thin stationary samples: keep at most one per 5 min when barely moving
        k = [s]
        for i in range(s + 1, e):
            j = k[-1]
            d = np.hypot((lng[i]-lng[j]) * M2(lat[i]), (lat[i]-lat[j]) * 110574.0)
            if (d < 30 and sog[i] < 1.0 and sec[i] - sec[j] < 300 and i != e - 1): continue
            k.append(i)
        k = np.array(k)
        if len(k) < 20: continue
        h = np.where(np.isnan(hd[k]) | (hd[k] >= 360), cog[k], hd[k]) % 360
        paths.append(dict(mmsi=int(mmsi), lng=lng[k], lat=lat[k], sec=sec[k], sog=sog[k], hd=h))
print('paths', len(paths), 'points', sum(len(p['sec']) for p in paths))

n = len(paths)
offs = np.zeros(n + 1, '<u4'); offs[1:] = np.cumsum([len(p['sec']) for p in paths])
vert = np.concatenate([np.c_[p['lng'], p['lat']] for p in paths]).astype('<f4')
tim = np.concatenate([p['sec'] for p in paths]).astype('<u4')
spd = np.concatenate([p['sog'] for p in paths]).astype('<f4')
hdg = (np.concatenate([p['hd'] for p in paths]).round() % 360).astype('<u2')
mm = np.array([p['mmsi'] for p in paths], '<u4')
uniq = sorted(set(mm.tolist())); vindex = {m: i for i, m in enumerate(uniq)}
vesselIdx = np.array([vindex[m] for m in mm], '<u4')
cat = np.array([category(vt.get(m, np.nan)) for m in mm], 'u1')
ln = np.array([vl.get(m, 0) if not np.isnan(vl.get(m, np.nan)) else 0 for m in mm], '<f4')
AISTYPE = np.array([vt.get(m, 0) if not np.isnan(vt.get(m, np.nan)) else 0 for m in mm], '<u2')
meanspeed = np.array([p['sog'].mean() for p in paths], '<f4')
files = {'pathOffsets.bin': offs, 'vertices.bin': vert, 'timestamp.bin': tim, 'speed.bin': spd, 'heading.bin': hdg,
         'mmsi.bin': mm, 'vesselIndex.bin': vesselIdx, 'category.bin': cat, 'length.bin': ln, 'aisVesselType.bin': AISTYPE}
for f, a in files.items(): a.tofile(f'{OUT}/{f}')
npts = len(tim)
def col(f, dt, comp, ln_, **kw): return {'file': f, 'dtype': dt, 'components': comp, 'length': ln_, **kw}

# story stats
dur = np.array([p['sec'][-1] - p['sec'][0] for p in paths])
cc = {c: int((cat == i).sum()) for i, c in enumerate(CATS)}
hourly = np.bincount((tim // 3600).astype(int), minlength=24)
m = {'id': 'ais-vessels', 'version': 1, 'kind': 'trajectories', 'count': n,
 'bbox': [float(vert[:, 0].min()), float(vert[:, 1].min()), float(vert[:, 0].max()), float(vert[:, 1].max())], 'crs': 'EPSG:4326',
 'columns': {
  'pathOffsets': col('pathOffsets.bin', 'uint32', 1, n + 1),
  'vertices': col('vertices.bin', 'float32', 2, npts),
  'timestamp': col('timestamp.bin', 'uint32', 1, npts, unit=f'seconds since {DAY}T00:00:00Z'),
  'speed': col('speed.bin', 'float32', 1, npts, unit='knots (SOG)'),
  'heading': col('heading.bin', 'uint16', 1, npts, unit='degrees true; falls back to COG when heading unavailable'),
  'mmsi': col('mmsi.bin', 'uint32', 1, n, description='per path; public AIS MMSI'),
  'vesselIndex': col('vesselIndex.bin', 'uint32', 1, n, description='dense vessel id 0..vesselCount-1 (a vessel can have several paths after gaps > 20 min)'),
  'category': col('category.bin', 'uint8', 1, n, categories=CATS),
  'length': col('length.bin', 'float32', 1, n, unit='m, 0 = unknown'),
  'aisVesselType': col('aisVesselType.bin', 'uint16', 1, n, description='raw AIS ship-type code')},
 'properties': {'day': DAY, 'vesselCount': len(uniq), 'sourceSampling': 'one fix per minute (NOAA), stationary vessels thinned to one per 5 min',
   'zonesDataset': 'ais-zones', 'categoryRule': 'AIS type 70-79 cargo, 80-89 tanker, 60-69 passenger, 31/32/52 tug, 30 fishing, 36/37 pleasure, else other',
   'attribution': 'NOAA Office for Coastal Management / U.S. Coast Guard Navigation Center, Nationwide AIS 2024 (CC0 1.0)',
   'storyNotes': [
     f'{len(uniq)} vessels in NY/NJ Harbor on {DAY} (UTC): ' + ', '.join(f'{v} {k}' for k, v in cc.items() if v) + ' paths.',
     f'Busiest UTC hour is {int(hourly.argmax())}:00 with {int(hourly.max())} fixes; quietest is {int(hourly.argmin())}:00 with {int(hourly.min())}.',
     f'Fastest fix {float(spd.max()):.1f} kn; {int((meanspeed < 0.5).sum())} tracks barely move (moored, anchored or at terminals).',
     'Traffic funnels through the Narrows under the Verrazzano-Narrows Bridge (about 40.606 N, 74.045 W) between the Lower and Upper Bay: the natural gate for zone-crossing events.',
     'Ferries and tugs make repeated crossings of the Upper Bay all day, giving many close encounters; deep-draft cargo/tankers follow Ambrose and Anchorage Channels to Port Newark-Elizabeth and Bayonne.']}}
json.dump(m, open(f'{OUT}/manifest.json', 'w'), indent=1)
print({f: os.path.getsize(f'{OUT}/{f}') for f in os.listdir(OUT)})
print(m['properties']['storyNotes'])
