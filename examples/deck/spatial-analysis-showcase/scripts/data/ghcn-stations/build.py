"""ghcn-stations: NOAA GHCN-Daily PRCP and TMAX on 2024-09-26/27 (Hurricane Helene) for the southeastern US.
Reads the AWS open-data parquet/by_year shards for 2024 (PRCP, TMAX) plus ghcnd-stations.txt. Quality-flagged values are dropped."""
import json, os, re, urllib.request, concurrent.futures as cf
import numpy as np, pandas as pd
RAW = '/private/tmp/claude-501/-Users-robertchristie-Documents-GitHub-deck-gl-master/85599621-aabc-441b-ab0f-35747be6f781/scratchpad/showcase/raw/ghcn-stations'
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../public/data/ghcn-stations')
os.makedirs(RAW, exist_ok=True); os.makedirs(OUT, exist_ok=True)
B = 'https://noaa-ghcn-pds.s3.amazonaws.com/'
W, S_, E, N_ = -92.0, 24.0, -74.0, 40.0
DAYS = ['20240926', '20240927']
def fetch(url, path):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        open(path, 'wb').write(urllib.request.urlopen(url, timeout=120).read())
    return path
def shards(el):
    x = urllib.request.urlopen(f'{B}?list-type=2&prefix=parquet/by_year/YEAR=2024/ELEMENT={el}/').read().decode()
    return re.findall(r'<Key>([^<]*)</Key>', x)
fetch(B + 'ghcnd-stations.txt', os.path.join(RAW, 'ghcnd-stations.txt'))
frames = {}
for el in ('PRCP', 'TMAX'):
    keys = shards(el)
    with cf.ThreadPoolExecutor(6) as ex:
        paths = list(ex.map(lambda k: fetch(B + k, os.path.join(RAW, f'{el}_{os.path.basename(k)}')), keys))
    parts = []
    for p in paths:
        d = pd.read_parquet(p)
        d = d[d['DATE'].astype(str).isin(DAYS)] if 'DATE' in d else d
        parts.append(d)
    frames[el] = pd.concat(parts)
    print(el, frames[el].shape, frames[el].columns.tolist())
st = pd.read_fwf(os.path.join(RAW, 'ghcnd-stations.txt'), colspecs=[(0, 11), (12, 20), (21, 30), (31, 37), (38, 40), (41, 71)], names=['ID', 'lat', 'lon', 'elev', 'state', 'name'], dtype={'ID': str}, na_filter=False)
st = st[(st.lon > W) & (st.lon < E) & (st.lat > S_) & (st.lat < N_)]
st = st.assign(elev=pd.to_numeric(st.elev, errors='coerce')).dropna(subset=['elev']); st = st[st.elev > -400]
def wide(el, scale, name):
    d = frames[el]; d = d[d['Q_FLAG'].isna() | (d['Q_FLAG'].astype(str).str.strip() == '')]
    d = d[d['ID'].isin(st.ID)]
    d = d.assign(day=d['DATE'].astype(str), v=d['DATA_VALUE'].astype(float) * scale)
    return d.pivot_table(index='ID', columns='day', values='v', aggfunc='first').rename(columns={k: f'{name}{k[-2:]}' for k in DAYS})
t = st.set_index('ID').join(wide('PRCP', 0.1, 'prcp')).join(wide('TMAX', 0.1, 'tmax'))
t = t[t['prcp27'].notna() | t['prcp26'].notna()].reset_index()
t = t[t.prcp27.notna()].reset_index(drop=True)      # require the main day
N = len(t); print(N, 'stations')
cols = {}
def put(name, arr, dt, extra=None):
    arr = np.asarray(arr, dt)
    if dt == '<f4': arr[np.isnan(arr)] = np.nan
    arr.tofile(os.path.join(OUT, f'{name}.bin')); cols[name] = {'file': f'{name}.bin', 'dtype': {'<f4': 'float32'}[dt], 'components': 1 if name != 'position' else 2, 'length': N, **(extra or {})}
pos = np.stack([t.lon.values, t.lat.values], 1).astype('<f4'); pos.tofile(os.path.join(OUT, 'position.bin'))
cols['position'] = {'file': 'position.bin', 'dtype': 'float32', 'components': 2, 'length': N}
put('elevation', t.elev.values, '<f4', {'unit': 'm'})
put('prcp', t.prcp27.values, '<f4', {'unit': 'mm', 'description': 'GHCN-D PRCP 2024-09-27 (observation day, local morning to morning)'})
put('prcpPrevDay', t.prcp26.values, '<f4', {'unit': 'mm', 'nan': 'missing', 'description': 'PRCP 2024-09-26'})
put('tmax', t.tmax27.values, '<f4', {'unit': 'degC', 'nan': 'missing', 'description': 'TMAX 2024-09-27'})
put('tmaxPrevDay', t.tmax26.values, '<f4', {'unit': 'degC', 'nan': 'missing'})
t[['ID', 'name', 'state']].to_csv(os.path.join(OUT, 'stations.csv'), index=False)
top = t.nlargest(5, 'prcp27')[['ID', 'name', 'state', 'prcp27', 'elev']]
pr = t.prcp27
man = {'id': 'ghcn-stations', 'version': 1, 'kind': 'points', 'count': N, 'bbox': [W, S_, E, N_], 'crs': 'EPSG:4326', 'columns': cols,
  'properties': {'event': 'Hurricane Helene, 2024-09-26/27 (landfall Perry FL ~03Z 09-27; record flooding in the southern Appalachians)', 'stations': 'stations.csv (ID, name, state; row order = column order)',
    'source': 'NOAA NCEI GHCN-Daily via AWS Open Data (noaa-ghcn-pds), quality-flagged values removed', 'stationsWithTmax': int(t.tmax27.notna().sum()),
    'prcpSummary': {'maxMm': float(pr.max()), 'p99Mm': float(pr.quantile(.99)), 'meanMm': float(pr.mean()), 'fractionAbove100mm': float((pr > 100).mean()), 'fractionDry': float((pr == 0).mean())},
    'wettest': top.to_dict('records'),
    'storyNotes': [f'{N} stations with a valid 2024-09-27 precipitation total across the Southeast (lon -92..-74, lat 24..40); elevation from the GHCN station file.',
      f"Wettest station: {top.iloc[0]['name'].strip()} ({top.iloc[0]['state']}) with {top.iloc[0]['prcp27']:.0f} mm in the day; {(pr > 100).mean()*100:.1f}% of stations exceed 100 mm.",
      'Rain concentrates along Helene\'s track and on the windward Blue Ridge: elevation is a real covariate, so regression-kriging beats plain IDW and cross-validation shows the gain.',
      'Station density is very uneven (dense in cities, sparse in Appalachian valleys and the Gulf): a good demo of kriging variance and leave-one-out error.',
      'TMAX (cool under the rain shield, hot far from it) is a second interpolation target with a clear lapse-rate relation to elevation.',
      'Units: precipitation mm, temperature degC (converted from tenths); NaN = missing.']}}
json.dump(man, open(os.path.join(OUT, 'manifest.json'), 'w'), indent=1)
print(top.to_string(), pr.describe(), {f: os.path.getsize(os.path.join(OUT, f)) for f in os.listdir(OUT)})
