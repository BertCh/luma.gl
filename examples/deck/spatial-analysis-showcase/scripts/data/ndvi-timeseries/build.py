"""ndvi-timeseries: 16 cloud-screened Sentinel-2 L2A NDVI snapshots (2017-2024 summers) of a 25.6 km window around Greenville CA
(Dixie Fire, 2021-07-13..2021-10-25) at 100 m (256x256, EPSG:32610). uint8: 0 = nodata/cloud, else v = round((ndvi+1)*127.5) clipped 1..255."""
import json, os, datetime
import numpy as np, requests, rasterio
from rasterio.windows import from_bounds
from rasterio.enums import Resampling
from pyproj import Transformer
RAW = '/private/tmp/claude-501/-Users-robertchristie-Documents-GitHub-deck-gl-master/85599621-aabc-441b-ab0f-35747be6f781/scratchpad/showcase/raw/ndvi-timeseries'
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../public/data/ndvi-timeseries')
os.makedirs(RAW, exist_ok=True); os.makedirs(OUT, exist_ok=True)
N, RES = 256, 100.0
to_utm = Transformer.from_crs(4326, 32610, always_xy=True); to_ll = Transformer.from_crs(32610, 4326, always_xy=True)
cx, cy = to_utm.transform(-121.0, 40.17)
west = round(cx / RES) * RES - N * RES / 2; north = round(cy / RES) * RES + N * RES / 2
bounds = (west, north - N * RES, west + N * RES, north)
feats = []
for y in range(2017, 2025):
    r = requests.post('https://earth-search.aws.element84.com/v1/search', json={'collections': ['sentinel-2-l2a'], 'bbox': [-121.0, 40.17, -120.99, 40.18],
        'datetime': f'{y}-06-01T00:00:00Z/{y}-09-30T23:59:59Z', 'limit': 100, 'query': {'eo:cloud_cover': {'lt': 30}}}).json()
    feats += [f for f in r['features'] if '10TFK' in f['id']]
print(len(feats), 'candidate scenes')
def read(href, resampling, window_shape=(N, N)):
    with rasterio.open(href) as d:
        return d.read(1, window=from_bounds(*bounds, transform=d.transform), out_shape=window_shape, resampling=resampling), d
cands = {}
with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR', AWS_NO_SIGN_REQUEST='YES'):
    for f in sorted(feats, key=lambda f: f['id']):
        date = f['properties']['datetime'][:10]
        if date in cands and f['id'].split('_')[3] == '1': continue
        scl, _ = read(f['assets']['scl']['href'], Resampling.nearest)
        bad = np.isin(scl, (0, 1, 3, 8, 9, 10, 11))
        cands.setdefault(date, []).append((float(bad.mean()), f, scl))
best = {d: min(v, key=lambda t: t[0]) for d, v in cands.items()}
chosen = []
for y in range(2017, 2025):
    for lo, hi in (('06-01', '07-31'), ('08-01', '09-30')):
        pool = [(v[0], d) for d, v in best.items() if f'{y}-{lo}' <= d <= f'{y}-{hi}' and v[0] < 0.05]
        if pool: chosen.append(min(pool)[1])
for forced in ('2021-07-13', '2021-09-21'):
    if forced in best and forced not in chosen: chosen.append(forced)
# thin to 16 keeping forced
while len(chosen) > 16:
    drop = min((d for d in chosen if d not in ('2021-07-13', '2021-09-21')), key=lambda d: -best[d][0]) if False else None
    chosen.pop(max((i for i, d in enumerate(chosen) if d not in ('2021-07-13', '2021-09-21') and d.startswith(('2017', '2018', '2019', '2020', '2022', '2023', '2024'))), default=0, key=lambda i: best[chosen[i]][0]))
chosen = sorted(chosen)
stack = np.zeros((len(chosen), N, N), np.uint8); info = []
with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR', AWS_NO_SIGN_REQUEST='YES'):
    for i, d in enumerate(chosen):
        _, f, scl = best[d]
        red, dr = read(f['assets']['red']['href'], Resampling.average); nir, dn = read(f['assets']['nir']['href'], Resampling.average)
        # NOTE: STAC claims a -0.1 BOA offset for post-2022 scenes, but the Earth Search COG values are already offset-free
        # (red median 0.08 in 2022, impossible if 0.1 were subtracted), so plain DN * 1e-4 is used for all dates.
        rr = red.astype(np.float64) * 1e-4; nn = nir.astype(np.float64) * 1e-4
        ndvi = (nn - rr) / np.maximum(nn + rr, 1e-6)
        valid = ~np.isin(scl, (0, 1, 3, 8, 9, 10, 11)) & (red > 0) & (nir > 0) & (nn + rr > 0.01)
        v = np.clip(np.round((ndvi + 1) * 127.5), 1, 255).astype(np.uint8); v[~valid] = 0
        stack[i] = v
        info.append({'date': d, 'scene': f['id'], 'validFraction': float(valid.mean()), 'meanNDVI': float(ndvi[valid].mean())})
        print(d, f['id'], round(valid.mean(), 3), round(ndvi[valid].mean(), 3))
stack.tofile(os.path.join(OUT, 'ndvi.bin'))
days = [(datetime.date.fromisoformat(i['date']) - datetime.date(2017, 1, 1)).days for i in info]
np.array(days, '<u2').tofile(os.path.join(OUT, 'days.bin'))
w_, s_ = to_ll.transform(bounds[0], bounds[1]); e_, n_ = to_ll.transform(bounds[2], bounds[3])
bb = [min(w_, to_ll.transform(bounds[0], bounds[3])[0]), s_, max(e_, to_ll.transform(bounds[2], bounds[1])[0]), n_]
fire = [i for i in info if '2021-08-01' < i['date'] < '2021-12-31'][:1]
man = {'id': 'ndvi-timeseries', 'version': 1, 'kind': 'raster', 'count': len(chosen) * N * N, 'bbox': bb, 'crs': 'EPSG:32610',
  'raster': {'file': 'ndvi.bin', 'encoding': 'uint8-bin', 'dtype': 'uint8', 'width': N, 'height': N, 'depth': len(chosen), 'layout': 'time-major [t][row][col], row 0 = north',
             'bounds': bb, 'boundsProjected': list(bounds), 'projection': 'EPSG:32610', 'cellSizeM': RES, 'noData': 0, 'unit': 'NDVI',
             'decode': 'ndvi = v / 127.5 - 1 for v > 0; v == 0 is cloud/shadow/no data'},
  'columns': {'days': {'file': 'days.bin', 'dtype': 'uint16', 'components': 1, 'length': len(chosen), 'unit': 'days since 2017-01-01'}},
  'properties': {'dates': [i['date'] for i in info], 'scenes': info, 'dixieFire': {'ignition': '2021-07-13', 'greenvilleDestroyed': '2021-08-04', 'contained': '2021-10-25'},
    'attribution': 'Contains modified Copernicus Sentinel data 2017-2024 (ESA)',
    'storyNotes': [f'{len(chosen)} cloud-screened summer dates 2017-2024 over a 25.6 km window at 100 m; every pixel carries a 16-step time series (v == 0 gaps are masked cloud).',
      'Mann-Kendall / Sen slope over all dates mostly captures the Dixie Fire step: NDVI collapses in the 2021-09-21 snapshot and recovers only partially through 2024 (window mean NDVI ~0.60 in 2021-07-13, ~0.39 on 2021-09-21, ~0.35 in 2022, ~0.47 by 2024-06).',
      'Pixels outside the burn scar (Lake Almanor shoreline, irrigated Indian Valley meadows) give flat or weakly positive trends: a clean contrast for significance maps.',
      'For a pure recovery-trend test, use only dates after 2021-09-21 (indices in properties.dates).',
      'Dates are not evenly spaced (two per summer): treat time as ordinal for Mann-Kendall or use days.bin for Sen slope per year.']}}
json.dump(man, open(os.path.join(OUT, 'manifest.json'), 'w'), indent=1)
print({f: os.path.getsize(os.path.join(OUT, f)) for f in os.listdir(OUT)}, bb)
