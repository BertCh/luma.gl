"""dixie-fire: Sentinel-2 L2A before/after the 2021 Dixie Fire (Greenville CA), ESA WorldCover 2021 and an AWS-Terrarium DEM,
all on ONE common 20 m UTM 10N grid (EPSG:32610, 750x750 = 15 km). Bands are shipped as raw reflectance*10000 uint16
(no indices precomputed: NDVI/NBR/dNBR are computed on the GPU)."""
import json, math, os
import numpy as np, requests, rasterio
from rasterio.windows import from_bounds
from rasterio.enums import Resampling
from rasterio.warp import reproject
from rasterio.transform import from_origin
from pyproj import Transformer
from PIL import Image
import urllib.request
RAW = '/private/tmp/claude-501/-Users-robertchristie-Documents-GitHub-deck-gl-master/85599621-aabc-441b-ab0f-35747be6f781/scratchpad/showcase/raw/dixie-fire'
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../public/data/dixie-fire')
os.makedirs(RAW, exist_ok=True); os.makedirs(OUT, exist_ok=True)
N, RES = 750, 20.0
to_utm = Transformer.from_crs(4326, 32610, always_xy=True); to_ll = Transformer.from_crs(32610, 4326, always_xy=True)
cx, cy = to_utm.transform(-121.0, 40.17)
west = round(cx / RES) * RES - N * RES / 2; north = round(cy / RES) * RES + N * RES / 2
bounds = (west, north - N * RES, west + N * RES, north)  # w s e n
transform = from_origin(west, north, RES, RES)
SCENES = {'before': ('S2B_10TFK_20210713_0_L2A', '2021-07-13'), 'after': ('S2B_10TFK_20210921_0_L2A', '2021-09-21')}
STAC = requests.post('https://earth-search.aws.element84.com/v1/search', json={'collections': ['sentinel-2-l2a'], 'ids': [v[0] for v in SCENES.values()]}).json()['features']
assets = {f['id']: f['assets'] for f in STAC}
BANDS = {'red': 'B04', 'nir': 'B08', 'swir22': 'B12'}
files = {}; stats = {}
with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR', AWS_NO_SIGN_REQUEST='YES'):
    for tag, (sid, date) in SCENES.items():
        arr = {}
        for key, b in BANDS.items():
            with rasterio.open(assets[sid][key]['href']) as d:
                w = from_bounds(*bounds, transform=d.transform)
                arr[key] = d.read(1, window=w, out_shape=(N, N), resampling=Resampling.average, boundless=False).astype(np.uint16)
            arr[key].astype('<u2').tofile(os.path.join(OUT, f'{b}_{tag}.bin')); files[f'{key}_{tag}'] = f'{b}_{tag}.bin'
        with rasterio.open(assets[sid]['scl']['href']) as d:
            scl = d.read(1, window=from_bounds(*bounds, transform=d.transform), out_shape=(N, N), resampling=Resampling.nearest).astype(np.uint8)
        Image.fromarray(scl, 'L').save(os.path.join(OUT, f'scl_{tag}.png'), optimize=True)
        r, n, s = (arr[k].astype(np.float64) / 1e4 for k in ('red', 'nir', 'swir22'))
        valid = (arr['red'] > 0) & (arr['nir'] > 0) & (arr['swir22'] > 0)
        ndvi = (n - r) / np.maximum(n + r, 1e-6); nbr = (n - s) / np.maximum(n + s, 1e-6)
        arr['ndvi'], arr['nbr'], arr['valid'] = ndvi, nbr, valid
        arr['scl'] = scl
        cloud = np.isin(scl, (3, 8, 9, 10))
        stats[tag] = {'scene': sid, 'date': date, 'cloudOrShadowFraction': float(cloud.mean()), 'validFraction': float(valid.mean()),
                      'meanNDVI': float(ndvi[valid].mean()), 'meanNBR': float(nbr[valid].mean()), 'sclHistogram': {int(k): int((scl == k).sum()) for k in np.unique(scl)}}
        globals()['A_' + tag] = arr
a, b = A_before, A_after
dnbr = a['nbr'] - b['nbr']; ok = a['valid'] & b['valid']
burn = {'dNBR>0.1 (low+)': 0.1, 'dNBR>0.27 (moderate-low+)': 0.27, 'dNBR>0.66 (high)': 0.66}
validation = {'meanDNBR': float(dnbr[ok].mean()), 'maxDNBR': float(dnbr[ok].max()),
              'fractionAbove': {k: float((dnbr[ok] > t).mean()) for k, t in burn.items()}, 'meanDNDVI': float((a['ndvi'] - b['ndvi'])[ok].mean()),
              'note': 'dNBR = NBR_before - NBR_after, NBR = (B08-B12)/(B08+B12), reflectance = DN/10000; GPU results should match within float32 rounding.'}
# ---- WorldCover on the same grid
wc = np.zeros((N, N), np.uint8)
with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR', AWS_NO_SIGN_REQUEST='YES'):
    with rasterio.open('https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_N39W123_Map.tif') as d:
        reproject(rasterio.band(d, 1), wc, dst_transform=transform, dst_crs='EPSG:32610', resampling=Resampling.mode)
Image.fromarray(wc, 'L').save(os.path.join(OUT, 'worldcover.png'), optimize=True)
LEG = {10: 'Tree cover', 20: 'Shrubland', 30: 'Grassland', 40: 'Cropland', 50: 'Built-up', 60: 'Bare / sparse vegetation', 70: 'Snow and ice', 80: 'Permanent water bodies', 90: 'Herbaceous wetland', 95: 'Mangroves', 100: 'Moss and lichen'}
wc_hist = {LEG.get(int(k), str(k)): float((wc == k).mean()) for k in np.unique(wc)}
validation['burnedFractionByWorldCover(dNBR>0.27)'] = {LEG[int(k)]: float((dnbr[ok & (wc == k)] > 0.27).mean()) for k in np.unique(wc) if (ok & (wc == k)).sum() > 500}
# ---- DEM (AWS Terrarium z13) resampled bilinear onto the UTM grid, re-encoded as Terrarium
from scipy.ndimage import map_coordinates
Z, TS = 13, 256; MR = 6378137.0; ORG = math.pi * MR; mres = 2 * ORG / (TS * 2 ** Z)
to_m = Transformer.from_crs(32610, 3857, always_xy=True)
xs = west + (np.arange(N) + .5) * RES; ys = north - (np.arange(N) + .5) * RES
X, Y = np.meshgrid(xs, ys); MX, MY = to_m.transform(X, Y)
gpx = (MX + ORG) / mres - .5; gpy = (ORG - MY) / mres - .5
tx0, tx1 = int(gpx.min() // TS), int(gpx.max() // TS); ty0, ty1 = int(gpy.min() // TS), int(gpy.max() // TS)
mos = np.zeros(((ty1 - ty0 + 1) * TS, (tx1 - tx0 + 1) * TS), np.float64)
for ty in range(ty0, ty1 + 1):
    for tx in range(tx0, tx1 + 1):
        p = os.path.join(RAW, f'{Z}_{tx}_{ty}.png')
        if not os.path.exists(p):
            open(p, 'wb').write(urllib.request.urlopen(urllib.request.Request(f'https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{Z}/{tx}/{ty}.png', headers={'User-Agent': 'showcase-build'}), timeout=60).read())
        t = np.asarray(Image.open(p).convert('RGB')).astype(np.float64)
        mos[(ty - ty0) * TS:(ty - ty0 + 1) * TS, (tx - tx0) * TS:(tx - tx0 + 1) * TS] = t[..., 0] * 256 + t[..., 1] + t[..., 2] / 256 - 32768
elev = map_coordinates(mos, [gpy - ty0 * TS, gpx - tx0 * TS], order=1, mode='nearest')
q = np.round(elev * 2) / 2; v = np.round((q + 32768) * 256).astype(np.int64)
Image.fromarray(np.stack([(v >> 16) & 255, (v >> 8) & 255, v & 255], -1).astype(np.uint8)).save(os.path.join(OUT, 'dem.png'), optimize=True)
w_, s_ = to_ll.transform(bounds[0], bounds[1]); e_, n_ = to_ll.transform(bounds[2], bounds[3])
ll_bounds = [min(w_, to_ll.transform(bounds[0], bounds[3])[0]), s_, max(e_, to_ll.transform(bounds[2], bounds[1])[0]), n_]
gv = to_utm.transform(-120.9511, 40.1396); gi = (int((north - gv[1]) / RES), int((gv[0] - west) / RES))
rd = lambda k, t: {'file': files[f'{k}_{t}']}
common = {'width': N, 'height': N, 'bounds': ll_bounds, 'boundsProjected': list(bounds), 'projection': 'EPSG:32610', 'cellSizeM': RES}
man = {'id': 'dixie-fire', 'version': 1, 'kind': 'raster', 'count': N * N, 'bbox': ll_bounds, 'crs': 'EPSG:32610',
  'raster': {**common, 'file': 'dem.png', 'encoding': 'terrarium', 'noData': None, 'unit': 'm', 'verticalQuantisationM': 0.5},
  'rasters': {
    'dem': {**common, 'file': 'dem.png', 'encoding': 'terrarium', 'unit': 'm', 'description': 'AWS Terrarium (3DEP ~10 m) bilinearly resampled to the 20 m grid'},
    **{f'{k}_{t}': {**common, 'file': files[f'{k}_{t}'], 'encoding': 'uint16-bin', 'dtype': 'uint16', 'scale': 0.0001, 'offset': 0, 'noData': 0,
                    'band': BANDS[k], 'date': SCENES[t][1], 'scene': SCENES[t][0], 'description': 'surface reflectance x 10000 (L2A, baseline 03.01, no BOA offset)'}
       for t in SCENES for k in BANDS},
    **{f'scl_{t}': {**common, 'file': f'scl_{t}.png', 'encoding': 'uint8-classes', 'date': SCENES[t][1],
                    'classes': {'0': 'no data', '1': 'saturated', '2': 'dark area', '3': 'cloud shadow', '4': 'vegetation', '5': 'not vegetated', '6': 'water', '7': 'unclassified', '8': 'cloud medium', '9': 'cloud high', '10': 'thin cirrus', '11': 'snow'},
                    'cloudMaskClasses': [3, 8, 9, 10]} for t in SCENES},
    'worldcover': {**common, 'file': 'worldcover.png', 'encoding': 'uint8-classes', 'classes': {str(k): v for k, v in LEG.items()}, 'description': 'ESA WorldCover 2021 v200, mode-resampled from 10 m'}},
  'properties': {
    'attribution': 'Contains modified Copernicus Sentinel data 2021 (ESA); ESA WorldCover 2021 (c) ESA, CC BY 4.0; terrain: AWS Terrain Tiles / USGS 3DEP',
    'origin': 'upper-left; rows run north to south; grid is EPSG:32610 (UTM 10N), 20 m, so analyses can use uniform cell size 20 m',
    'greenvilleGridRowCol': list(gi), 'validation': validation, 'sceneStats': stats, 'worldCoverShare': wc_hist,
    'storyNotes': [
      f"Window 15 km x 15 km at 20 m centred near Greenville CA; before = {SCENES['before'][1]} (S2B), after = {SCENES['after'][1]} (S2B), both under {max(s['cloudOrShadowFraction'] for s in stats.values())*100:.2f}% cloud in the window.",
      'The Dixie Fire ignited 2021-07-13 near Cresta Dam, destroyed Greenville on 2021-08-04 and burned ~963,000 acres (second largest California fire on record); the pre date is the day it started.',
      f"{validation['fractionAbove']['dNBR>0.27 (moderate-low+)']*100:.0f}% of valid window pixels exceed dNBR 0.27 (moderate-low severity or worse); mean dNBR {validation['meanDNBR']:.2f}.",
      f"Mean NDVI falls from {stats['before']['meanNDVI']:.2f} to {stats['after']['meanNDVI']:.2f}; Greenville (grid row/col {gi[0]},{gi[1]}) lies inside the burn.",
      'WorldCover shows conifer tree cover dominating with grass/shrub in Indian Valley: zonal dNBR per class and slope-dependent debris-flow suitability are natural follow-ups.',
      'Same grid for every layer: no reprojection needed to combine DEM, WorldCover and bands.']}}
json.dump(man, open(os.path.join(OUT, 'manifest.json'), 'w'), indent=1)
print(json.dumps({'validation': validation, 'stats': stats, 'bbox': ll_bounds, 'wc': wc_hist}, indent=1))
for f in sorted(os.listdir(OUT)): print(f, os.path.getsize(os.path.join(OUT, f)))
