"""gfs-wind: NOAA GFS 0.25 deg 10 m and 250 hPa u/v wind around Hurricane Helene's landfall (Perry FL, 2024-09-27 ~03:10Z).
Run 2024-09-26 12Z, hourly f000..f023 (10 m) and 3-hourly f000..f021 (250 hPa jet). Quantised 8-bit RG PNGs, min/max in manifest.
GRIB2 variables fetched through .idx byte ranges, decoded with eccodes (python wheel)."""
import json, os, re, tempfile, urllib.request
import numpy as np
from PIL import Image
import eccodes
RAW = '/private/tmp/claude-501/-Users-robertchristie-Documents-GitHub-deck-gl-master/85599621-aabc-441b-ab0f-35747be6f781/scratchpad/showcase/raw/gfs-wind'
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../public/data/gfs-wind')
os.makedirs(RAW, exist_ok=True); os.makedirs(OUT, exist_ok=True)
DATE, CYC = '20240926', '12'
BASE = f'https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.{DATE}/{CYC}/atmos/gfs.t{CYC}z.pgrb2.0p25.f'
W, S_, E, N_ = -105.0, 12.0, -60.0, 50.0       # lon/lat crop (cell centres on 0.25 deg grid)
def get(url, rng=None):
    h = {'User-Agent': 'showcase-build'}; 
    if rng: h['Range'] = f'bytes={rng[0]}-{rng[1]}'
    return urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=120).read()
def fields(fh, wanted):
    idx = get(BASE + f'{fh:03d}.idx').decode().strip().split('\n')
    rows = [l.split(':') for l in idx]; out = {}
    for i, r in enumerate(rows):
        key = (r[3], r[4])
        if key in wanted:
            start = int(r[1]); end = int(rows[i + 1][1]) - 1 if i + 1 < len(rows) else ''
            cache = os.path.join(RAW, f'f{fh:03d}_{r[3]}_{r[4].replace(" ", "")}.grib2')
            if not os.path.exists(cache): open(cache, 'wb').write(get(BASE + f'{fh:03d}', (start, end)))
            with open(cache, 'rb') as f:
                g = eccodes.codes_grib_new_from_file(f)
                ni, nj = eccodes.codes_get(g, 'Ni'), eccodes.codes_get(g, 'Nj')
                vals = eccodes.codes_get_values(g).reshape(nj, ni)
                lat0, lat1 = eccodes.codes_get(g, 'latitudeOfFirstGridPointInDegrees'), eccodes.codes_get(g, 'latitudeOfLastGridPointInDegrees')
                eccodes.codes_release(g)
            out[key] = (vals, lat0, lat1)
    return out
def crop(a, lat0, lat1):
    # global 0.25 grid, lon 0..359.75, lat0 = 90 (north first)
    lats = np.linspace(lat0, lat1, a.shape[0]); lons = np.arange(a.shape[1]) * 0.25 - 360 * (np.arange(a.shape[1]) * 0.25 >= 180)
    ri = np.where((lats <= N_ + 1e-6) & (lats >= S_ - 1e-6))[0]; ci = np.where((lons >= W - 1e-6) & (lons <= E + 1e-6))[0]
    ci = ci[np.argsort(lons[ci])]
    return a[np.ix_(ri, ci)], lats[ri], lons[ci]
def build(name, level, steps, vmax_hint):
    us, vs = [], []
    for fh in steps:
        f = fields(fh, {('UGRD', level), ('VGRD', level)})
        u, la, lo = crop(*f[('UGRD', level)]); v, _, _ = crop(*f[('VGRD', level)]); us.append(u); vs.append(v)
    u = np.stack(us); v = np.stack(vs)
    umin, umax, vmin, vmax = float(u.min()), float(u.max()), float(v.min()), float(v.max())
    umin, vmin = np.floor(umin), np.floor(vmin); umax, vmax = np.ceil(umax), np.ceil(vmax)
    fl = []
    for i in range(len(steps)):
        rg = np.zeros(u[i].shape + (3,), np.uint8)
        rg[..., 0] = np.round((u[i] - umin) / (umax - umin) * 255); rg[..., 1] = np.round((v[i] - vmin) / (vmax - vmin) * 255)
        fn = f'{name}_{i:02d}.png'; Image.fromarray(rg, 'RGB').save(os.path.join(OUT, fn), optimize=True); fl.append(fn)
    speed = np.hypot(u, v)
    return {'files': fl, 'forecastHours': list(steps), 'uRange': [umin, umax], 'vRange': [vmin, vmax], 'decode': 'u = uMin + R/255*(uMax-uMin); v = vMin + G/255*(vMax-vMin)',
            'maxSpeedMs': float(speed.max()), 'maxSpeedFrame': int(np.unravel_index(speed.argmax(), speed.shape)[0]), 'maxSpeedLonLat': [float(lo[np.unravel_index(speed.argmax(), speed.shape)[2]]), float(la[np.unravel_index(speed.argmax(), speed.shape)[1]])],
            'width': u.shape[2], 'height': u.shape[1], 'rowOrder': 'north to south', 'lats': [float(la[0]), float(la[-1])], 'lons': [float(lo[0]), float(lo[-1])]}
w10 = build('wind10m', '10 m above ground', list(range(0, 24)), 40)
jet = build('jet250', '250 mb', list(range(0, 24, 3)), 80)
man = {'id': 'gfs-wind', 'version': 1, 'kind': 'raster', 'count': w10['width'] * w10['height'], 'bbox': [W, S_, E, N_], 'crs': 'EPSG:4326',
  'raster': {'file': w10['files'][0], 'encoding': 'rg-uv-8bit', 'width': w10['width'], 'height': w10['height'], 'bounds': [W - .125, S_ - .125, E + .125, N_ + .125], 'cellSizeDeg': 0.25, 'unit': 'm/s', 'noData': None},
  'properties': {'model': 'NOAA GFS 0.25 deg', 'run': '2024-09-26T12:00:00Z', 'validStart': '2024-09-26T12:00:00Z', 'wind10m': {**w10, 'stepHours': 1, 'level': '10 m above ground'},
    'jet250hPa': {**jet, 'stepHours': 3, 'level': '250 hPa'},
    'attribution': 'NOAA / NCEP Global Forecast System (public domain)',
    'storyNotes': [
      'Hurricane Helene made landfall near Perry, Florida (Big Bend) as a category 4 at ~03:10Z on 2024-09-27: it sits at frames 14-15 of the hourly 10 m stack (run 2024-09-26 12Z).',
      f"Peak GFS 10 m speed in the crop is {w10['maxSpeedMs']:.0f} m/s at frame {w10['maxSpeedFrame']} near lon/lat {w10['maxSpeedLonLat'][0]:.2f}, {w10['maxSpeedLonLat'][1]:.2f}; the cyclonic eye and outer rainbands make a vivid particle-advection and LIC spiral.",
      f"The 250 hPa jet reaches {jet['maxSpeedMs']:.0f} m/s in the crop; a trough over the central US steered Helene north, a classic streamline/jet story.",
      'Grid is 0.25 deg, 181 x 153 cells, rows north to south, cell centres at 0.25 deg multiples; bounds in the manifest extend half a cell.',
      '8-bit quantisation: step is (max-min)/255 per component (about 0.3-0.4 m/s); use bilinear sampling of decoded u/v in the shader, never of the encoded colour pair across the dateline.']}}
json.dump(man, open(os.path.join(OUT, 'manifest.json'), 'w'), indent=1)
tot = sum(os.path.getsize(os.path.join(OUT, f)) for f in os.listdir(OUT)); print('total bytes', tot, w10['maxSpeedMs'], w10['maxSpeedFrame'], w10['maxSpeedLonLat'], jet['maxSpeedMs'], w10['uRange'], w10['vRange'], w10['width'], w10['height'])
