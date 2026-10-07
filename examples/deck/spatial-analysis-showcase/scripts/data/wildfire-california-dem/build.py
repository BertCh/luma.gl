"""wildfire-california-dem: northern and central California DEM from key-free Terrarium tiles.
Source: AWS Open Data `elevation-tiles-prod` (Mapzen/Tilezen Terrain Tiles, zoom 9, Web Mercator).
Output: Terrarium PNG quantised to 1 m, ocean and water at or below 0 m written as noData (-9999).
Run with a Python that has numpy and Pillow:  RAW=<empty dir> python3 -I build.py"""
import io, json, math, os, urllib.request
import numpy as np
from PIL import Image

RAW = os.environ.get('RAW') or os.path.join(os.environ.get('TMPDIR', '/tmp'), 'wildfire-california-dem-raw')
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../public/data/wildfire-california-dem')
os.makedirs(RAW, exist_ok=True)
os.makedirs(OUT, exist_ok=True)

Z, TS = 9, 256
WEST, SOUTH, EAST, NORTH = -123.3, 36.4, -120.1, 41.0     # requested lon/lat window
R = 6378137.0
ORIGIN = math.pi * R
RES = 2 * ORIGIN / (TS * 2 ** Z)                           # Web Mercator metres per pixel (equatorial)


def merc(lon, lat):
    return math.radians(lon) * R, R * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))


def unmerc(x, y):
    return math.degrees(x / R), math.degrees(2 * math.atan(math.exp(y / R)) - math.pi / 2)


x0m, y0m = merc(WEST, NORTH)
x1m, y1m = merc(EAST, SOUTH)
px0 = int(math.floor((x0m + ORIGIN) / RES))
py0 = int(math.floor((ORIGIN - y0m) / RES))
px1 = int(math.ceil((x1m + ORIGIN) / RES))
py1 = int(math.ceil((ORIGIN - y1m) / RES))
W, H = px1 - px0, py1 - py0


def tile(x, y):
    path = os.path.join(RAW, f'{Z}_{x}_{y}.png')
    if not os.path.exists(path):
        url = f'https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{Z}/{x}/{y}.png'
        with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'showcase-build'}), timeout=60) as r:
            open(path, 'wb').write(r.read())
    return np.asarray(Image.open(path).convert('RGB'))


mosaic = np.zeros((H, W, 3), np.uint8)
for ty in range(py0 // TS, (py1 - 1) // TS + 1):
    for tx in range(px0 // TS, (px1 - 1) // TS + 1):
        t = tile(tx, ty)
        gx0, gy0 = tx * TS, ty * TS
        x0, x1 = max(px0, gx0), min(px1, gx0 + TS)
        y0, y1 = max(py0, gy0), min(py1, gy0 + TS)
        mosaic[y0 - py0:y1 - py0, x0 - px0:x1 - px0] = t[y0 - gy0:y1 - gy0, x0 - gx0:x1 - gx0]

h = mosaic[..., 0].astype(np.float64) * 256 + mosaic[..., 1] + mosaic[..., 2] / 256 - 32768
q = np.round(h)
# Repair isolated spikes (source voids): a pixel more than 500 m from its 3x3 median takes the median.
pad = np.pad(q, 1, mode='edge')
neighbours = np.stack([pad[dy:dy + H, dx:dx + W] for dy in range(3) for dx in range(3)])
median = np.median(neighbours, axis=0)
spikes = np.abs(q - median) > 500
q[spikes] = median[spikes]
water = q <= 0
SENTINEL = -9999
v = np.where(water, SENTINEL + 32768, q + 32768).astype(np.int64) * 256
rgb = np.stack([(v >> 16) & 255, (v >> 8) & 255, v & 255], -1).astype(np.uint8)
Image.fromarray(rgb).save(os.path.join(OUT, 'dem.png'), optimize=True)

mb = [px0 * RES - ORIGIN, ORIGIN - py1 * RES, px1 * RES - ORIGIN, ORIGIN - py0 * RES]   # w s e n, Mercator metres
w, s = unmerc(mb[0], mb[1])
e, n = unmerc(mb[2], mb[3])
mid = (s + n) / 2
land = q[~water]
manifest = {
    'id': 'wildfire-california-dem', 'version': 1, 'kind': 'raster', 'count': int(W * H), 'bbox': [w, s, e, n], 'crs': 'EPSG:4326',
    'raster': {
        'file': 'dem.png', 'encoding': 'terrarium', 'width': W, 'height': H, 'bounds': [w, s, e, n],
        'boundsMercator': mb, 'projection': 'EPSG:3857', 'cellSizeMercatorM': RES,
        'cellSizeGroundM': RES * math.cos(math.radians(mid)), 'noData': -9999, 'unit': 'm',
        'verticalQuantisationM': 1, 'spikesRepaired': int(spikes.sum()), 'webMercatorTile': {'zoom': Z, 'tileSize': TS, 'originPixel': [px0, py0]}},
    'properties': {
        'source': 'Terrain Tiles on AWS (Mapzen/Tilezen: USGS NED, SRTM, GMTED2010, ETOPO1 and others), Terrarium encoding, zoom 9',
        'landFraction': float(land.size / q.size),
        'minElevationM': float(land.min()), 'maxElevationM': float(land.max()),
        'notes': 'Cells at or below 0 m (ocean, bays, some delta land) are noData.'}}
json.dump(manifest, open(os.path.join(OUT, 'manifest.json'), 'w'), indent=1)
print('size', W, H, 'png bytes', os.path.getsize(os.path.join(OUT, 'dem.png')), 'bbox', [round(x, 3) for x in manifest['bbox']],
      'ground m', RES * math.cos(math.radians(mid)), 'land', manifest['properties']['landFraction'], land.min(), land.max())
