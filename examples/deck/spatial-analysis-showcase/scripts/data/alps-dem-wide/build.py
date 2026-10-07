"""alps-dem-wide: the whole Gornergrat view (Zermatt to Monte Rosa) as a Terrarium PNG in Web Mercator.

Source: Mapterhorn terrain tiles. Re-runnable; tiles are cached in the raw directory.
Usage: python -I build.py [south] [quantisation_m]   (defaults: 45.86, 0.25)
"""
import json, math, os, sys, urllib.request
import numpy as np
from PIL import Image

RAW = os.environ.get('RAW', '/private/tmp/claude-501/-Users-robertchristie-Documents-GitHub-deck-gl-master/'
                     '14421614-4041-4f9e-a118-5d838928ca9c/scratchpad/fidelity/raw/alps-dem-wide')
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../public/data/alps-dem-wide')
USER_AGENT = 'luma-showcase-data-build/1.0 (https://github.com/visgl/luma.gl)'
os.makedirs(RAW, exist_ok=True)
os.makedirs(OUT, exist_ok=True)

ZOOM, TILE_SIZE = 12, 512
WEST, EAST, NORTH = 7.58, 7.98, 46.12
SOUTH = float(sys.argv[1]) if len(sys.argv) > 1 else 45.86
QUANT = float(sys.argv[2]) if len(sys.argv) > 2 else 0.25

R = 6378137.0
ORIGIN = math.pi * R


def merc(lon, lat):
    return (math.radians(lon) * R, R * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2)))


def unmerc(x, y):
    return (math.degrees(x / R), math.degrees(2 * math.atan(math.exp(y / R)) - math.pi / 2))


res = 2 * ORIGIN / (TILE_SIZE * 2 ** ZOOM)
x_west, y_south = merc(WEST, SOUTH)
x_east, y_north = merc(EAST, NORTH)
# Snap outward to whole Mercator pixels; (px0, py0) is the global pixel of the top-left corner.
px0 = math.floor((x_west + ORIGIN) / res)
px1 = math.ceil((x_east + ORIGIN) / res)
py0 = math.floor((ORIGIN - y_north) / res)
py1 = math.ceil((ORIGIN - y_south) / res)
W, H = px1 - px0, py1 - py0


def tile(x, y):
    path = os.path.join(RAW, f'{ZOOM}_{x}_{y}.webp')
    if not os.path.exists(path):
        request = urllib.request.Request(f'https://tiles.mapterhorn.com/{ZOOM}/{x}/{y}.webp',
                                         headers={'User-Agent': USER_AGENT})
        with open(path, 'wb') as f:
            f.write(urllib.request.urlopen(request, timeout=60).read())
    return np.asarray(Image.open(path).convert('RGB'))


mosaic = np.zeros((H, W, 3), np.uint8)
for ty in range(py0 // TILE_SIZE, (py1 - 1) // TILE_SIZE + 1):
    for tx in range(px0 // TILE_SIZE, (px1 - 1) // TILE_SIZE + 1):
        t = tile(tx, ty)
        gx0, gy0 = tx * TILE_SIZE, ty * TILE_SIZE
        x0, x1 = max(px0, gx0), min(px1, gx0 + TILE_SIZE)
        y0, y1 = max(py0, gy0), min(py1, gy0 + TILE_SIZE)
        mosaic[y0 - py0:y1 - py0, x0 - px0:x1 - px0] = t[y0 - gy0:y1 - gy0, x0 - gx0:x1 - gx0]

h = mosaic[..., 0].astype(np.float64) * 256 + mosaic[..., 1] + mosaic[..., 2] / 256 - 32768
# Quantise so the PNG deflates well; error <= QUANT / 2.
q = np.round(h / QUANT) * QUANT
v = np.round((q + 32768) * 256).astype(np.int64)
rgb = np.stack([(v >> 16) & 255, (v >> 8) & 255, v & 255], -1).astype(np.uint8)
png_path = os.path.join(OUT, 'dem.png')
Image.fromarray(rgb).save(png_path, optimize=True)

mb = [px0 * res - ORIGIN, ORIGIN - py1 * res, px1 * res - ORIGIN, ORIGIN - py0 * res]  # w s e n, mercator m
w, s = unmerc(mb[0], mb[1])
e, n = unmerc(mb[2], mb[3])
mid_lat = (s + n) / 2
iy, ix = np.unravel_index(np.argmax(q), q.shape)


def px2ll(column, row):
    return unmerc(mb[0] + (column + .5) * res, mb[3] - (row + .5) * res)


def elevation_at(lon, lat):
    x, y = merc(lon, lat)
    return float(q[int((mb[3] - y) / res), int((x - mb[0]) / res)])


peak = px2ll(ix, iy)
sites = {'Gornergrat': (7.7843, 45.9832), 'Matterhorn': (7.6586, 45.9766),
         'Zermatt': (7.7466, 46.0175), 'Dufourspitze': (7.8668, 45.9369)}
observers = {k: {'lon': a, 'lat': b, 'elevationM': elevation_at(a, b)}
             for k, (a, b) in sites.items() if w < a < e and s < b < n}
km_x = (mb[2] - mb[0]) * math.cos(math.radians(mid_lat)) / 1000
km_y = (mb[3] - mb[1]) * math.cos(math.radians(mid_lat)) / 1000
ground = res * math.cos(math.radians(mid_lat))
manifest = {
    'id': 'alps-dem-wide', 'version': 1, 'kind': 'raster', 'count': W * H, 'bbox': [w, s, e, n], 'crs': 'EPSG:4326',
    'raster': {'file': 'dem.png', 'encoding': 'terrarium', 'width': W, 'height': H, 'bounds': [w, s, e, n],
               'boundsMercator': mb, 'projection': 'EPSG:3857', 'cellSizeMercatorM': res,
               'cellSizeGroundM': ground, 'noData': None, 'unit': 'm',
               'verticalQuantisationM': QUANT,
               'webMercatorTile': {'zoom': ZOOM, 'tileSize': TILE_SIZE, 'originPixel': [px0, py0]}},
    'properties': {
        'source': 'Mapterhorn terrain tiles (swissALTI3D, (c) swisstopo OGD, in Switzerland; other open DEMs elsewhere, see README)',
        'observers': observers,
        'minElevationM': float(q.min()), 'maxElevationM': float(q.max()),
        'highestPoint': {'lon': peak[0], 'lat': peak[1], 'elevationM': float(q.max())},
        'storyNotes': [
            f'Window ~{km_x:.1f} x {km_y:.1f} km at {ground:.1f} m ground resolution '
            f'(Web Mercator {res:.1f} m/px, zoom {ZOOM}); relief {q.min():.0f} to {q.max():.0f} m.',
            'Covers the whole Gornergrat view: Mattertal and Zermatt, the Matterhorn, the Gorner, Findel and '
            'Zmutt glaciers, Monte Rosa with Dufourspitze (4634 m), and the Weisshorn and Dom to the north.',
            'Gornergrat (3089 m) looks across the Gorner Glacier at 360 degrees: the natural observer for '
            'viewshed and horizon stories; the Matterhorn and Dufourspitze are the far targets.',
            f'Data is Terrarium-quantised to {QUANT} m; Web Mercator pixels, so use cellSizeMode web-mercator '
            'with northEdge/southEdge from boundsMercator.']}}
with open(os.path.join(OUT, 'manifest.json'), 'w') as f:
    json.dump(manifest, f, indent=1)
print('png bytes', os.path.getsize(png_path), 'size', W, H, 'bbox', [w, s, e, n], 'quant', QUANT)
print('min/max', q.min(), q.max(), 'highest', peak)
print(observers)
