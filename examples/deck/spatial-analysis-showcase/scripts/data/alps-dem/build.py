"""alps-dem: Zermatt / Matterhorn / Gornergrat DEM, Terrarium PNG in Web Mercator.
Source: Mapterhorn tiles (swissALTI3D OGD in Switzerland). Re-runnable; caches tiles in raw dir."""
import io, json, math, os, sys, urllib.request
import numpy as np
from PIL import Image
RAW = os.environ.get('RAW', '/private/tmp/claude-501/-Users-robertchristie-Documents-GitHub-deck-gl-master/85599621-aabc-441b-ab0f-35747be6f781/scratchpad/showcase/raw/alps-dem')
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../public/data/alps-dem')
os.makedirs(RAW, exist_ok=True); os.makedirs(OUT, exist_ok=True)
Z, TS, N = 13, 512, 2048          # zoom, tile size, output px
CENTER = (7.742, 45.985)          # lon, lat
R = 6378137.0; ORIGIN = math.pi * R
def merc(lon, lat): return (math.radians(lon) * R, R * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2)))
def unmerc(x, y): return (math.degrees(x / R), math.degrees(2 * math.atan(math.exp(y / R)) - math.pi / 2))
res = 2 * ORIGIN / (TS * 2 ** Z)
cx, cy = merc(*CENTER)
px0 = round((cx + ORIGIN) / res - N / 2); py0 = round((ORIGIN - cy) / res - N / 2)  # global pixel of top-left
def tile(x, y):
    p = os.path.join(RAW, f'{Z}_{x}_{y}.webp')
    if not os.path.exists(p):
        req = urllib.request.Request(f'https://tiles.mapterhorn.com/{Z}/{x}/{y}.webp', headers={'User-Agent': 'showcase-build'})
        open(p, 'wb').write(urllib.request.urlopen(req, timeout=60).read())
    return np.asarray(Image.open(p).convert('RGB'))
mosaic = np.zeros((N, N, 3), np.uint8)
for ty in range(py0 // TS, (py0 + N - 1) // TS + 1):
    for tx in range(px0 // TS, (px0 + N - 1) // TS + 1):
        t = tile(tx, ty)
        gx0, gy0 = tx * TS, ty * TS
        x0, x1 = max(px0, gx0), min(px0 + N, gx0 + TS); y0, y1 = max(py0, gy0), min(py0 + N, gy0 + TS)
        mosaic[y0 - py0:y1 - py0, x0 - px0:x1 - px0] = t[y0 - gy0:y1 - gy0, x0 - gx0:x1 - gx0]
h = mosaic[..., 0].astype(np.float64) * 256 + mosaic[..., 1] + mosaic[..., 2] / 256 - 32768
# quantise to 0.25 m (B channel in {0,64,128,192}) so the PNG deflates well; error <= 0.125 m
q = np.round(h * 4) / 4
v = np.round((q + 32768) * 256).astype(np.int64)
rgb = np.stack([(v >> 16) & 255, (v >> 8) & 255, v & 255], -1).astype(np.uint8)
Image.fromarray(rgb).save(os.path.join(OUT, 'dem.png'), optimize=True)
mb = [(px0) * res - ORIGIN, ORIGIN - (py0 + N) * res, (px0 + N) * res - ORIGIN, ORIGIN - py0 * res]  # w s e n (mercator m)
w, s = unmerc(mb[0], mb[1]); e, n = unmerc(mb[2], mb[3])
mid_lat = (s + n) / 2
# stats + story notes
iy, ix = np.unravel_index(np.argmax(q), q.shape)
def px2ll(ix, iy): return unmerc(mb[0] + (ix + .5) * res, mb[3] - (iy + .5) * res)
peak = px2ll(ix, iy)
def hat(lon, lat):
    x, y = merc(lon, lat); return float(q[int((mb[3] - y) / res), int((x - mb[0]) / res)])
sites = {'Matterhorn': (7.6586, 45.9766), 'Gornergrat': (7.7843, 45.9832), 'Zermatt': (7.7491, 46.0207), 'Dufourspitze': (7.8667, 45.9369)}
inside = {k: {'lon': a, 'lat': b, 'elevationM': hat(a, b)} for k, (a, b) in sites.items() if w < a < e and s < b < n}
man = {
  'id': 'alps-dem', 'version': 1, 'kind': 'raster', 'count': N * N, 'bbox': [w, s, e, n], 'crs': 'EPSG:4326',
  'raster': {'file': 'dem.png', 'encoding': 'terrarium', 'width': N, 'height': N, 'bounds': [w, s, e, n],
             'boundsMercator': mb, 'projection': 'EPSG:3857', 'cellSizeMercatorM': res,
             'cellSizeGroundM': res * math.cos(math.radians(mid_lat)), 'noData': None, 'unit': 'm',
             'verticalQuantisationM': 0.25, 'webMercatorTile': {'zoom': Z, 'tileSize': TS, 'originPixel': [px0, py0]}},
  'properties': {
    'source': 'Mapterhorn terrain tiles (swissALTI3D, (c) swisstopo OGD; Copernicus GLO-30 outside Switzerland)',
    'observers': inside,
    'minElevationM': float(q.min()), 'maxElevationM': float(q.max()), 'highestPoint': {'lon': peak[0], 'lat': peak[1], 'elevationM': float(q.max())},
    'storyNotes': [
      f'Window ~{(mb[2]-mb[0])*math.cos(math.radians(mid_lat))/1000:.1f} km square at {res*math.cos(math.radians(mid_lat)):.1f} m ground resolution (Web Mercator {res:.1f} m/px); relief {q.min():.0f} to {q.max():.0f} m.',
      'Matterhorn (4478 m) rises ~2800 m above Zermatt (1600 m): the steepest, most iconic horn in the Alps, ideal for slope, curvature and summit detection.',
      'Gornergrat (3089 m) is the viewshed observer: it looks straight across the Gorner Glacier to Monte Rosa and Matterhorn.',
      'Gorner and Findel glaciers give flat, wide ice tongues between sharp arete ridges: geomorphons separate ridge, shoulder, hollow and valley cleanly.',
      'Mattertal valley floor at Zermatt is a narrow glacial U-valley; lateral moraines and cirques show up as TPI hollows.',
      'Data is Terrarium-quantised to 0.25 m; Web Mercator pixels, so use cellSizeMode web-mercator with northEdge/southEdge from boundsMercator.']}}
json.dump(man, open(os.path.join(OUT, 'manifest.json'), 'w'), indent=1)
print('png bytes', os.path.getsize(os.path.join(OUT, 'dem.png')), man['bbox'], inside, q.min(), q.max())
