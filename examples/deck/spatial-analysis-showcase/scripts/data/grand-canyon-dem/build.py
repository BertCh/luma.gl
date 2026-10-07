"""grand-canyon-dem: central Grand Canyon (Bright Angel / Phantom Ranch / Colorado River), Terrarium PNG, Web Mercator.
Source: AWS Terrain Tiles (Mapzen/Tilezen; USGS 3DEP/NED ~10 m in CONUS), z13 256 px tiles."""
import json, math, os, urllib.request
import numpy as np
from PIL import Image
RAW = '/private/tmp/claude-501/-Users-robertchristie-Documents-GitHub-deck-gl-master/85599621-aabc-441b-ab0f-35747be6f781/scratchpad/showcase/raw/grand-canyon-dem'
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../public/data/grand-canyon-dem')
os.makedirs(RAW, exist_ok=True); os.makedirs(OUT, exist_ok=True)
Z, TS, N = 13, 256, 2048
CENTER = (-112.10, 36.10)
R = 6378137.0; ORIGIN = math.pi * R
def merc(lon, lat): return (math.radians(lon) * R, R * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2)))
def unmerc(x, y): return (math.degrees(x / R), math.degrees(2 * math.atan(math.exp(y / R)) - math.pi / 2))
res = 2 * ORIGIN / (TS * 2 ** Z)
cx, cy = merc(*CENTER)
px0 = round((cx + ORIGIN) / res - N / 2); py0 = round((ORIGIN - cy) / res - N / 2)
def tile(x, y):
    p = os.path.join(RAW, f'{Z}_{x}_{y}.png')
    if not os.path.exists(p):
        req = urllib.request.Request(f'https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{Z}/{x}/{y}.png', headers={'User-Agent': 'showcase-build'})
        open(p, 'wb').write(urllib.request.urlopen(req, timeout=60).read())
    return np.asarray(Image.open(p).convert('RGB'))
mosaic = np.zeros((N, N, 3), np.uint8)
for ty in range(py0 // TS, (py0 + N - 1) // TS + 1):
    for tx in range(px0 // TS, (px0 + N - 1) // TS + 1):
        t = tile(tx, ty); gx0, gy0 = tx * TS, ty * TS
        x0, x1 = max(px0, gx0), min(px0 + N, gx0 + TS); y0, y1 = max(py0, gy0), min(py0 + N, gy0 + TS)
        mosaic[y0 - py0:y1 - py0, x0 - px0:x1 - px0] = t[y0 - gy0:y1 - gy0, x0 - gx0:x1 - gx0]
h = mosaic[..., 0].astype(np.float64) * 256 + mosaic[..., 1] + mosaic[..., 2] / 256 - 32768
q = np.round(h * 2) / 2     # 0.5 m vertical quantisation (source is ~1 m accurate at best)
v = np.round((q + 32768) * 256).astype(np.int64)
rgb = np.stack([(v >> 16) & 255, (v >> 8) & 255, v & 255], -1).astype(np.uint8)
Image.fromarray(rgb).save(os.path.join(OUT, 'dem.png'), optimize=True)
mb = [px0 * res - ORIGIN, ORIGIN - (py0 + N) * res, (px0 + N) * res - ORIGIN, ORIGIN - py0 * res]
w, s = unmerc(mb[0], mb[1]); e, n = unmerc(mb[2], mb[3]); ml = (s + n) / 2
def hat(lon, lat):
    x, y = merc(lon, lat); return float(q[int((mb[3] - y) / res), int((x - mb[0]) / res)])
sites = {'Grand Canyon Village': (-112.1401, 36.0544), 'Phantom Ranch': (-112.0953, 36.1070), 'Colorado River at Bright Angel Creek': (-112.0930, 36.1060),
         'Yavapai Point': (-112.1180, 36.0660), 'North Rim Bright Angel Point': (-112.0525, 36.1920)}
inside = {k: {'lon': a, 'lat': b, 'elevationM': hat(a, b)} for k, (a, b) in sites.items() if w < a < e and s < b < n}
man = {'id': 'grand-canyon-dem', 'version': 1, 'kind': 'raster', 'count': N * N, 'bbox': [w, s, e, n], 'crs': 'EPSG:4326',
  'raster': {'file': 'dem.png', 'encoding': 'terrarium', 'width': N, 'height': N, 'bounds': [w, s, e, n], 'boundsMercator': mb, 'projection': 'EPSG:3857',
             'cellSizeMercatorM': res, 'cellSizeGroundM': res * math.cos(math.radians(ml)), 'noData': None, 'unit': 'm', 'verticalQuantisationM': 0.5,
             'webMercatorTile': {'zoom': Z, 'tileSize': TS, 'originPixel': [px0, py0]}},
  'properties': {'source': 'AWS Terrain Tiles (Mapzen/Tilezen) built from USGS 3DEP/NED', 'places': inside,
    'minElevationM': float(q.min()), 'maxElevationM': float(q.max()),
    'storyNotes': [
      f'~{(mb[2]-mb[0])*math.cos(math.radians(ml))/1000:.0f} km window at {res*math.cos(math.radians(ml)):.1f} m ground resolution; relief {q.min():.0f} to {q.max():.0f} m (about 1.9 km of canyon depth rim to river).',
      'Colorado River runs through the inner gorge near the bottom of the window: flow accumulation should light it up as the trunk stream; Bright Angel Creek is the main north-bank tributary at Phantom Ranch.',
      'Stratified Redwall/Supai/Tonto terraces make contours bunch into cliffs with wide flat benches: the Tonto Platform is a natural cost-distance corridor.',
      'South Rim sits near 2100 m and North Rim near 2500 m: watersheds on each rim drain through different side canyons, a clean divide demo.',
      'Hiking cost distance: Bright Angel Trail and the North/South Kaibab trails exploit side-canyon fault lines; a slope-based cost surface should rediscover similar routes from Grand Canyon Village to Phantom Ranch.',
      'Web Mercator pixels (Terrarium); use cellSizeMode web-mercator with northEdge/southEdge from boundsMercator.']}}
json.dump(man, open(os.path.join(OUT, 'manifest.json'), 'w'), indent=1)
print(os.path.getsize(os.path.join(OUT, 'dem.png')), man['bbox'], inside, q.min(), q.max())
