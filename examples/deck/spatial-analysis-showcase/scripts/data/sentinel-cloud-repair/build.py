"""sentinel-cloud-repair: two cloudy/clear Sentinel-2 Level-2A pairs (Earth Search Collection 1) on a shared
1536 x 1536 grid, with a cloud + cloud-shadow mask built by the s2cloudless-style recipe:

1. cloud   = Sen2Cor cloud probability (CLD_20m) >= 50 %
2. shadow  = dark NIR (B08 reflectance < 0.15, not SCL water) inside the cloud mask projected away from the sun
             for cloud tops up to 8 km (shift = height * tan(solar zenith) along the solar azimuth + 180 deg)
3. cleanup = 40 m erosion (drops speckle) then 100 m dilation (buffers mixed edges), on the 20 m grid

Because every target has a clear prior a few days earlier, two multitemporal refinements (as in MAJA and Tmask)
catch what the single-date recipe misses: cloud also includes pixels within 600 m of a probability cloud whose blue
reflectance rose by more than 0.06 since the prior (thin cloud fringes), and shadow requires target NIR below 70 % of
the prior's NIR (so naturally dark terrain is not shadow), which lets the absolute dark-NIR threshold relax to 0.2.

Mask values: 0 clear, 1 cloud, 2 cloud shadow. The prior must be clear under the same recipe (checked below)."""
import json, math, os, urllib.request
import numpy as np, rasterio
from rasterio.windows import Window
from rasterio.enums import Resampling
from scipy import ndimage
from pyproj import Transformer
from PIL import Image

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../public/data/sentinel-cloud-repair')
os.makedirs(OUT, exist_ok=True)
N = 1536
CLOUD_PROBABILITY = 50
DARK_NIR = 0.15
DARK_NIR_WITH_PRIOR = 0.2
SHADOW_NIR_RATIO = 0.7
BLUE_BRIGHTENING = 0.06
FRINGE_M = 600
MAX_CLOUD_HEIGHT_M = 8000
ERODE_M, DILATE_M = 40, 100
PAD = 320  # 20 m cells read around each window so off-crop clouds still cast shadows into it

# x0/y0/span are 10 m pixel offsets in the tile; span / N gives the output cell size.
PLACES = [
    {'id': 'pilbara', 'title': 'Hamersley Range, Pilbara', 'x0': 2700, 'y0': 2800, 'span': 2560,
     'target': 'S2B_T50KPA_20260204T021823_L2A', 'prior': 'S2C_T50KPA_20260130T022322_L2A'},
    {'id': 'plateau', 'title': 'Colorado Plateau, Utah', 'x0': 4600, 'y0': 4352, 'span': 2048,
     'target': 'S2A_T12SWF_20260729T180917_L2A', 'prior': 'S2C_T12SWF_20260727T180433_L2A'},
]


def stac_item(item_id):
    url = f'https://earth-search.aws.element84.com/v1/collections/sentinel-2-c1-l2a/items/{item_id}'
    return json.load(urllib.request.urlopen(url))


def read(href, window, shape, resampling, bands=1):
    with rasterio.open(href) as source:
        return source.read(list(range(1, bands + 1)) if bands > 1 else 1, window=window, out_shape=shape,
                           resampling=resampling, boundless=True, fill_value=0)


def disk(radius):
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    return x * x + y * y <= radius * radius


def padded_window(x20, y20, size20):
    return Window(x20 - PAD, y20 - PAD, size20 + 2 * PAD, size20 + 2 * PAD)


def read_reflectance(item, asset, window):
    """10 m band surface reflectance averaged onto the 20 m window."""
    band = item['assets'][asset]['raster:bands'][0]
    dn = read(item['assets'][asset]['href'], Window(window.col_off * 2, window.row_off * 2, window.width * 2,
              window.height * 2), (window.height, window.width), Resampling.average).astype(np.float64)
    return np.where(dn > 0, dn * band['scale'] + band['offset'], np.nan)


def cloud_shadow_mask(item, window, prior=None):
    """Mask on the padded 20 m window: 0 clear, 1 cloud, 2 shadow, 255 no data."""
    shape = (window.height, window.width)
    assets = item['assets']
    probability = read(assets['cloud']['href'], window, shape, Resampling.nearest)
    scl = read(assets['scl']['href'], window, shape, Resampling.nearest)
    nir = read_reflectance(item, 'nir', window)

    cloud = probability >= CLOUD_PROBABILITY
    with np.errstate(invalid='ignore'):
        if prior is None:
            dark = nir < DARK_NIR
        else:
            brighter = read_reflectance(item, 'blue', window) - read_reflectance(prior, 'blue', window)
            near_cloud = ndimage.binary_dilation(cloud, disk(FRINGE_M // 20))
            cloud |= near_cloud & (brighter > BLUE_BRIGHTENING)
            dark = (nir < DARK_NIR_WITH_PRIOR) & (nir < SHADOW_NIR_RATIO * read_reflectance(prior, 'nir', window))
        dark &= (scl != 6) & ~cloud
    sun_azimuth = math.radians(item['properties']['view:sun_azimuth'])
    zenith = math.radians(90 - item['properties']['view:sun_elevation'])
    # Shadows fall opposite the sun; image rows run south, so north is -y.
    dx, dy = -math.sin(sun_azimuth), math.cos(sun_azimuth)
    projected = cloud.copy()
    for step in range(1, int(MAX_CLOUD_HEIGHT_M * math.tan(zenith) / 20) + 1):
        projected |= ndimage.shift(cloud, (dy * step, dx * step), order=0, cval=False)
    shadow = dark & projected

    core = ndimage.binary_erosion(cloud | shadow, disk(ERODE_M // 20))
    final = ndimage.binary_dilation(core, disk(DILATE_M // 20))
    cloud_final = ndimage.binary_dilation(core & cloud, disk(DILATE_M // 20)) & final
    classes = np.where(cloud_final, 1, np.where(final, 2, 0)).astype(np.uint8)
    classes[(scl == 0) | np.isnan(nir)] = 255
    return classes[PAD:-PAD, PAD:-PAD]


def to_output_grid(classes20, span):
    """Nearest-neighbour resample of the 20 m mask onto the N x N output grid."""
    index = ((np.arange(N) + 0.5) * span / N / 2).astype(int)
    return classes20[np.ix_(index, index)]


def write_rgb(item, place, name):
    window = Window(place['x0'], place['y0'], place['span'], place['span'])
    rgb = read(item['assets']['visual']['href'], window, (3, N, N), Resampling.average, bands=3)
    if (rgb == 0).all(axis=0).any():
        raise RuntimeError(f'{name} has no-data pixels')
    Image.fromarray(np.moveaxis(rgb, 0, -1), 'RGB').save(os.path.join(OUT, name), optimize=True)


rasters = {}
summaries = []
primary = None
for place in PLACES:
    target, prior = stac_item(place['target']), stac_item(place['prior'])
    x20, y20, size20 = place['x0'] // 2, place['y0'] // 2, place['span'] // 2
    print(f"{place['title']}: masks")
    window = padded_window(x20, y20, size20)
    prior_mask = to_output_grid(cloud_shadow_mask(prior, window), place['span'])
    target_mask = to_output_grid(cloud_shadow_mask(target, window, prior), place['span'])
    if (target_mask == 255).any():
        raise RuntimeError(f"{place['id']} target window has no-data pixels")
    prior_invalid = float((prior_mask != 0).mean())
    if prior_invalid > 0.001:
        raise RuntimeError(f"{place['id']} prior is not clear: {prior_invalid:.4%} masked")
    Image.fromarray(target_mask, 'L').save(os.path.join(OUT, f"{place['id']}-mask.png"), optimize=True)
    print(f"{place['title']}: imagery")
    write_rgb(target, place, f"{place['id']}-cloudy.png")
    write_rgb(prior, place, f"{place['id']}-prior.png")

    epsg = target['properties']['proj:epsg']
    _, _, origin_x, _, _, origin_y = target['assets']['visual']['proj:transform'][:6]
    west, north = origin_x + place['x0'] * 10, origin_y - place['y0'] * 10
    east, south = west + place['span'] * 10, north - place['span'] * 10
    to_lnglat = Transformer.from_crs(epsg, 4326, always_xy=True)
    corners = [to_lnglat.transform(x, y) for x in (west, east) for y in (south, north)]
    bounds = [min(c[0] for c in corners), min(c[1] for c in corners),
              max(c[0] for c in corners), max(c[1] for c in corners)]
    cell = place['span'] * 10 / N
    common = {'width': N, 'height': N, 'bounds': bounds, 'projection': f'EPSG:{epsg}', 'cellSizeGroundM': cell}
    target_date, prior_date = target['properties']['datetime'][:10], prior['properties']['datetime'][:10]
    target_raster = {**common, 'file': f"{place['id']}-cloudy.png", 'encoding': 'rgba8', 'unit': 'sRGB',
                     'date': target_date, 'description': f"Cloudy Sentinel-2 L2A true-colour observation, {place['title']}"}
    if primary is None:
        primary = target_raster
    else:
        rasters[f"{place['id']}Target"] = target_raster
    rasters[f"{place['id']}Prior"] = {**common, 'file': f"{place['id']}-prior.png", 'encoding': 'rgba8', 'unit': 'sRGB',
                                      'date': prior_date, 'description': 'Earlier clear observation used as the temporal prior'}
    rasters[f"{place['id']}Mask"] = {**common, 'file': f"{place['id']}-mask.png", 'encoding': 'uint8-classes',
                                     'unit': 'cloud mask',
                                     'description': '0 clear, 1 cloud, 2 cloud shadow (CLD_20m >= 50 plus projected dark-NIR shadow, buffered)'}
    summaries.append({'id': place['id'], 'title': place['title'], 'target': place['target'], 'prior': place['prior'],
                      'targetDate': target_date, 'priorDate': prior_date, 'bounds': bounds, 'cellSizeGroundM': cell,
                      'cloudPercent': round(float((target_mask == 1).mean()) * 100, 2),
                      'shadowPercent': round(float((target_mask == 2).mean()) * 100, 2),
                      'priorMaskedPercent': round(prior_invalid * 100, 3)})

all_bounds = [s['bounds'] for s in summaries]
manifest = {
    'id': 'sentinel-cloud-repair',
    'version': 2,
    'kind': 'raster',
    'count': N * N,
    'bbox': [min(b[0] for b in all_bounds), min(b[1] for b in all_bounds),
             max(b[2] for b in all_bounds), max(b[3] for b in all_bounds)],
    'crs': 'mixed UTM',
    'raster': primary,
    'rasters': rasters,
    'properties': {
        'source': 'Copernicus Sentinel-2 Level-2A Collection 1 via Element 84 Earth Search',
        'attribution': 'Contains modified Copernicus Sentinel data (2026)',
        'grid': f'{N} x {N}',
        'mask': {'cloudProbabilityThreshold': CLOUD_PROBABILITY, 'darkNirReflectance': DARK_NIR,
                 'darkNirReflectanceWithPrior': DARK_NIR_WITH_PRIOR, 'shadowNirRatioToPrior': SHADOW_NIR_RATIO,
                 'blueBrighteningVsPrior': BLUE_BRIGHTENING, 'fringeM': FRINGE_M,
                 'maxCloudHeightM': MAX_CLOUD_HEIGHT_M, 'erodeM': ERODE_M, 'dilateM': DILATE_M,
                 'classes': {'0': 'clear', '1': 'cloud', '2': 'cloud shadow'}},
        'places': summaries,
    },
}
with open(os.path.join(OUT, 'manifest.json'), 'w') as file:
    json.dump(manifest, file, indent=2)
print(json.dumps(summaries, indent=2))
