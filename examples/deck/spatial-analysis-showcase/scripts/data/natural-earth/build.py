"""Builds public/data/natural-earth from Natural Earth GeoJSON files (see README.md).

Usage: python -I build.py [dir with the downloaded ne_*.geojson files]
"""
import json, os, sys
import numpy as np
import shapely
from shapely.geometry import box, mapping, shape

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'chicago-tracts'))
from common import out_dir, polygons_to_binary, write_col, write_manifest  # noqa: E402

ID = 'natural-earth'
RAW = sys.argv[1] if len(sys.argv) > 1 else (
    '/private/tmp/claude-501/-Users-robertchristie-Documents-GitHub-deck-gl-master/'
    '14421614-4041-4f9e-a118-5d838928ca9c/scratchpad/fidelity/raw/natural-earth')
NL_BBOX = (2.5, 50.6, 7.5, 53.8)


def read(name):
    return json.load(open(f'{RAW}/{name}.geojson'))['features']


def round_coords(value, digits):
    if isinstance(value[0], (int, float)):
        return [round(value[0], digits), round(value[1], digits)]
    return [round_coords(v, digits) for v in value]


def dedupe_ring(ring):
    out = [ring[0]]
    for p in ring[1:]:
        if p != out[-1]:
            out.append(p)
    return out


def clean_geometry(geometry, digits):
    """Rounds coordinates, drops repeated vertices and degenerate rings/lines."""
    kind = geometry['type']
    coords = round_coords(geometry['coordinates'], digits)
    if kind == 'Polygon':
        rings = [dedupe_ring(r) for r in coords]
        rings = [r for r in rings if len(r) >= 4]
        return {'type': 'Polygon', 'coordinates': rings} if rings else None
    if kind == 'MultiPolygon':
        polygons = []
        for polygon in coords:
            rings = [r for r in (dedupe_ring(r) for r in polygon) if len(r) >= 4]
            if rings:
                polygons.append(rings)
        return {'type': 'MultiPolygon', 'coordinates': polygons} if polygons else None
    if kind == 'LineString':
        line = dedupe_ring(coords)
        return {'type': 'LineString', 'coordinates': line} if len(line) >= 2 else None
    lines = [l for l in (dedupe_ring(l) for l in coords) if len(l) >= 2]
    return {'type': 'MultiLineString', 'coordinates': lines} if lines else None


def build_collection(features, keep, digits, clip=None):
    out = []
    for feature in features:
        geometry = shape(feature['geometry'])
        if clip is not None:
            geometry = shapely.make_valid(geometry).intersection(clip)
            if geometry.is_empty:
                continue
            kind = feature['geometry']['type']
            want = 'Polygon' if 'Polygon' in kind else 'LineString'
            parts = [g for g in getattr(geometry, 'geoms', [geometry]) if want in g.geom_type]
            if not parts:
                continue
            geometry = shapely.multipolygons(parts) if want == 'Polygon' else shapely.multilinestrings(parts)
            geom_json = mapping(geometry)
        else:
            geom_json = feature['geometry']
        cleaned = clean_geometry(geom_json, digits)
        if cleaned is None:
            continue
        props = {k: feature['properties'].get(k) for k in keep}
        out.append({'type': 'Feature', 'properties': props, 'geometry': cleaned})
    return {'type': 'FeatureCollection', 'features': out}


d = out_dir(ID)
for stale in os.listdir(d):
    os.remove(os.path.join(d, stale))
clip = box(*NL_BBOX)

# (output name, source name, kept properties, digits, clip, description)
SPECS = [
    ('ne_110m_land', 'ne_110m_land', [], 4, None, '1:110m land polygons'),
    ('ne_110m_coastline', 'ne_110m_coastline', [], 4, None, '1:110m coastline lines'),
    ('ne_110m_admin_0_countries', 'ne_110m_admin_0_countries', ['NAME', 'ISO_A3', 'CONTINENT', 'REGION_UN'], 4, None,
     '1:110m countries (NAME, ISO_A3, CONTINENT, REGION_UN)'),
    ('ne_110m_admin_0_boundary_lines', 'ne_110m_admin_0_boundary_lines_land', [], 4, None,
     '1:110m land boundary lines between countries'),
    ('ne_110m_lakes', 'ne_110m_lakes', ['name'], 4, None, '1:110m lakes'),
    ('ne_10m_land_nl', 'ne_10m_land', [], 5, clip, '1:10m land clipped to the Netherlands region'),
    ('ne_10m_coastline_nl', 'ne_10m_coastline', [], 5, clip, '1:10m coastline clipped to the Netherlands region'),
    ('ne_10m_boundary_lines_nl', 'ne_10m_admin_0_boundary_lines_land', [], 5, clip,
     '1:10m land boundary lines clipped to the Netherlands region'),
    ('ne_10m_lakes_nl', 'ne_10m_lakes', ['name'], 5, clip, '1:10m lakes clipped to the Netherlands region'),
    ('ne_10m_rivers_nl', 'ne_10m_rivers_lake_centerlines', ['name', 'scalerank'], 5, clip,
     '1:10m rivers and lake centerlines clipped to the Netherlands region'),
]
collections = {}
files = {}
for out_name, source, keep, digits, region, description in SPECS:
    features = read(source)
    if out_name == 'ne_110m_admin_0_countries':
        for feature in features:
            props = feature['properties']
            if props.get('ISO_A3') in (None, '-99'):  # France, Norway, N. Cyprus, Somaliland, Kosovo
                props['ISO_A3'] = props.get('ADM0_A3') or props.get('ISO_A3_EH')
    collection = build_collection(features, keep, digits, region)
    collections[out_name] = collection
    with open(f'{d}/{out_name}.geojson', 'w') as f:
        json.dump(collection, f, separators=(',', ':'))
    files[f'{out_name}.geojson'] = f'{description}; {len(collection["features"])} features'
    print(out_name, len(collection['features']))


def polygons(collection):
    return [shape(f['geometry']) for f in collection['features']]


manifest = {'id': ID, 'version': 1, 'kind': 'polygons', 'bbox': [-180, -90, 180, 83.6451], 'crs': 'EPSG:4326',
            'columns': {}}
countries = collections['ne_110m_admin_0_countries']['features']
manifest['count'] = len(countries)

# Countries are the primary geometry (unprefixed columns).
vertices, ring_offsets, polygon_ring_offsets, part_feature = polygons_to_binary(polygons(collections['ne_110m_admin_0_countries']))
write_col(d, manifest, 'vertices', vertices, 'float32', 'vertices.bin', 2)
write_col(d, manifest, 'ringOffsets', ring_offsets, 'uint32', 'ringOffsets.bin')
write_col(d, manifest, 'polygonRingOffsets', polygon_ring_offsets, 'uint32', 'polygonRingOffsets.bin')
write_col(d, manifest, 'partFeature', part_feature, 'uint32', 'partFeature.bin')
names = [f['properties']['NAME'] for f in countries]
iso3_codes = [f['properties']['ISO_A3'] for f in countries]
continents = sorted({f['properties']['CONTINENT'] for f in countries})
regions = sorted({f['properties']['REGION_UN'] for f in countries})
write_col(d, manifest, 'name', np.arange(len(names)), 'uint16', 'name.bin')
write_col(d, manifest, 'iso3', np.arange(len(names)), 'uint16', 'iso3.bin')
write_col(d, manifest, 'continent', [continents.index(f['properties']['CONTINENT']) for f in countries], 'uint8',
          'continent.bin', categories=continents)
write_col(d, manifest, 'regionUn', [regions.index(f['properties']['REGION_UN']) for f in countries], 'uint8',
          'regionUn.bin', categories=regions)
# Land polygons (prefixed `land*` columns).
land_v, land_r, land_p, land_f = polygons_to_binary(polygons(collections['ne_110m_land']))
write_col(d, manifest, 'landVertices', land_v, 'float32', 'landVertices.bin', 2)
write_col(d, manifest, 'landRingOffsets', land_r, 'uint32', 'landRingOffsets.bin')
write_col(d, manifest, 'landPolygonRingOffsets', land_p, 'uint32', 'landPolygonRingOffsets.bin')
write_col(d, manifest, 'landPartFeature', land_f, 'uint32', 'landPartFeature.bin')

manifest['geometry'] = {'type': 'polygons', 'file': 'ne_110m_admin_0_countries.geojson'}
manifest['names'] = names
manifest['properties'] = {
    'description': ('Natural Earth world reference geometry: 1:110m land, coastline, countries, '
                    'boundary lines and lakes for the whole world, and 1:10m land, coastline, '
                    'boundary lines, lakes and rivers clipped to the Netherlands region '
                    '[2.5, 50.6, 7.5, 53.8]. Binary polygon columns: unprefixed = countries '
                    '(feature i of ne_110m_admin_0_countries.geojson), land* = ne_110m_land. '
                    'GeoArrow-style: polygon p owns rings polygonRingOffsets[p]..[p+1], ring r '
                    'owns vertices ringOffsets[r]..[r+1]; partFeature maps part -> feature. '
                    'Natural Earth antimeridian splits are kept as published.'),
    'files': files,
    'columns': {'name': 'country i is manifest.names[i]', 'iso3': 'country i is properties.iso3Codes[i]',
                'continent': 'category index (CONTINENT)', 'regionUn': 'category index (REGION_UN)'},
    'iso3Codes': iso3_codes,
    'nlRegion': list(NL_BBOX),
    'source': 'https://www.naturalearthdata.com (nvkelso/natural-earth-vector GeoJSON), downloaded 2026-10-07',
    'license': 'Public domain (Natural Earth)',
    'attribution': 'Made with Natural Earth',
    'notes': ['ISO_A3 of France, Norway and a few others is -99 in Natural Earth; ADM0_A3 is used instead.']}
write_manifest(d, manifest)
