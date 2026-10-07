"""Builds public/data/chicago-parks from an Overpass download (see README.md).

Usage: python -I build.py [path/to/osm.json]
"""
import json, os, sys
import numpy as np
import shapely
from shapely import STRtree
from shapely.geometry import MultiPolygon, Polygon, box, mapping, shape
from shapely.ops import polygonize, transform, unary_union
from pyproj import Transformer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'chicago-tracts'))
from common import out_dir, polygons_to_binary, write_col, write_manifest  # noqa: E402

ID = 'chicago-parks'
RAW = sys.argv[1] if len(sys.argv) > 1 else (
    '/private/tmp/claude-501/-Users-robertchristie-Documents-GitHub-deck-gl-master/'
    '14421614-4041-4f9e-a118-5d838928ca9c/scratchpad/fidelity/raw/chicago-parks/osm.json')
BBOX = (-87.94, 41.64, -87.52, 42.02)
KINDS = ['park', 'nature reserve', 'forest preserve / protected area', 'woodland']
SIMPLIFY_M = 5.0
MIN_AREA_M2 = {'park': 5000, 'nature reserve': 5000, 'forest preserve / protected area': 5000,
               'woodland': 20000}
UMBRELLA = {'Forest Preserve District of Cook County'}  # administrative boundary, not a place

to_m = Transformer.from_crs(4326, 26916, always_xy=True).transform
to_deg = Transformer.from_crs(26916, 4326, always_xy=True).transform


def classify(tags):
    if tags.get('boundary') == 'protected_area':
        return 'forest preserve / protected area'
    if tags.get('leisure') == 'nature_reserve':
        return 'nature reserve'
    if tags.get('leisure') == 'park':
        return 'park'
    return 'woodland'


def way_line(geometry):
    return [(p['lon'], p['lat']) for p in geometry]


def element_polygon(element):
    """Polygon (lng/lat) of an Overpass way or multipolygon relation, or None."""
    if element['type'] == 'way':
        pts = way_line(element['geometry'])
        if len(pts) < 4 or pts[0] != pts[-1]:
            return None
        return Polygon(pts)
    outer, inner = [], []
    for member in element.get('members', []):
        if member['type'] != 'way' or 'geometry' not in member:
            continue
        pts = way_line(member['geometry'])
        if len(pts) < 2:
            continue
        (inner if member.get('role') == 'inner' else outer).append(shapely.LineString(pts))
    if not outer:
        return None
    shells = list(polygonize(unary_union(outer)))
    if not shells:
        return None
    result = unary_union(shells)
    if inner:
        result = result.difference(unary_union(list(polygonize(unary_union(inner)))))
    return result


def polygonal(geometry):
    """Only the polygon parts of a geometry, as a (Multi)Polygon or None."""
    if geometry.is_empty:
        return None
    if geometry.geom_type in ('Polygon', 'MultiPolygon'):
        return geometry
    parts = [g for g in getattr(geometry, 'geoms', []) if g.geom_type in ('Polygon', 'MultiPolygon')]
    return unary_union(parts) if parts else None


def clean(geometry_m, min_area):
    """Drops slivers below min_area and tiny holes; returns (Multi)Polygon in meters or None."""
    geometry_m = polygonal(shapely.make_valid(geometry_m))
    if geometry_m is None:
        return None
    parts = []
    for part in getattr(geometry_m, 'geoms', [geometry_m]):
        if part.area < min_area:
            continue
        holes = [r for r in part.interiors if Polygon(r).area >= 1000]
        parts.append(Polygon(part.exterior, holes))
    if not parts:
        return None
    return parts[0] if len(parts) == 1 else MultiPolygon(parts)


def to_output(geometry_m):
    """Simplify in meters, back to lng/lat on a 1e-5 degree grid; (Multi)Polygon or None."""
    simple = geometry_m.simplify(SIMPLIFY_M, preserve_topology=True)
    deg = shapely.set_precision(transform(to_deg, simple), 1e-5)
    return polygonal(shapely.make_valid(deg))


elements = json.load(open(RAW))['elements']
clip = transform(to_m, box(*BBOX))
records = []
for element in elements:
    tags = element.get('tags', {})
    if tags.get('name') in UMBRELLA:
        continue
    polygon = element_polygon(element)
    if polygon is None:
        continue
    kind = classify(tags)
    geometry_m = clean(transform(to_m, shapely.make_valid(polygon)).intersection(clip), MIN_AREA_M2[kind])
    if geometry_m is None:
        continue
    records.append({'kind': kind, 'name': tags.get('name', ''), 'osmId': element['id'],
                    'osmType': 0 if element['type'] == 'way' else 1, 'geometry': geometry_m})
print('elements', len(elements), 'usable', len(records))

# Dissolve overlapping or touching pieces of the same kind that carry the same name (a park mapped
# as several ways). Unnamed pieces stay separate. The largest piece names the osmId.
merged = []
for kind in KINDS:
    named = {}
    for record in records:
        if record['kind'] != kind:
            continue
        if record['name']:
            named.setdefault(record['name'], []).append(record)
        else:
            merged.append(record)
    for name, group in named.items():
        if len(group) == 1:
            merged.append(group[0])
            continue
        union = unary_union([r['geometry'].buffer(0.5) for r in group]).buffer(-0.5)
        for part in getattr(union, 'geoms', [union]):
            members = [r for r in group if r['geometry'].intersects(part)]
            lead = max(members, key=lambda r: r['geometry'].area)
            merged.append({**lead, 'geometry': part})
records = merged

# Woodland is the least specific kind: remove what a park, reserve or preserve already covers.
covered = unary_union([r['geometry'] for r in records if r['kind'] != 'woodland'])
final = []
for record in records:
    geometry_m = record['geometry']
    if record['kind'] == 'woodland':
        geometry_m = clean(geometry_m.difference(covered), MIN_AREA_M2['woodland'])
        if geometry_m is None:
            continue
    else:
        geometry_m = clean(geometry_m, MIN_AREA_M2[record['kind']])
        if geometry_m is None:
            continue
    out = to_output(geometry_m)
    if out is None or out.is_empty:
        continue
    final.append({**record, 'geometry': out, 'areaKm2': round(geometry_m.area / 1e6, 4)})

final.sort(key=lambda r: (KINDS.index(r['kind']), -r['areaKm2']))
geometries = [r['geometry'] for r in final]
names = ['']
for record in final:
    if record['name'] not in names:
        names.append(record['name'])
count = len(final)
print('features', count, {k: sum(r['kind'] == k for r in final) for k in KINDS})

d = out_dir(ID)
for stale in os.listdir(d):
    os.remove(os.path.join(d, stale))
bounds = shapely.total_bounds(geometries)
manifest = {'id': ID, 'version': 1, 'kind': 'polygons', 'count': count,
            'bbox': [round(float(x), 5) for x in bounds], 'crs': 'EPSG:4326', 'columns': {}}
vertices, ring_offsets, polygon_ring_offsets, part_feature = polygons_to_binary(geometries)
write_col(d, manifest, 'vertices', vertices, 'float32', 'vertices.bin', 2)
write_col(d, manifest, 'ringOffsets', ring_offsets, 'uint32', 'ringOffsets.bin')
write_col(d, manifest, 'polygonRingOffsets', polygon_ring_offsets, 'uint32', 'polygonRingOffsets.bin')
write_col(d, manifest, 'partFeature', part_feature, 'uint32', 'partFeature.bin')
write_col(d, manifest, 'name', [names.index(r['name']) for r in final], 'uint16', 'name.bin')
write_col(d, manifest, 'kind', [KINDS.index(r['kind']) for r in final], 'uint8', 'kind.bin',
          categories=KINDS)
write_col(d, manifest, 'areaKm2', [r['areaKm2'] for r in final], 'float32', 'areaKm2.bin')
write_col(d, manifest, 'osmId', [r['osmId'] for r in final], 'uint32', 'osmId.bin')
write_col(d, manifest, 'osmType', [r['osmType'] for r in final], 'uint8', 'osmType.bin',
          categories=['way', 'relation'])

features = [{'type': 'Feature',
             'properties': {'name': r['name'], 'kind': r['kind'], 'areaKm2': r['areaKm2'],
                            'osmId': r['osmId']},
             'geometry': mapping(r['geometry'])} for r in final]
with open(f'{d}/green-space.geojson', 'w') as f:
    json.dump({'type': 'FeatureCollection', 'features': features}, f, separators=(',', ':'))

# One dissolved multipolygon of all green space, for masks.
mask_m = unary_union([transform(to_m, g) for g in geometries]).buffer(1).buffer(-1)
mask = to_output(clean(mask_m.simplify(8.0, preserve_topology=True), 20000))  # coarser: masks need no detail
with open(f'{d}/green-mask.geojson', 'w') as f:
    json.dump({'type': 'FeatureCollection', 'features': [
        {'type': 'Feature', 'properties': {'name': 'All green space',
                                           'areaKm2': round(mask_m.area / 1e6, 3)},
         'geometry': mapping(mask)}]}, f, separators=(',', ':'))

manifest['geometry'] = {'type': 'polygons', 'file': 'green-space.geojson'}
manifest['names'] = names
manifest['properties'] = {
    'description': ('Parks, nature reserves, forest preserves / protected areas and woodland '
                    'in and around Chicago from OpenStreetMap. Feature i is the same in the binary '
                    'columns and in green-space.geojson. Binary polygons are GeoArrow-style: '
                    'polygon p (a part) owns rings polygonRingOffsets[p]..[p+1], ring r owns '
                    'vertices ringOffsets[r]..[r+1]; partFeature maps part -> feature index. '
                    'green-mask.geojson is one dissolved multipolygon of all green space.'),
    'files': {'green-space.geojson': 'one feature per place (name, kind, areaKm2, osmId)',
              'green-mask.geojson': 'single dissolved multipolygon of all green space'},
    'columns': {'name': 'index into manifest.names (0 = unnamed)',
                'kind': 'category index into columns.kind.categories',
                'areaKm2': 'planar area in EPSG:26916, km2',
                'osmId': 'OpenStreetMap element id (largest piece when pieces were merged)',
                'osmType': '0 = way, 1 = relation (ids repeat across types)'},
    'source': 'OpenStreetMap contributors via the Overpass API, downloaded 2026-10-07',
    'license': 'ODbL 1.0',
    'attribution': '© OpenStreetMap contributors (ODbL)',
    'processing': ('Kind priority: protected_area > nature_reserve > park > woodland. Same-name, '
                   'same-kind pieces merged; woodland has parks/reserves/preserves subtracted. '
                   'Clipped to the bbox, simplified 5 m (topology kept; 8 m for the mask), slivers < 0.5 ha '
                   '(woodland < 2 ha) dropped, 1e-5 degree grid.')}
mask_area = round(mask_m.area / 1e6, 1)
manifest['properties']['storyNotes'] = [
    f'{count} places cover {mask_area} km2 of green space inside the bbox.',
    'Forest preserves ring the city: the Cook County preserves are mapped as protected areas, '
    'and Chicago parks as parks; woodland is what neither covers.']
write_manifest(d, manifest)
