"""Builds public/data/alps-context from an Overpass download (see README.md and query.overpassql).

Usage: python -I build.py [path/to/osm.json]
"""
import json, os, re, sys
import shapely
from shapely.geometry import LineString, MultiPolygon, Point, Polygon, box, mapping
from shapely.ops import linemerge, polygonize, transform, unary_union
from pyproj import Transformer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'chicago-tracts'))
from common import out_dir, polygons_to_binary, write_col, write_manifest  # noqa: E402

ID = 'alps-context'
RAW = sys.argv[1] if len(sys.argv) > 1 else (
    '/private/tmp/claude-501/-Users-robertchristie-Documents-GitHub-deck-gl-master/'
    '14421614-4041-4f9e-a118-5d838928ca9c/scratchpad/fidelity/raw/alps-context/osm.json')
WEST, SOUTH, EAST, NORTH = 7.58, 45.86, 7.98, 46.12
WINDOW = box(WEST, SOUTH, EAST, NORTH)
SIMPLIFY_M = 5.0
MIN_GLACIER_M2 = 5000.0
MIN_LAKE_M2 = 5000.0
NOT_LAKE = {'river', 'stream', 'canal', 'ditch', 'drain', 'riverbank', 'moat'}

to_m = Transformer.from_crs(4326, 32632, always_xy=True).transform
to_deg = Transformer.from_crs(32632, 4326, always_xy=True).transform


def rounded(geometry):
    return shapely.set_precision(geometry, 1e-5)


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
        if member['type'] != 'way' or not member.get('geometry'):
            continue
        pts = way_line(member['geometry'])
        if len(pts) < 2:
            continue
        (inner if member.get('role') == 'inner' else outer).append(LineString(pts))
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
    if geometry.is_empty:
        return None
    if geometry.geom_type in ('Polygon', 'MultiPolygon'):
        return geometry
    parts = [g for g in getattr(geometry, 'geoms', []) if g.geom_type in ('Polygon', 'MultiPolygon')]
    return unary_union(parts) if parts else None


def process_polygon(element, min_area):
    """Clip to the window, drop parts < min_area, simplify in UTM; returns (geometry, area_m2) or None."""
    polygon = element_polygon(element)
    if polygon is None:
        return None
    clipped = polygonal(shapely.make_valid(polygon).intersection(WINDOW))
    if clipped is None:
        return None
    geometry_m = polygonal(shapely.make_valid(transform(to_m, clipped)))
    if geometry_m is None:
        return None
    parts = []
    for part in getattr(geometry_m, 'geoms', [geometry_m]):
        if part.area < min_area:
            continue
        parts.append(Polygon(part.exterior, [r for r in part.interiors if Polygon(r).area >= 1000]))
    if not parts:
        return None
    merged = parts[0] if len(parts) == 1 else MultiPolygon(parts)
    simple = merged.simplify(SIMPLIFY_M, preserve_topology=True)
    out = polygonal(shapely.make_valid(rounded(transform(to_deg, simple))))
    if out is None or out.is_empty:
        return None
    return out, merged.area


def parse_number(text):
    match = re.search(r'-?\d+(?:[.,]\d+)?', (text or '').replace("'", ''))
    return round(float(match.group().replace(',', '.')), 1) if match else None


def names_of(tags):
    name = tags.get('name') or tags.get('name:de')
    name_de = tags.get('name:de')
    return name, (name_de if name_de and name_de != name else None)


def collection(features):
    return {'type': 'FeatureCollection', 'features': features}


def point_feature(element, properties):
    return {'type': 'Feature', 'properties': properties,
            'geometry': {'type': 'Point', 'coordinates': [round(element['lon'], 5), round(element['lat'], 5)]}}


data = json.load(open(RAW))
timestamp = data['osm3s']['timestamp_osm_base']
elements = data['elements']
glaciers, lakes, peaks, saddles, places, railway = [], [], [], [], [], []
glacier_polygons = []

for element in elements:
    tags = element.get('tags', {})
    kind = element['type']
    if tags.get('natural') == 'glacier' and kind in ('way', 'relation'):
        result = process_polygon(element, MIN_GLACIER_M2)
        if result:
            name, name_de = names_of(tags)
            glacier_polygons.append({'name': name or '', 'nameDe': name_de, 'osmId': element['id'],
                                     'osmType': 0 if kind == 'way' else 1, 'geometry': result[0],
                                     'areaKm2': round(result[1] / 1e6, 4)})
    elif tags.get('natural') == 'water' and kind in ('way', 'relation') and tags.get('water') not in NOT_LAKE:
        result = process_polygon(element, MIN_LAKE_M2)
        if result:
            lakes.append({'type': 'Feature',
                          'properties': {'name': tags.get('name') or None, 'areaHa': round(result[1] / 1e4, 2),
                                         'osmId': element['id']},
                          'geometry': mapping(result[0])})
    elif kind == 'node' and tags.get('natural') == 'peak':
        elevation = parse_number(tags.get('ele'))
        name, name_de = names_of(tags)
        if name is None and elevation is None:
            continue
        peaks.append(point_feature(element, {'name': name, 'nameDe': name_de, 'elevationM': elevation,
                                             'prominenceM': parse_number(tags.get('prominence')),
                                             'osmId': element['id']}))
    elif kind == 'node' and tags.get('natural') == 'saddle':
        saddles.append(point_feature(element, {'name': tags.get('name'), 'elevationM': parse_number(tags.get('ele')),
                                               'osmId': element['id']}))
    elif kind in ('node', 'way'):
        if kind == 'way':
            if not element.get('geometry') or not (tags.get('railway') in ('station', 'halt') or tags.get('aerialway') == 'station'):
                continue
            centre = shapely.MultiPoint(way_line(element['geometry'])).centroid
            element = {**element, 'lon': centre.x, 'lat': centre.y}
        if tags.get('place') in ('town', 'village', 'hamlet', 'isolated_dwelling'):
            place_kind = 'settlement'
        elif tags.get('railway') in ('station', 'halt'):
            place_kind = 'rail-station'
        elif tags.get('aerialway') == 'station':
            place_kind = 'lift-station'
        elif tags.get('tourism') == 'alpine_hut':
            place_kind = 'hut'
        else:
            continue
        name = tags.get('name')
        if not name:
            continue
        properties = {'name': name, 'kind': place_kind, 'elevationM': parse_number(tags.get('ele')),
                      'osmId': element['id']}
        if place_kind == 'settlement':
            properties['place'] = tags['place']
        places.append(point_feature(element, properties))
    elif kind == 'relation' and tags.get('route') == 'railway' and 'Gornergrat' in tags.get('name', ''):
        lines = [LineString(way_line(m['geometry'])) for m in element['members']
                 if m['type'] == 'way' and len(m.get('geometry') or []) > 1]
        merged = linemerge(unary_union(lines)).intersection(WINDOW)
        for part in getattr(merged, 'geoms', [merged]):
            if part.geom_type != 'LineString' or part.is_empty:
                continue
            simple = transform(to_deg, transform(to_m, part).simplify(1.0))
            railway.append({'type': 'Feature', 'properties': {'name': tags['name'], 'osmId': element['id']},
                            'geometry': mapping(rounded(simple))})

# Stable output order.glacier_polygons.sort(key=lambda r: -r['areaKm2'])
peaks.sort(key=lambda f: -(f['properties']['elevationM'] or 0))
saddles.sort(key=lambda f: -(f['properties']['elevationM'] or 0))
places.sort(key=lambda f: (f['properties']['kind'], f['properties']['name']))
lakes.sort(key=lambda f: -f['properties']['areaHa'])

d = out_dir(ID)
for stale in os.listdir(d):
    os.remove(os.path.join(d, stale))

geometries = [r['geometry'] for r in glacier_polygons]
names = ['']
for record in glacier_polygons:
    if record['name'] not in names:
        names.append(record['name'])
glacier_features = [{'type': 'Feature',
                     'properties': {'name': r['name'] or None, 'nameDe': r['nameDe'], 'areaKm2': r['areaKm2'],
                                    'osmId': r['osmId']},
                     'geometry': mapping(r['geometry'])} for r in glacier_polygons]
files = {'glaciers.geojson': glacier_features, 'peaks.geojson': peaks, 'saddles.geojson': saddles,
         'places.geojson': places, 'lakes.geojson': lakes, 'railway.geojson': railway}
for file, features in files.items():
    with open(f'{d}/{file}', 'w') as f:
        json.dump(collection(features), f, separators=(',', ':'), ensure_ascii=False)

manifest = {'id': ID, 'version': 1, 'kind': 'polygons', 'count': len(glacier_polygons),
            'bbox': [WEST, SOUTH, EAST, NORTH], 'crs': 'EPSG:4326', 'columns': {}}
vertices, ring_offsets, polygon_ring_offsets, part_feature = polygons_to_binary(geometries)
write_col(d, manifest, 'vertices', vertices, 'float32', 'vertices.bin', 2)
write_col(d, manifest, 'ringOffsets', ring_offsets, 'uint32', 'ringOffsets.bin')
write_col(d, manifest, 'polygonRingOffsets', polygon_ring_offsets, 'uint32', 'polygonRingOffsets.bin')
write_col(d, manifest, 'partFeature', part_feature, 'uint32', 'partFeature.bin')
write_col(d, manifest, 'name', [names.index(r['name']) for r in glacier_polygons], 'uint16', 'name.bin')
write_col(d, manifest, 'areaKm2', [r['areaKm2'] for r in glacier_polygons], 'float32', 'areaKm2.bin')
write_col(d, manifest, 'osmId', [r['osmId'] for r in glacier_polygons], 'uint32', 'osmId.bin')
write_col(d, manifest, 'osmType', [r['osmType'] for r in glacier_polygons], 'uint8', 'osmType.bin',
          categories=['way', 'relation'])
manifest['geometry'] = {'type': 'polygons', 'file': 'glaciers.geojson'}
manifest['names'] = names
manifest['properties'] = {
    'description': ('OpenStreetMap context for the Zermatt / Gornergrat window: glacier polygons as GeoArrow-style '
                    'binary columns (feature i is also glaciers.geojson feature i) plus plain GeoJSON side files '
                    'read with dataset.fileUrl(). Polygon p (a part) owns rings polygonRingOffsets[p]..[p+1], ring r '
                    'owns vertices ringOffsets[r]..[r+1]; partFeature maps part -> feature index.'),
    'files': {
        'glaciers.geojson': 'natural=glacier polygons: name, nameDe, areaKm2, osmId',
        'peaks.geojson': 'natural=peak points: name, nameDe, elevationM, prominenceM, osmId',
        'saddles.geojson': 'natural=saddle points: name, elevationM, osmId',
        'places.geojson': 'named settlements, rail stations, lift stations and huts: name, kind, elevationM, osmId (+ place for settlements)',
        'lakes.geojson': 'natural=water polygons >= 0.5 ha: name, areaHa, osmId',
        'railway.geojson': 'Gornergratbahn line(s): name, osmId'},
    'columns': {'name': 'index into manifest.names (0 = unnamed)',
                'areaKm2': 'planar area in EPSG:32632, km2',
                'osmId': 'OpenStreetMap element id',
                'osmType': '0 = way, 1 = relation (ids repeat across types)'},
    'counts': {file: len(features) for file, features in files.items()},
    'source': f'OpenStreetMap contributors via the Overpass API (data timestamp {timestamp})',
    'license': 'ODbL 1.0',
    'attribution': '© OpenStreetMap contributors (ODbL)'}
write_manifest(d, manifest)
print('timestamp', timestamp)
print({file: len(features) for file, features in files.items()})
