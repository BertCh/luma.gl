"""Builds public/data/chicago-boundary: city limit, Lake Michigan and the land mask (see README.md).

Usage: python -I build.py [path/to/ne_10m_lakes.geojson]
Reads public/data/chicago-community-areas/community-areas.geojson from this repo (build that first).
"""
import json, os, sys
import shapely
from shapely.geometry import MultiPolygon, Polygon, box, mapping, shape
from shapely.ops import transform, unary_union
from pyproj import Transformer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'chicago-tracts'))
from common import APP, out_dir, polygons_to_binary, write_col, write_manifest  # noqa: E402

ID = 'chicago-boundary'
LAKES = sys.argv[1] if len(sys.argv) > 1 else (
    '/private/tmp/claude-501/-Users-robertchristie-Documents-GitHub-deck-gl-master/'
    '14421614-4041-4f9e-a118-5d838928ca9c/scratchpad/fidelity/raw/natural-earth/ne_10m_lakes.geojson')
REGION = (-88.0, 41.55, -87.2, 42.15)
to_m = Transformer.from_crs(4326, 26916, always_xy=True).transform
to_deg = Transformer.from_crs(26916, 4326, always_xy=True).transform


def snap(geometry):
    return shapely.make_valid(shapely.set_precision(geometry, 1e-5))


def drop_small_holes(geometry, min_area):
    parts = [Polygon(p.exterior, [r for r in p.interiors if Polygon(r).area >= min_area])
             for p in getattr(geometry, 'geoms', [geometry])]
    return parts[0] if len(parts) == 1 else MultiPolygon(parts)


# (a) City limit: dissolve the 77 community areas, close sub-2 m gaps, simplify 2 m.
areas = json.load(open(f'{APP}/public/data/chicago-community-areas/community-areas.geojson'))['features']
assert len(areas) == 77
union = unary_union([transform(to_m, shape(f['geometry'])) for f in areas])
closed = union.buffer(1.5, join_style='mitre').buffer(-1.5, join_style='mitre')
city_m = drop_small_holes(closed.simplify(2.0, preserve_topology=True), 1000)
city = snap(transform(to_deg, city_m))
city_area = round(city_m.area / 1e6, 3)
print('city parts', len(getattr(city, 'geoms', [city])), 'area km2', city_area)

# (b) Lake Michigan within the region (Natural Earth 1:10m, generalised ~100 m).
region = box(*REGION)
lake_feature = next(f for f in json.load(open(LAKES))['features'] if f['properties']['name'] == 'Lake Michigan')
lake = snap(shapely.make_valid(shape(lake_feature['geometry'])).intersection(region))
lake = lake if lake.geom_type == 'MultiPolygon' else MultiPolygon([lake])
lake_area = round(transform(to_m, lake).area / 1e6, 1)

# (c) Land mask: the region box minus the lake.
land_mask = snap(region.difference(lake))
land_mask = land_mask if land_mask.geom_type == 'MultiPolygon' else MultiPolygon([land_mask])
print('lake parts', len(lake.geoms), 'km2', lake_area, '| land mask parts', len(land_mask.geoms))

d = out_dir(ID)
for stale in os.listdir(d):
    os.remove(os.path.join(d, stale))


def collection(geometry, properties):
    return {'type': 'FeatureCollection',
            'features': [{'type': 'Feature', 'properties': properties, 'geometry': mapping(geometry)}]}


files = {
    'city.geojson': collection(city, {'name': 'City of Chicago', 'areaKm2': city_area}),
    'lake.geojson': collection(lake, {'name': 'Lake Michigan (region clip)', 'areaKm2': lake_area}),
    'land-mask.geojson': collection(land_mask, {'name': 'Land within region'}),
}
for name, data in files.items():
    with open(f'{d}/{name}', 'w') as f:
        json.dump(data, f, separators=(',', ':'))

manifest = {'id': ID, 'version': 1, 'kind': 'polygons', 'count': 1,
            'bbox': list(REGION), 'crs': 'EPSG:4326',
            'columns': {}}
for prefix, geometry in (('', city), ('lake', lake), ('landMask', land_mask)):
    v, ring, poly, part = polygons_to_binary([geometry])
    name = (lambda base: base if not prefix else prefix + base[0].upper() + base[1:])
    write_col(d, manifest, name('vertices'), v, 'float32', name('vertices') + '.bin', 2)
    write_col(d, manifest, name('ringOffsets'), ring, 'uint32', name('ringOffsets') + '.bin')
    write_col(d, manifest, name('polygonRingOffsets'), poly, 'uint32', name('polygonRingOffsets') + '.bin')
    write_col(d, manifest, name('partFeature'), part, 'uint32', name('partFeature') + '.bin')
write_col(d, manifest, 'areaKm2', [city_area], 'float32', 'areaKm2.bin')
manifest['geometry'] = {'type': 'polygons', 'file': 'city.geojson'}
manifest['names'] = ['City of Chicago']
manifest['properties'] = {
    'description': ('Derived Chicago geography: the city limit (dissolved community areas), Lake '
                    'Michigan clipped to the Chicago region and the land mask (region box minus the '
                    'lake). Each geometry is one multi-part feature with GeoArrow-style binary '
                    'columns: unprefixed = city limit, lake* = lake, landMask* = land mask. Polygon '
                    'p owns rings polygonRingOffsets[p]..[p+1], ring r owns vertices '
                    'ringOffsets[r]..[r+1]; partFeature maps part -> feature (always 0).'),
    'files': {'city.geojson': 'city limit (single feature)',
              'lake.geojson': 'Lake Michigan within the region (single feature)',
              'land-mask.geojson': 'region box minus the lake: the land side for fade-the-lake masks'},
    'region': list(REGION),
    'notes': ['The Natural Earth lake shoreline is generalised (~100 m): it does not follow the real '
              'Chicago shoreline, harbours or piers, and may overlap or leave gaps against the city '
              'limit.',
              'The city limit along the lake is the community-area edge, which is the shoreline '
              'in the City of Chicago source.'],
    'source': ('City of Chicago community areas (dissolved) and Natural Earth 1:10m lakes, built '
               '2026-10-07'),
    'license': 'City of Chicago Data Portal Terms of Use (community areas); public domain (Natural Earth)',
    'attribution': 'City of Chicago; Made with Natural Earth'}
write_manifest(d, manifest)
