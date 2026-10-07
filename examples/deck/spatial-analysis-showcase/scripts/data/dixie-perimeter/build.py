"""Builds public/data/dixie-perimeter from the DIXIE feature of public/data/poopdeck-wildfires.

No download: the final 2021 Dixie Fire perimeter (NIFC) is already shipped in the wildfires dataset.
Usage: python -I build.py
"""
import datetime, json, os, sys
import numpy as np
import shapely
from shapely.geometry import MultiPolygon, Polygon, mapping
from shapely.ops import transform, unary_union
from pyproj import Transformer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'chicago-tracts'))
from common import APP, out_dir, polygons_to_binary, write_col, write_manifest  # noqa: E402

ID = 'dixie-perimeter'
SOURCE = f'{APP}/public/data/poopdeck-wildfires'
to_m = Transformer.from_crs(4326, 32610, always_xy=True).transform  # UTM 10N, as dixie-fire
to_deg = Transformer.from_crs(32610, 4326, always_xy=True).transform


def read(manifest, name):
    spec = manifest['columns'][name]
    return np.fromfile(f"{SOURCE}/{spec['file']}", dtype='<' + {'float32': 'f4', 'uint32': 'u4', 'uint16': 'u2', 'uint8': 'u1'}[spec['dtype']])


manifest_in = json.load(open(f'{SOURCE}/manifest.json'))
names = manifest_in['columns']['name']['categories']
feature = list(read(manifest_in, 'name')).index(names.index('DIXIE'))
acres = float(read(manifest_in, 'acres')[feature])
year = int(read(manifest_in, 'year')[feature])
perimeter_time = int(read(manifest_in, 'perimeterTime')[feature])
assert year == 2021 and acres > 900000
vertices = read(manifest_in, 'vertices').reshape(-1, 2).astype('f8')
ring_offsets = read(manifest_in, 'ringOffsets')
polygon_ring_offsets = read(manifest_in, 'polygonRingOffsets')
first, last = read(manifest_in, 'featurePolygonOffsets')[feature:feature + 2]

# Parts are shells (counter-clockwise) followed by their holes (clockwise); rebuild with shapely.
shells, holes = [], []
for polygon in range(first, last):
    for ring in range(polygon_ring_offsets[polygon], polygon_ring_offsets[polygon + 1]):
        coords = vertices[ring_offsets[ring]:ring_offsets[ring + 1]]
        (shells if Polygon(coords).exterior.is_ccw else holes).append(coords)
parts = unary_union([shapely.make_valid(Polygon(c)) for c in shells])
if holes:
    parts = parts.difference(unary_union([shapely.make_valid(Polygon(c)) for c in holes]))
perimeter_m = transform(to_m, parts)
print('parts', len(shells), 'holes', len(holes), 'vertices', len(vertices[ring_offsets[polygon_ring_offsets[first]]:ring_offsets[polygon_ring_offsets[last]]]))

# Drop slivers under 1 ha, simplify 20 m (topology kept), area in the metric projection.
pieces = [p for p in getattr(perimeter_m, 'geoms', [perimeter_m]) if p.area >= 10000]
perimeter_m = MultiPolygon(pieces)
simple_m = perimeter_m.simplify(20.0, preserve_topology=True)
perimeter = shapely.make_valid(shapely.set_precision(transform(to_deg, simple_m), 1e-5))
area_km2 = round(simple_m.area / 1e6, 1)
print('simplified parts', len(perimeter.geoms), 'area km2', area_km2, 'acres', round(simple_m.area / 4046.8564))

d = out_dir(ID)
for stale in os.listdir(d):
    os.remove(os.path.join(d, stale))
perimeter_date = (datetime.datetime(2020, 1, 1) + datetime.timedelta(seconds=perimeter_time)).strftime('%Y-%m-%d')
properties = {'name': 'Dixie Fire', 'year': 2021, 'acres': acres, 'areaKm2': area_km2,
              'perimeterDate': perimeter_date}
with open(f'{d}/dixie-perimeter.geojson', 'w') as f:
    json.dump({'type': 'FeatureCollection',
               'features': [{'type': 'Feature', 'properties': properties, 'geometry': mapping(perimeter)}]},
              f, separators=(',', ':'))
manifest = {'id': ID, 'version': 1, 'kind': 'polygons', 'count': 1,
            'bbox': [round(float(x), 5) for x in shapely.total_bounds([perimeter])], 'crs': 'EPSG:4326',
            'columns': {}}
v, ring, poly, part = polygons_to_binary([perimeter])
write_col(d, manifest, 'vertices', v, 'float32', 'vertices.bin', 2)
write_col(d, manifest, 'ringOffsets', ring, 'uint32', 'ringOffsets.bin')
write_col(d, manifest, 'polygonRingOffsets', poly, 'uint32', 'polygonRingOffsets.bin')
write_col(d, manifest, 'partFeature', part, 'uint32', 'partFeature.bin')
write_col(d, manifest, 'acres', [acres], 'float32', 'acres.bin')
manifest['geometry'] = {'type': 'polygons', 'file': 'dixie-perimeter.geojson'}
manifest['names'] = ['Dixie Fire']
manifest['properties'] = {
    'description': ('Final 2021 Dixie Fire perimeter (California, 963,405 acres) as one multi-part '
                    'feature, extracted from the NIFC perimeters in the poopdeck-wildfires dataset '
                    'and simplified 20 m. Binary polygons are GeoArrow-style: polygon p owns rings '
                    'polygonRingOffsets[p]..[p+1], ring r owns vertices ringOffsets[r]..[r+1].'),
    'acres': 'NIFC GIS acres of the full perimeter',
    'perimeterTime': perimeter_time,
    'source': 'public/data/poopdeck-wildfires (NIFC Open Data, packaged by poopdeck.gl)',
    'license': 'Public domain (US Government work, NIFC)',
    'attribution': 'Fire perimeters: National Interagency Fire Center (NIFC) Open Data, public domain; packaged by poopdeck.gl',
    'notes': ['perimeterTime is the NIFC record date (seconds since 2020-01-01Z), usually the last mapping, not the ignition date.']}
write_manifest(d, manifest)
