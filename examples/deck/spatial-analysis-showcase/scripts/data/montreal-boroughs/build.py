"""Builds public/data/montreal-boroughs: the boroughs and linked cities of the Montreal agglomeration.

Usage: python -I build.py <limites-administratives-agglomeration.geojson> [path/to/bixi-flows/manifest.json]
The input is the WGS 84 GeoJSON of the Ville de Montreal dataset (see README.md); keep the download
outside the app tree. The BIXI manifest (default: this repo's bixi-flows) supplies the borough strings
the match key must equal.
"""
import json, os, sys, unicodedata
import numpy as np
import shapely
from pyproj import Transformer
from shapely.geometry import MultiPolygon, Polygon, mapping, shape
from shapely.ops import transform

ID = 'montreal-boroughs'
HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
SOURCE = sys.argv[1]
BIXI = sys.argv[2] if len(sys.argv) > 2 else f'{APP}/public/data/bixi-flows/manifest.json'
SIMPLIFY_METERS = 25.0
MIN_PART_AREA = 2000.0  # square metres; sliver parts and pinholes below this are dropped
to_m = Transformer.from_crs(4326, 32188, always_xy=True).transform  # NAD83 / MTM zone 8
to_deg = Transformer.from_crs(32188, 4326, always_xy=True).transform


def key_of(text):
    """Spelling-insensitive key: NFC, lower case, any dash or ' - ' becomes '-'."""
    text = unicodedata.normalize('NFC', text).lower()
    for dash in ('—', '–', ' - '):
        text = text.replace(dash, '-')
    return text.strip()


bixi_names = json.load(open(BIXI))['columns']['stationBorough']['categories']
bixi_by_key = {key_of(name): name for name in bixi_names}

source = json.load(open(SOURCE))['features']
features = []
geometries = []
for f in source:
    g = shapely.make_valid(transform(to_m, shape(f['geometry'])))
    parts = [p for p in getattr(g, 'geoms', [g]) if p.geom_type == 'Polygon']
    kept = [Polygon(p.exterior, [r for r in p.interiors if Polygon(r).area >= MIN_PART_AREA])
            for p in parts if p.area >= MIN_PART_AREA]
    geometries.append(MultiPolygon(kept))
    features.append(f['properties'])

# A coverage simplification moves shared borders once, so neighbours never gain gaps or overlaps.
simplified = shapely.coverage_simplify(np.array(geometries, dtype=object), SIMPLIFY_METERS)

out_features = []
unmatched_source = []
matched = set()
for properties, geometry in zip(features, simplified):
    geometry = shapely.set_precision(transform(to_deg, geometry), 1e-5)
    geometry = shapely.make_valid(geometry)
    if geometry.geom_type == 'Polygon':
        geometry = MultiPolygon([geometry])
    name = properties['NOM']
    station_name = bixi_by_key.get(key_of(name))
    if station_name:
        matched.add(station_name)
    else:
        unmatched_source.append(name)
    area = transform(to_m, geometry).area / 1e6
    out_features.append({
        'type': 'Feature',
        'properties': {
            'name': name,
            'kind': 'borough' if properties['TYPE'] == 'Arrondissement' else 'linked city',
            'stationBoroughName': station_name or name,
            'areaKm2': round(area, 2)
        },
        'geometry': mapping(geometry)
    })

out_features.sort(key=lambda feature: feature['properties']['name'])
unmatched_bixi = [name for name in bixi_names if name not in matched]
print('boroughs with BIXI stations:', len(matched), 'of', len(out_features))
print('BIXI borough strings with no polygon (outside the agglomeration):', unmatched_bixi)
assert set(unmatched_bixi) <= {'Boucherville', 'Laval', 'Longueuil', 'Sainte-Julie', 'Terrebonne'}, unmatched_bixi

d = f'{APP}/public/data/{ID}'
os.makedirs(d, exist_ok=True)
for stale in os.listdir(d):
    os.remove(os.path.join(d, stale))
collection = {'type': 'FeatureCollection', 'features': out_features}
with open(f'{d}/boroughs.geojson', 'w') as handle:
    json.dump(collection, handle, separators=(',', ':'), ensure_ascii=False)

# GeoArrow-style binary columns, one entry per polygon part (convention of the other polygon datasets).
vertices, ring_offsets, polygon_ring_offsets, part_feature = [], [0], [0], []
vertex_count = ring_count = 0
for index, feature in enumerate(out_features):
    for part in shape(feature['geometry']).geoms:
        for ring in [part.exterior] + list(part.interiors):
            coordinates = np.asarray(ring.coords)[:, :2]
            vertices.append(coordinates)
            vertex_count += len(coordinates)
            ring_offsets.append(vertex_count)
            ring_count += 1
        polygon_ring_offsets.append(ring_count)
        part_feature.append(index)
columns = {}


def write_column(name, array, dtype, components=1):
    array = np.ascontiguousarray(np.asarray(array).astype('<' + np.dtype(dtype).str[1:]))
    array.tofile(f'{d}/{name}.bin')
    columns[name] = {'file': f'{name}.bin', 'dtype': dtype, 'components': components,
                     'length': int(array.size // components)}


write_column('vertices', np.vstack(vertices), 'float32', 2)
write_column('ringOffsets', ring_offsets, 'uint32')
write_column('polygonRingOffsets', polygon_ring_offsets, 'uint32')
write_column('partFeature', part_feature, 'uint32')
all_bounds = np.array([shape(feature['geometry']).bounds for feature in out_features])
bbox = [round(float(all_bounds[:, 0].min()), 5), round(float(all_bounds[:, 1].min()), 5),
        round(float(all_bounds[:, 2].max()), 5), round(float(all_bounds[:, 3].max()), 5)]
manifest = {
    'id': ID, 'version': 1, 'kind': 'polygons', 'count': len(out_features), 'bbox': bbox,
    'crs': 'EPSG:4326', 'columns': columns,
    'geometry': {'type': 'polygons', 'file': 'boroughs.geojson'},
    'names': [feature['properties']['name'] for feature in out_features],
    'properties': {
        'description': ('Boroughs (arrondissements) and linked cities of the Montreal agglomeration, '
                        'simplified to about 25 m. Feature properties: name (exactly as written in the '
                        'source, NOM), kind, stationBoroughName (the borough string used by the '
                        'bixi-flows dataset; equal to name where no BIXI station has that borough), '
                        'areaKm2. Polygon part p owns rings polygonRingOffsets[p]..[p+1]; partFeature '
                        'maps part to feature.'),
        'source': "Ville de Montreal, Limites administratives de l'agglomeration de Montreal (arrondissements et villes liees), WGS 84 GeoJSON, dataset modified 2023-11-29 (DATEMODIF), downloaded 2026-10-07",
        'license': 'Creative Commons Attribution 4.0 International',
        'attribution': 'Ville de Montréal, données ouvertes (CC BY 4.0)',
        'simplifyMeters': SIMPLIFY_METERS,
        'notes': ['Stations in Laval, Longueuil, Boucherville, Sainte-Julie and Terrebonne lie outside the agglomeration: they have no polygon here.']
    }
}
with open(f'{d}/manifest.json', 'w') as handle:
    json.dump(manifest, handle, indent=1, ensure_ascii=False)
total = 0
for name in sorted(os.listdir(d)):
    size = os.path.getsize(f'{d}/{name}')
    total += size
    print(f'  {name:26s}{size:>9,d}')
print(f'  TOTAL {total:,d} bytes')
