import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'chicago-tracts'))
from common import *
import geopandas as gpd, numpy as np, shapely

ID = 'chicago-community-areas'
g = gpd.read_file(f'{RAW}/chicago-community-areas/ca.geojson').to_crs(4326)
g['id'] = g['area_numbe'].astype(int)
g = g.sort_values('id').reset_index(drop=True)
assert len(g) == 77 and list(g.id) == list(range(1, 78))
g['geometry'] = shapely.set_precision(g.geometry.simplify_coverage(0.00004), 1e-5)  # ~4 m, topology kept
g['name'] = g['community'].str.title()
g['areaKm2'] = (g.to_crs(26916).area / 1e6).round(3)
verts, ringOff, polyRing, partFeat = polygons_to_binary(g.geometry)
d = out_dir(ID)
m = {'id': ID, 'version': 1, 'kind': 'polygons', 'count': len(g), 'bbox': [round(x, 5) for x in g.total_bounds], 'crs': 'EPSG:4326', 'columns': {}}
write_col(d, m, 'vertices', verts, 'float32', 'vertices.bin', 2)
write_col(d, m, 'ringOffsets', ringOff, 'uint32', 'ringOffsets.bin')
write_col(d, m, 'polygonRingOffsets', polyRing, 'uint32', 'polygonRingOffsets.bin')
write_col(d, m, 'partFeature', partFeat, 'uint32', 'partFeature.bin')
write_col(d, m, 'areaId', g.id, 'uint8', 'areaId.bin')
write_col(d, m, 'areaKm2', g.areaKm2, 'float32', 'areaKm2.bin')
g[['id', 'name', 'areaKm2', 'geometry']].to_file(f'{d}/community-areas.geojson', driver='GeoJSON', COORDINATE_PRECISION=5)
m['geometry'] = {'type': 'polygons', 'file': 'community-areas.geojson'}
m['names'] = list(g.name)
m['properties'] = {
  'description': '77 official Chicago community areas; feature i has areaId i+1. Binary polygons are GeoArrow-style: polygon p (a polygon part) owns rings polygonRingOffsets[p]..[p+1], ring r owns vertices ringOffsets[r]..[r+1]; partFeature maps part -> feature index. Topology-preserving simplification (~4 m), shared edges identical.',
  'storyNotes': [
    'The 77 community areas are fixed since the 1920s Chicago School of sociology; most city statistics (health, housing, demographics) are published by them.',
    f'Largest: {g.loc[g.areaKm2.idxmax(),"name"]} ({g.areaKm2.max():.1f} km2, includes O\'Hare); smallest: {g.loc[g.areaKm2.idxmin(),"name"]} ({g.areaKm2.min():.2f} km2).',
    'Names are the official portal names; area 76 (O\'Hare) is a thin corridor connected to the city only by a strip, a good test of queen vs rook contiguity and island handling.',
    'A classic 77-unit benchmark geography (the PySAL "chicago" example) so results can be cross-checked against GeoDa/PySAL.']}
write_manifest(d, m)
