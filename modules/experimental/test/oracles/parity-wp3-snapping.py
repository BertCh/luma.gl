# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Shapely STRtree nearest-edge reference on f32 planar endpoint chords.

Chicago: full 77,140 directed roads, all 29,557 original nodes, 2,048 reproducible
spatially ordered/systematically sampled places from the full 105,808-row input.
WGS84 -> local AEQD (41.885,-87.635), then float32; no geographic-mode claim.
Preserve CSR edge-row -> original source-edge mapping. Travel-time weights are
unmodified f32 seconds. Reference geometry is Shapely query_nearest,
line_locate_point and line_interpolate_point, on the actual rounded coordinates.
For real data retain exact nearest tie sets and every candidate within twice the
G-profile coordinate bound of the minimum. Reciprocal edges can round differently
on GPU: accept only this explicitly enumerated uncertainty interval and compare
ALL numeric outputs to the independent projection on the selected edge. No global
ID tolerance. Synthetic axis-aligned ties require the documented smallest CSR ID.

G bound: 16 ulps at maximum coordinate magnitude covers subtraction, projection
and interpolation; fraction bound = coordinate bound / edge length plus 8 f32
epsilons; weighted cost bound = fraction bound * edge weight + 8 weight ulps.
No relative tolerance is added. Synthetic exact threshold includes distance=2 and
just-outside=2.25. Real queries within one coordinate bound of radius=50m are
excluded before selection and counted. Candidate capacity bounds ALL expanded
query-square/edge-envelope intersections with STRtree (plus margin), not only
nearest winners; GPU must report zero overflow. Synthetic scans and BVH run with
both default geometric costs and explicit costs, all three start directions.
"""
import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import numpy
import pyproj
import shapely
from shapely import LineString, Point, STRtree

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[3]/'examples/deck/spatial-analysis-showcase/public/data'
VERSIONS = {name: importlib.metadata.version(name) for name in ['numpy','pyproj','shapely']}
VERSIONS['python'] = platform.python_version()


def write(output, name, value):
    value.update(versions=VERSIONS, conventions=__doc__, generator=Path(__file__).name,
                 reference='shapely.STRtree.query_nearest; line_locate_point; line_interpolate_point')
    path = output/f'parity-wp3-snapping-{name}.json'
    path.write_text(json.dumps(value, separators=(',', ':'), allow_nan=False)+'\n')
    assert path.stat().st_size <= 1_500_000, (name, path.stat().st_size)
    print(path.name, path.stat().st_size, flush=True)


def generate(output, name, positions, sources, targets, weights, points, pointIds, source):
    order = numpy.argsort(sources, kind='stable')
    sources, targets, weights = sources[order], targets[order], weights[order]
    offsets = numpy.concatenate(([0], numpy.cumsum(numpy.bincount(sources, minlength=len(positions)))))
    lines = [LineString(positions[[first,second]]) for first,second in zip(sources,targets)]
    tree = STRtree(lines)
    scale = float(max(numpy.max(numpy.abs(positions)), numpy.max(numpy.abs(points)), 1))
    bound = 16 * float(numpy.spacing(numpy.float32(scale)))
    radius = 50 if name == 'chicago' else 2
    queries, selectedIds, excluded = [], [], 0
    capacity = 0
    for pointCoordinates, pointId in zip(points, pointIds):
        point = Point(pointCoordinates)
        nearest, distances = tree.query_nearest(point, return_distance=True, all_matches=True)
        distance = float(min(distances))
        if name == 'chicago' and abs(distance-radius) <= bound:
            excluded += 1
            continue
        nearest = sorted(int(edge) for edge in nearest)
        candidates = []
        if distance <= radius:
            allowed = (tree.query(point, predicate='dwithin', distance=distance+2*bound)
                       if name == 'chicago' else [min(nearest)])
            for edge in sorted(int(value) for value in allowed):
                line = lines[edge]
                offset = float(shapely.line_locate_point(line, point))
                fraction = offset/line.length if line.length else 0
                location = list(shapely.line_interpolate_point(line, offset).coords[0])
                candidates.append(dict(edge=edge, source=int(sources[edge]), target=int(targets[edge]),
                    fraction=fraction, distance=float(line.distance(point)), position=location,
                    length=float(line.length), weight=float(weights[edge]),
                    fractionBound=(bound/line.length+8*float(numpy.finfo(numpy.float32).eps)) if line.length else 0))
        capacity += len(tree.query(shapely.box(point.x-radius-bound,point.y-radius-bound,
                                               point.x+radius+bound,point.y+radius+bound)))
        queries.append(dict(point=list(pointCoordinates), nearestEdges=nearest, nearestDistance=distance, candidates=candidates))
        selectedIds.append(int(pointId))
        if name == 'chicago' and len(queries) == 2048:
            break
    write(output, name+'-topology', dict(nodeCount=len(positions), offsets=offsets.tolist(),
          neighbors=targets.tolist(), originalEdgeIds=order.tolist(), source=source))
    write(output, name+'-weights', dict(weights=weights.tolist(), source=source))
    # Geometry is split from topology so every JSON remains below 1.5 MB.
    write(output, name+'-positions', dict(positions=positions.reshape(-1).tolist(), source=source))
    write(output, name+'-queries', dict(case=name, radius=radius, coordinateBound=bound,
          candidateCapacity=max(1,capacity+len(queries)*4), boundingBoxCandidateCount=capacity,
          queries=queries, originalPointIds=selectedIds, excludedRadiusBoundary=excluded,
          exactTieBreak=name != 'chicago', source=source))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, default=HERE/'fixtures')
    output = parser.parse_args().output_dir
    output.mkdir(parents=True, exist_ok=True)
    generate(output, 'synthetic', numpy.array([[0,0],[10,0],[10,10],[0,10],[20,20]], dtype=float),
             numpy.array([0,1,0,1,2,3,4]), numpy.array([1,0,1,2,3,0,4]),
             numpy.array([4,8,12,3,0,7,9], dtype=float),
             numpy.array([[5,0],[0,0],[5,2],[5,2.25],[10,5],[20,20],[20,21],[30,30],[5,10],[10,10]], dtype=float),
             list(range(10)), dict(description='exact reciprocal/duplicate/node ties, degenerate edge, zero weight and unsnapped points'))
    roads, places = DATA/'chicago-roads', DATA/'chicago-places'
    transformer = pyproj.Transformer.from_crs('EPSG:4326',
        '+proj=aeqd +lat_0=41.885 +lon_0=-87.635 +datum=WGS84 +units=m', always_xy=True)
    def project(path):
        longitudeLatitude = numpy.fromfile(path,dtype='<f4').reshape(-1,2)
        return numpy.column_stack(transformer.transform(longitudeLatitude[:,0],longitudeLatitude[:,1])).astype(numpy.float32).astype(float)
    positions = project(roads/'nodes.bin')
    points = project(places/'position.bin')
    # Spatially order into 500m northing strips and easting, then systematic sample.
    ordering = numpy.lexsort((points[:,0], numpy.floor(points[:,1]/500)))
    primary = ordering[numpy.linspace(0,len(points)-1,2048,dtype=int)]
    primarySet = set(primary.tolist())
    replacements = [index for index in ordering[numpy.linspace(0,len(points)-1,512,dtype=int)] if index not in primarySet]
    selection = numpy.concatenate((primary, replacements))
    source = dict(hashes={str(path.relative_to(DATA)):hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in [roads/'nodes.bin',roads/'edgeSource.bin',roads/'edgeTarget.bin',roads/'edgeTravelTime.bin',places/'position.bin']},
                  attribution={name:json.loads((DATA/name/'manifest.json').read_text())['properties']
                               for name in ['chicago-roads','chicago-places']},
                  originalPlaceCount=len(points), selection='2048 systematic spatial-order proposals across full extent; boundary replacements from complementary systematic sample')
    generate(output, 'chicago', positions, numpy.fromfile(roads/'edgeSource.bin',dtype='<u4'),
             numpy.fromfile(roads/'edgeTarget.bin',dtype='<u4'), numpy.fromfile(roads/'edgeTravelTime.bin',dtype='<f4'),
             points[selection], selection, source)


if __name__ == '__main__':
    main()
