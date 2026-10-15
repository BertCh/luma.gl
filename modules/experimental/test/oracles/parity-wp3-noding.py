# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Shapely unary_union noding with source-preserving substring pieces.

Native planar coordinates. Chicago node coordinates are projected WGS84 -> AEQD
(41.885,-87.635) in meters, rounded to a declared 0.25m input precision grid,
then f32. STRtree selects every whole chord intersecting each fixed window (no
clipping); identical undirected chords keep their first original edge ID. Counts
and SHA256 cover the full source, including deduplicated and degenerate inputs.
Synthetic cases retain duplicate/reversed overlap, T junctions, exact crossings,
polyline interior vertices and an isolated line. All crossings are at tolerance
zero except a separate exactly representable endpoint-snap case at tolerance 1.
GEOS unary_union supplies all noding vertices. Its endpoints split each original
line with shapely.ops.substring, preserving direction, multiplicity and provenance.
Snap cells are an explicit convention adapter: floor((point-minimum)/tolerance),
representative is the first piece endpoint; zero tolerance uses rounded f32
coordinates. Node rank is lexicographic cell order, CSR order is edge/end order.

G profile: 16 ulps at the largest coordinate bounds GPU intersection arithmetic.
Every topology/provenance entry is exact; every piece vertex and node coordinate
is compared. For a piece of k segments, endpoint perturbations <=sqrt(2)*G per
vertex bound length error by 2*k*sqrt(2)*G, plus gamma_(3*k)*length for f32 norm
and summation. No topology ambiguity is accepted using this coordinate bound.
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
from pathlib import Path

import numpy
import pyproj
import shapely
from shapely import LineString, MultiLineString, Point, STRtree
from shapely.ops import substring, unary_union

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[3]/'examples/deck/spatial-analysis-showcase/public/data'
VERSIONS = {name: importlib.metadata.version(name) for name in ['numpy','pyproj','shapely']}
VERSIONS['python'] = platform.python_version()


def generate(output, name, coordinates, tolerance, source):
    coordinates = [numpy.asarray(line,dtype=numpy.float32).astype(float).tolist() for line in coordinates]
    lines = [LineString(line) for line in coordinates]
    noded = unary_union(lines)
    components = list(noded.geoms) if hasattr(noded,'geoms') else [noded]
    junctions = sorted(set(tuple(point) for line in components for point in [line.coords[0],line.coords[-1]]))
    points = [Point(point) for point in junctions]
    tree = STRtree(points)
    pieces, sourceIds = [], []
    for sourceId,line in enumerate(lines):
        # GEOS intersection endpoints can lie a few binary64 ulps off their source.
        # 1e-8m is only a float64 incidence-query allowance, not a GPU tolerance.
        candidates = tree.query(line, predicate='dwithin', distance=1e-8)
        cuts = sorted(set([0.0,line.length]+[float(line.project(points[index])) for index in candidates]))
        for first,second in zip(cuts,cuts[1:]):
            if second-first > 1e-8:
                piece = substring(line,first,second)
                pieces.append(numpy.asarray(piece.coords,dtype=numpy.float32).astype(float).tolist())
                sourceIds.append(sourceId)
    ends = numpy.asarray([point for piece in pieces for point in [piece[0],piece[-1]]])
    origin = ends.min(axis=0)
    cells = numpy.floor((ends-origin)/tolerance) if tolerance else ends
    keys = [tuple(cell) for cell in cells]
    ordered = sorted(set(keys))
    ranks = {cell:index for index,cell in enumerate(ordered)}
    nodePositions = [ends[keys.index(cell)].tolist() for cell in ordered]
    fromNodes = [ranks[keys[2*index]] for index in range(len(pieces))]
    toNodes = [ranks[keys[2*index+1]] for index in range(len(pieces))]
    lengths = [LineString(piece).length for piece in pieces]
    offsets, neighbors, edgeIds, weights = [0], [], [], []
    for node in range(len(ordered)):
        for edge,(first,second) in enumerate(zip(fromNodes,toNodes)):
            for endpoint,other in [(first,second),(second,first)]:
                if endpoint == node:
                    neighbors.append(other)
                    edgeIds.append(edge)
                    weights.append(lengths[edge])
        offsets.append(len(neighbors))
    inputOffsets, pieceOffsets = [0], [0]
    for line in coordinates:
        inputOffsets.append(inputOffsets[-1]+len(line))
    for piece in pieces:
        pieceOffsets.append(pieceOffsets[-1]+len(piece))
    scale = max(1,float(numpy.max(numpy.abs(numpy.concatenate(coordinates)))))
    coordinateBound = 16*float(numpy.spacing(numpy.float32(scale)))
    epsilon = float(numpy.finfo(numpy.float32).eps)/2
    lengthBounds = [2*(len(piece)-1)*math.sqrt(2)*coordinateBound +
                    (3*(len(piece)-1)*epsilon/(1-3*(len(piece)-1)*epsilon))*length
                    for piece,length in zip(pieces,lengths)]
    # Independent GEOS geometric-set equality and source length conservation certificates.
    assert noded.hausdorff_distance(MultiLineString(pieces)) <= math.sqrt(2)*coordinateBound
    sourceLengths = [line.length for line in lines]
    for sourceId,length in enumerate(sourceLengths):
        indices = [index for index,value in enumerate(sourceIds) if value == sourceId]
        assert abs(sum(lengths[index] for index in indices)-length) <= sum(lengthBounds[index] for index in indices)
    # Each input segment pair can generate at most one intersection record (including overlap).
    segments = [LineString([first,second]) for line in coordinates for first,second in zip(line,line[1:])]
    candidatePairs = STRtree(segments).query(segments)
    pairCount = int(numpy.count_nonzero(candidatePairs[0] < candidatePairs[1]))
    fixture = dict(case=name, versions=VERSIONS, reference='shapely.ops.unary_union; shapely.ops.substring; STRtree',
        generator=Path(__file__).name, conventions=__doc__, source=source, tolerance=tolerance,
        positions=[value for line in coordinates for point in line for value in point], lineOffsets=inputOffsets,
        intersectionCapacity=max(16,pairCount+8), coordinateBound=coordinateBound,
        piecePositions=[value for piece in pieces for point in piece for value in point],
        pieceOffsets=pieceOffsets, sourceIds=sourceIds, nodePositions=[value for point in nodePositions for value in point],
        fromNodes=fromNodes, toNodes=toNodes, lengths=lengths, lengthBounds=lengthBounds,
        offsets=offsets, neighbors=neighbors, edgeIds=edgeIds, weights=weights, sourceLengths=sourceLengths,
        geosUnionLength=noded.length, geosHausdorff=noded.hausdorff_distance(MultiLineString(pieces)),
        boundingBoxPairCount=pairCount)
    path = output/f'parity-wp3-noding-{name}.json'
    path.write_text(json.dumps(fixture,separators=(',', ':'),allow_nan=False)+'\n')
    assert path.stat().st_size <= 1_500_000
    print(path.name, path.stat().st_size, 'lines',len(lines),'pieces',len(pieces),'nodes',len(ordered),flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir',type=Path,default=HERE/'fixtures')
    output = parser.parse_args().output_dir
    output.mkdir(parents=True,exist_ok=True)
    generate(output,'cross-overlap',[
        [[-8,0],[8,0]], [[0,-8],[0,8]], [[-4,0],[4,0]], [[8,0],[-8,0]],
        [[4,0],[4,8]], [[-8,-4],[-6,-2],[-4,-4]], [[-6,-4],[-6,0]], [[20,20],[24,23]]
    ],0,dict(description='crossings, reversed duplicate, nested overlap, T junction, polyline and island'))
    generate(output,'snap-cells',[
        [[0,0],[4,0]], [[4.25,0.25],[8,0]], [[8.75,0],[12,0]],
        [[12,0.75],[16,2]], [[16.25,2.25],[20,2]], [[20.125,3],[24,3]]
    ],1,dict(description='same cell merges; close endpoints across cell boundaries stay separate'))
    roads = DATA/'chicago-roads'
    raw = numpy.fromfile(roads/'nodes.bin',dtype='<f4').reshape(-1,2)
    transformer = pyproj.Transformer.from_crs('EPSG:4326',
        '+proj=aeqd +lat_0=41.885 +lon_0=-87.635 +datum=WGS84 +units=m',always_xy=True)
    positions = numpy.round(numpy.column_stack(transformer.transform(raw[:,0],raw[:,1]))*4)/4
    sources = numpy.fromfile(roads/'edgeSource.bin',dtype='<u4')
    destinations = numpy.fromfile(roads/'edgeTarget.bin',dtype='<u4')
    unique, originalIds, duplicates, degenerate = {}, [], 0, 0
    for edge,(first,second) in enumerate(zip(sources,destinations)):
        endpoints = (tuple(positions[first]),tuple(positions[second]))
        key = tuple(sorted(endpoints))
        if key[0] == key[1]:
            degenerate += 1
        elif key in unique:
            duplicates += 1
        else:
            unique[key] = endpoints
            originalIds.append(edge)
    lines = [LineString(line) for line in unique.values()]
    tree = STRtree(lines)
    source = dict(hashes={str(path.relative_to(DATA)):hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [roads/'nodes.bin',roads/'edgeSource.bin',roads/'edgeTarget.bin']},
        attribution=json.loads((roads/'manifest.json').read_text())['properties'],
        originalNodeCount=len(raw), originalEdgeCount=len(sources), uniqueChordCount=len(lines),
        removedDuplicates=duplicates, removedDegenerate=degenerate,
        precisionGridMeters=0.25, selection='all unique whole chords intersecting fixed window; no clipping')
    for name,bounds in [('loop',[-900,-900,900,900]),('west',[-3000,-1000,-1500,1000])]:
        selected = sorted(int(index) for index in tree.query(shapely.box(*bounds),predicate='intersects'))
        generate(output,'chicago-'+name,[list(lines[index].coords) for index in selected],0,
                 {**source,'window':bounds,'originalEdgeIds':[originalIds[index] for index in selected]})


if __name__ == '__main__':
    main()
