# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Shapely intersection/length of every original segment against polygon features.

https://shapely.readthedocs.io/en/stable/reference/shapely.intersection.html
Uses all 1890 complete Chicago roads / 3105 segments in prior 3km window; source
hash links the original hashed dataset and selection. Polygons: 64 overlapping
500m windows with 1/32m offset (source snaps to 1/16m), holes, multipolygon and
empty-output feature. STRtree prunes intersections; GEOS supplies clipped pieces.
Intersections are per original segment: no union loses retraced multiplicity.
Per-feature sums use float64, weighted by exact f32 per-line weights. Counts are
segments with positive length (not number of clipped pieces). Stationary segments,
singletons, corner-only contact, holes and retracing are explicit synthetic cases.
Exact ring-edge overlap is documented ambiguous by the GPU and is NOT included;
the generator asserts no input has positive-length overlap with any boundary.
Planar only, local meters. GEOS runs on f32 inputs. Empty cells exactly zero.
G per-feature bound: 16 coordinate ulps per clipped piece plus gamma_(pieces-1)
for piece sums and gamma_(contributors-1) for reduction. Weighted bound scales
geometry errors by abs(weight) and includes one rounding per multiplication.
This analytic bound is established before GPU execution, not fitted to results.
"""
import hashlib
import json
import platform
from pathlib import Path
import numpy as np
import shapely
from shapely import LineString,Polygon,MultiPolygon,STRtree,box
HERE=Path(__file__).resolve().parent
source=HERE/'fixtures/parity-wp2-prepared-nearest.json'
roads=json.loads(source.read_text())['cases'][0]

def make_case(name,positions,offsets,features,weights):
    positions=np.asarray(positions,dtype=np.float32).astype(float).reshape(-1,2)
    features=[shapely.transform(feature,lambda coordinates: coordinates.astype(np.float32).astype(float)) for feature in features]
    tree=STRtree(features)
    lengths=np.zeros(len(features)); weighted=np.zeros(len(features)); counts=np.zeros(len(features),dtype=int)
    pieces=np.zeros(len(features),dtype=int); weighted_pieces=np.zeros(len(features))
    candidate_count=0; stationary_count=0
    for line,(start,stop) in enumerate(zip(offsets[:-1],offsets[1:])):
        for first,second in zip(positions[start:stop-1],positions[start+1:stop]):
            if np.array_equal(first,second):
                stationary_count+=1
                continue
            segment=LineString([first,second])
            candidate_count+=len(tree.query(segment))
            for feature in tree.query(segment,predicate='intersects'):
                assert segment.intersection(features[feature].boundary).length==0, 'ambiguous boundary overlap'
                intersection=segment.intersection(features[feature]); length=intersection.length
                if length>0:
                    lengths[feature]+=length; weighted[feature]+=length*weights[line]; counts[feature]+=1
                    piece_count=sum(part.length>0 for part in shapely.get_parts(intersection))
                    pieces[feature]+=piece_count; weighted_pieces[feature]+=piece_count*abs(weights[line])
    polygon_positions=[]; ring_offsets=[0]; polygon_offsets=[0]; feature_offsets=[0]
    for feature in features:
        for polygon in shapely.get_parts(feature):
            for ring in [polygon.exterior,*polygon.interiors]:
                polygon_positions.extend(np.asarray(ring.coords)[:-1].reshape(-1).tolist()); ring_offsets.append(len(polygon_positions)//2)
            polygon_offsets.append(len(ring_offsets)-1)
        feature_offsets.append(len(polygon_offsets)-1)
    scale=max(1.,np.abs(positions).max(),np.abs(polygon_positions).max())
    geometry=16*float(np.spacing(np.float32(scale)))
    epsilon=np.finfo(np.float32).eps
    def gamma(operations): return operations*epsilon/(1-operations*epsilon)
    summation=gamma(np.maximum(pieces-1,0))+gamma(np.maximum(counts-1,0))
    return dict(name=name,positions=positions.reshape(-1).tolist(),lineOffsets=offsets,lineWeights=weights,
                polygons=dict(positions=polygon_positions,ringOffsets=ring_offsets,polygonOffsets=polygon_offsets,featureOffsets=feature_offsets),
                expected=dict(lengths=lengths.tolist(),weightedLengths=weighted.tolist(),segmentCounts=counts.tolist()),
                lengthBounds=(pieces*geometry+summation*lengths).tolist(),weightedBounds=(weighted_pieces*geometry+(summation+epsilon)*weighted).tolist(),
                candidateCount=candidate_count,stationarySegmentCount=stationary_count)

features=[]
for row in range(8):
    for column in range(8):
        left=-2000+column*500+.03125; bottom=-2000+row*500+.03125
        outer=box(left,bottom,left+625,bottom+625)
        features.append(Polygon(outer.exterior.coords,[box(left+150,bottom+200,left+350,bottom+450).exterior.coords]) if (row+column)%3==0 else outer)
features.extend([MultiPolygon([box(-2500.03125,-1000.03125,-1800.03125,-500.03125),box(1800.03125,500.03125,2500.03125,1000.03125)]),box(100000,100000,100010,100010)])
real=make_case('complete-chicago-roads-holes-overlaps-multipart',roads['frames'][0]['positions'],roads['lineOffsets'],features,
               [[.5,1.,2.][line%3] for line in range(len(roads['lineOffsets'])-1)])
lines=[[[-5,5],[15,5]],[[2,2],[3,3],[3,8]],[[25,-5],[25,15],[45,15],[45,5]],
       [[-1,-1],[0,0]],[[1,1],[1,1]],[[3,3]],[[1,2],[9,2],[1,2]],[[100,0],[101,0]]]
offsets=np.concatenate([[0],np.cumsum([len(line) for line in lines])]).tolist()
edge=make_case('holes-retrace-corner-singleton-stationary',np.concatenate(lines),offsets,
               [Polygon(box(0,0,10,10).exterior.coords,[box(4,4,6,6).exterior.coords]),MultiPolygon([box(20,0,30,10),box(40,0,50,10)]),box(100,100,110,110)],
               [1.,3.,2.,.5,1.,1.,2.,1.])
fixture=dict(generator=Path(__file__).name,reference='shapely.STRtree.query + intersection/length; float64 per-feature weighted sum',
             versions=dict(python=platform.python_version(),numpy=np.__version__,shapely=shapely.__version__,geos=shapely.geos_version_string),
             source=dict(dataset='prior complete Chicago road window',hashes={source.name:hashlib.sha256(source.read_bytes()).hexdigest()}),
             conventions=dict(crs='WGS84 AEQD -87.63,41.88',inputPrecision='float32 before reference',units='meters',boundary='corner-only zero; positive ring-edge overlap excluded per public ambiguity',multiplicity='each source segment separately'),
             tolerance=dict(profile='G',length='16 coordinate ulps per piece + gamma piece and segment sums',weighted='weighted geometry bound + multiplication/summation roundoff'),cases=[real,edge])
path=HERE/'fixtures/parity-wp2-line-length-per-polygon.json'
path.write_text(json.dumps(fixture,separators=(',',':'),allow_nan=False)+'\n'); assert path.stat().st_size<1500000
print(path.name,path.stat().st_size,[(entry['candidateCount'],sum(entry['expected']['segmentCounts'])) for entry in fixture['cases']])
