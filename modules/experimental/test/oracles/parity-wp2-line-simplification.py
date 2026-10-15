# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Shapely/GEOS planar Douglas-Peucker on every complete gull-migration track.

Reference https://shapely.readthedocs.io/en/stable/reference/shapely.simplify.html
preserve_topology=False, open lines, strict distance > tolerance. Each bird uses
WGS84 AEQD centered on its first observation, rounded to f32 BEFORE GEOS. This is
planar projected parity, not spherical simplification. All 31 tracks / 98538
vertices retained; separate files preserve whole tracks and avoid 1.5 MB limits.
IDs map GEOS coordinates back to ordered source rows; first/last source rows
are retained even for repeated endpoints. Empty/singleton lines are explicit
API adapters, since GEOS LineString requires >=2 coordinates. Synthetic ties,
exact threshold, repeated/stationary vertices, reversals and collinearity.
No local simplifier is used. Exact row IDs imply identical output geometries
(and zero Hausdorff discrepancy to GEOS); also store GEOS Hausdorff displacement
from the original for fixture validation. No float tolerance on row selection.
"""
import hashlib
import json
import platform
from pathlib import Path
import numpy as np
import pyproj
import shapely
from shapely import LineString

HERE=Path(__file__).resolve().parent
DATA=HERE.parents[3]/'examples/deck/spatial-analysis-showcase/public/data/gull-migration'
versions=dict(python=platform.python_version(),numpy=np.__version__,shapely=shapely.__version__,geos=shapely.geos_version_string,pyproj=pyproj.__version__)
hashes={name:hashlib.sha256((DATA/name).read_bytes()).hexdigest() for name in ['vertices.bin','pathOffsets.bin','manifest.json']}

def make_case(name, lines, tolerances):
    lines=[np.asarray(line,dtype=np.float32).astype(float).reshape(-1,2) for line in lines]
    offsets=np.concatenate([[0],np.cumsum([len(line) for line in lines])]).tolist()
    frames=[]
    for tolerance in tolerances:
        kept=[]; counts=[]; hausdorff=[]
        for first,line in zip(offsets[:-1],lines):
            if len(line)<2:
                selected=list(range(len(line))); displacement=0.
            else:
                geometry=LineString(line)
                simplified=shapely.simplify(geometry,tolerance,preserve_topology=False)
                coordinates=np.asarray(simplified.coords)
                selected=[0]; cursor=1
                for coordinate in coordinates[1:-1]:
                    while not np.array_equal(line[cursor],coordinate): cursor+=1
                    selected.append(cursor); cursor+=1
                selected.append(len(line)-1)
                displacement=float(shapely.hausdorff_distance(geometry,simplified))
                assert displacement<=tolerance+1e-7
            kept.extend(first+index for index in selected); counts.append(len(selected)); hausdorff.append(displacement)
        mask=np.zeros(offsets[-1],dtype=int); mask[kept]=1
        frames.append(dict(tolerance=tolerance,ids=kept,mask=mask.tolist(),lineCounts=counts,lineStarts=np.concatenate([[0],np.cumsum(counts)[:-1]]).tolist(),hausdorff=hausdorff))
    return dict(name=name,positions=np.concatenate(lines).reshape(-1).tolist(),lineOffsets=offsets,frames=frames)

def write(suffix,entry,source):
    path=HERE/f'fixtures/parity-wp2-line-simplification-{suffix}.json'
    fixture=dict(generator=Path(__file__).name,reference='shapely.simplify(preserve_topology=False); shapely.hausdorff_distance',versions=versions,
                 source=source,conventions=dict(method='planar Douglas-Peucker segment distance',inputPrecision='float32 before reference',units='meters',selection='strict distance > tolerance; lowest row tie; endpoints retained'),tolerance=dict(profile='G',ids='exact; identical retained coordinates imply zero Hausdorff difference to GEOS'),cases=[entry])
    path.write_text(json.dumps(fixture,separators=(',',':'),allow_nan=False)+'\n'); assert path.stat().st_size<1500000
    print(path.name,path.stat().st_size,len(entry['positions'])//2)

positions=np.fromfile(DATA/'vertices.bin','<f4').reshape(-1,2)
offsets=np.fromfile(DATA/'pathOffsets.bin','<u4')
for track,(start,stop) in enumerate(zip(offsets[:-1],offsets[1:])):
    coordinates=positions[start:stop]
    origin=coordinates[0]
    crs=f'+proj=aeqd +lon_0={origin[0]} +lat_0={origin[1]} +datum=WGS84 +units=m'
    transformer=pyproj.Transformer.from_crs(4326,crs,always_xy=True)
    projected=np.column_stack(transformer.transform(coordinates[:,0],coordinates[:,1]))
    write(f'gull-{track:02}',make_case(f'complete-gull-track-{track}',[projected],[250.,2000.]),dict(dataset='gull-migration',hashes=hashes,track=track,sourceRange=[int(start),int(stop)],fullCount=98538,crs=crs))
write('edges',make_case('threshold-ties-degeneracy',[
    [],[[3,4]],[[0,0],[0,0],[0,0]],[[0,0],[1,1],[2,0]],
    [[0,0],[1,1],[2,-1],[3,0]],[[0,0],[1,0],[2,0],[1,0]],
    [[0,0],[1,0],[1,0],[2,2],[3,0]],[[0,0],[1,1],[2,0],[0,0]]
],[0.,.5,1.,2.]),dict(dataset='synthetic',crs='local Cartesian'))
