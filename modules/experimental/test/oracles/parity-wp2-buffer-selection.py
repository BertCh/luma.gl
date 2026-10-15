# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Shapely buffer selection for GPUBufferSelection's segment-only feature API.

Input corpus is the independently projected/snapped Chicago and synthetic corpus
from parity-wp2-prepared-nearest.py (regenerate that first). Split every complete
polyline into adjacent segments; repeat its source ID. No vertices are dropped.
GEOS STRtree.query_nearest computes ALL segment ties, then lowest segment row is
selected per API, before source-ID mapping. The same complete nearest set gives
selection at radius 128/3 and radius zero. GEOS dwithin independently verifies
selection for both radii. Expanded segment boxes supply exact scratch demand.
All input/output arrays retained, float32, planar AEQD meters; G tolerance as in
source corpus (16 coordinate ulps). Empty matches use ID 0xffffffff/distance -1.
"""
import hashlib
import json
import platform
from pathlib import Path
import numpy as np
import shapely
from shapely import LineString, Point, STRtree, box

HERE=Path(__file__).resolve().parent
source=HERE/'fixtures/parity-wp2-prepared-nearest.json'
corpus=json.loads(source.read_text())
cases=[]
for case in corpus['cases']:
    frames=[]
    for frame in case['frames']:
        vertices=np.asarray(frame['positions']).reshape(-1,2)
        starts,ends,source_ids=[],[],[]
        for line_index,(start,stop) in enumerate(zip(case['lineOffsets'][:-1],case['lineOffsets'][1:])):
            for row in range(start,stop-1):
                starts.append(vertices[row].tolist()); ends.append(vertices[row+1].tolist()); source_ids.append(case['sourceIds'][line_index])
        tree=STRtree([LineString([start,end]) for start,end in zip(starts,ends)])
        points=[Point(point) for point in np.asarray(frame['queries']).reshape(-1,2)]
        neighbors=[]
        for point in points:
            rows,distances=tree.query_nearest(point,all_matches=True,return_distance=True)
            ordered=sorted(zip(rows,distances),key=lambda pair:(pair[1],pair[0]))
            row,distance=ordered[0]
            neighbors.append([dict(id=source_ids[row],distance=float(distance))] if distance<=case['radius'] else [])
        for radius in [0,case['radius']]:
            pairs=tree.query(points,predicate='dwithin',distance=radius)
            selected=set(map(int,pairs[0]))
            assert selected=={index for index,rows in enumerate(neighbors) if rows and rows[0]['distance']<=radius}
        radius=case['radius']
        boxes=[box(point.x-radius,point.y-radius,point.x+radius,point.y+radius) for point in points]
        candidate_count=tree.query(boxes).shape[1]
        frames.append(dict(name=frame['name'],starts=np.asarray(starts).reshape(-1).tolist(),ends=np.asarray(ends).reshape(-1).tolist(),
                           queries=frame['queries'],neighbors=neighbors,candidateCount=candidate_count))
    cases.append(dict(name=case['name'],radius=case['radius'],sourceIds=source_ids,absolute=case['absolute'],frames=frames))
fixture=dict(generator=Path(__file__).name,reference='shapely.STRtree.query_nearest/query(dwithin)/query(boxes)',
             versions=dict(python=platform.python_version(),numpy=np.__version__,shapely=shapely.__version__,geos=shapely.geos_version_string),
             source=dict(fixture=source.name,sha256=hashlib.sha256(source.read_bytes()).hexdigest(),original=corpus['source']),
             conventions=corpus['conventions']|dict(features='every adjacent segment of complete source polylines; repeated source IDs'),
             tolerance=corpus['tolerance'],cases=cases)
output=HERE/'fixtures/parity-wp2-buffer-selection.json'
output.write_text(json.dumps(fixture,separators=(',',':'),allow_nan=False)+'\n')
assert output.stat().st_size<=1500000
print(output.name,output.stat().st_size,[(case['name'],len(case['sourceIds']),[frame['candidateCount'] for frame in case['frames']]) for case in cases])
