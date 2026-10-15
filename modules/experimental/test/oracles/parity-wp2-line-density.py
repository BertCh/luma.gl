# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""GEOS segment/cell intersection lengths, preserving multiplicity of source paths.

shapely.STRtree.query(predicate='intersects') prunes cells; shapely.intersection
and length measure every segment-cell piece. Sum per cell in float64, divide by
cellWidth*cellHeight for planar density. No query-disk/kernel API exists here.
Reuses all 1890 complete roads (3105 segments) in the independently selected
Chicago 3km window, including paths extending outside it; no vertex thinning.
Two grids exercise panning/anisotropic cells. Half-open internal cells own their
left/bottom edge; last column/row also owns the outer max edge. GEOS difference
removes right/top boundary line overlap, retaining the GPU's documented floor
ownership without copying its clipping/traversal. Synthetic exact grid lines,
corners, retraced paths, stationary segments, singleton and outside lines.
G budget is per-cell: <=16 coordinate ulps PER contributing segment, plus the
standard gamma_(m-1) float32 summation bound times the positive reference sum.
Each cell stores its own derived bound; density divides this bound by cell area
and adds one epsilon32 times density. No global empirical tolerance inflation.
"""
import hashlib
import json
import platform
from pathlib import Path
import numpy as np
import shapely
from shapely import LineString, STRtree, box

HERE=Path(__file__).resolve().parent
source=HERE/'fixtures/parity-wp2-prepared-nearest.json'
roads=json.loads(source.read_text())['cases'][0]


def make_case(name,positions,offsets,columns,rows,grids):
    positions=np.asarray(positions,dtype=np.float32).astype(float).reshape(-1,2)
    segments=[LineString([first,second]) for start,stop in zip(offsets[:-1],offsets[1:])
              for first,second in zip(positions[start:stop-1],positions[start+1:stop]) if not np.array_equal(first,second)]
    frames=[]
    for minimum_x,minimum_y,width,height in grids:
        cells=[box(minimum_x+column*width,minimum_y+row*height,minimum_x+(column+1)*width,minimum_y+(row+1)*height)
               for row in range(rows) for column in range(columns)]
        tree=STRtree(cells)
        lengths=np.zeros(len(cells)); contributions=np.zeros(len(cells),dtype=int)
        for segment in segments:
            for cell_index in tree.query(segment,predicate='intersects'):
                piece=shapely.intersection(segment,cells[cell_index])
                row,column=divmod(int(cell_index),columns)
                left,bottom,right,top=cells[cell_index].bounds
                if column+1<columns: piece=shapely.difference(piece,LineString([(right,bottom),(right,top)]))
                if row+1<rows: piece=shapely.difference(piece,LineString([(left,top),(right,top)]))
                if piece.length>0:
                    lengths[cell_index]+=piece.length
                    contributions[cell_index]+=1
        scale=max(1.,np.abs(positions).max(),abs(minimum_x),abs(minimum_y),abs(minimum_x+columns*width),abs(minimum_y+rows*height))
        quantization=16*float(np.spacing(np.float32(scale)))
        epsilon=np.finfo(np.float32).eps
        operations=np.maximum(contributions-1,0)
        gamma=operations*epsilon/(1-operations*epsilon)
        bounds=contributions*quantization+gamma*lengths
        density=lengths/(width*height)
        # Empty cells must stay exactly zero; no floor hides leaked geometry.
        frames.append(dict(parameters=[minimum_x,minimum_y,width,height],lengths=lengths.tolist(),densities=density.tolist(),
                           lengthBounds=bounds.tolist(),densityBounds=(bounds/(width*height)+epsilon*density).tolist(),
                           positivePieces=int(contributions.sum()),contributors=contributions.tolist()))
    return dict(name=name,positions=positions.reshape(-1).tolist(),lineOffsets=offsets,columns=columns,rows=rows,frames=frames)

real=make_case('chicago-complete-road-window',roads['frames'][0]['positions'],roads['lineOffsets'],12,12,
               [[-1500,-1500,250,250],[-1375,-1250,200,225]])
lines=[[[-1,-1],[5,5]],[[0,1],[4,1]],[[2,0],[2,4]],[[0,4],[4,4]],[[4,0],[4,4]],
       [[0,0],[4,0]],[[.5,.5],[3.5,.5],[.5,.5]],[[1,1],[1,1]],[[2,2]],[[10,10],[20,20]]]
offsets=np.concatenate([[0],np.cumsum(list(map(len,lines))) ]).tolist()
edge=make_case('grid-boundaries-corners-multiplicity-and-degeneracy',np.concatenate(lines),offsets,4,4,[[0,0,1,1],[-.5,-.25,1,1.25]])
fixture=dict(generator=Path(__file__).name,reference='shapely.STRtree.query + intersection/difference/length; float64 per-cell sum',
             versions=dict(python=platform.python_version(),numpy=np.__version__,shapely=shapely.__version__,geos=shapely.geos_version_string),
             source=dict(dataset='prior independently selected complete Chicago road window',hashes={source.name:hashlib.sha256(source.read_bytes()).hexdigest()}),
             conventions=dict(inputPrecision='float32; real source snapped to 1/16m',crs='WGS84 AEQD -87.63,41.88',units='meters; density m/m^2',boundary='half-open except closed outer maximum',multiplicity='each source segment separately'),
             tolerance=dict(profile='G',length='16 coordinate ulps per contributor + gamma(m-1)*sum',density='length bound / area + epsilon32*density'),cases=[real,edge])
path=HERE/'fixtures/parity-wp2-line-density.json'
path.write_text(json.dumps(fixture,separators=(',',':'),allow_nan=False)+'\n')
assert path.stat().st_size<=1500000
print(path.name,path.stat().st_size,[(entry['name'],[frame['positivePieces'] for frame in entry['frames']]) for entry in fixture['cases']])
