# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""NetworkX Dijkstra on the documented D8 ground-distance graph.

Composite metric reference: WGS84 first fundamental form with meridional and
prime-vertical radii (Snyder, Map Projections—A Working Manual, USGS PP1395,
https://pubs.usgs.gov/pp/1395/report.pdf). EPSG:3857 inverse latitude is supplied
by pyproj. The differential metric is evaluated at the edge midpoint latitude,
not an ellipsoidal finite geodesic substituted for the class's local metric.
NetworkX independently solves the graph with a virtual source for nonzero
source costs, active-prefix filtering and source-mask union. Endpoint-average
friction, D8 corner cutting, f32 source/calibration/settings and f64 reference.
All cell costs, band labels/counts and full path objectives are compared (L/E).
Both row directions and all three documented metrics; finite/unbounded cost
limits; runtime calibration and source costs; scene bands. No GPU solver port.
"""
import json
import math
import platform
from pathlib import Path
import numpy as np
import networkx as nx
import pyproj
from pyproj import Transformer

HERE = Path(__file__).resolve().parent
OFFSETS = [(0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1),(-1,0),(-1,1)]
NONE = 2**32 - 1
inverse = Transformer.from_crs(3857,4326,always_xy=True)

def encode(value):
    if isinstance(value,dict): return {key:encode(entry) for key,entry in value.items()}
    if isinstance(value,(list,tuple,np.ndarray)): return [encode(entry) for entry in value]
    if isinstance(value,np.integer): return int(value)
    if isinstance(value,(float,np.floating)):
        return float(value) if math.isfinite(value) else ('Infinity' if value > 0 else None)
    return value

def make_case(mode, direction, bounded, source_kind):
    width,height=11,9
    spacing = np.array([.002,.003] if mode=='geographic' else [120,160],dtype=np.float32).astype(float)
    origin = np.array([0,40 if mode=='geographic' else 4800000 if mode=='web-mercator' else 0],dtype=np.float32).astype(float)
    sign = -1 if direction=='south' else 1
    rows,columns = np.indices((height,width))
    raw = (1+(rows*7+columns*13)%19).astype(float)
    raw[2:7,5] = -9999
    validity=np.ones((height,width),dtype=int);validity[4,2]=0
    friction=np.where((raw!=-9999)&(validity!=0),raw*1.25+.5,np.inf)
    graph=nx.Graph();graph.add_nodes_from(range(width*height))
    edges=np.zeros((width*height,8))
    for row,column in np.ndindex((height,width)):
        cell=row*width+column
        for code,(vertical,horizontal) in enumerate(OFFSETS):
            other_row,other_column=row+vertical,column+horizontal
            if not(0<=other_row<height and 0<=other_column<width): continue
            edge_y=origin[1]+sign*spacing[1]*(row+other_row+1)/2
            if mode=='uniform': ground=spacing
            else:
                latitude=edge_y if mode=='geographic' else inverse.transform(0,edge_y)[1]
                angle=np.deg2rad(latitude)
                flattening=1/298.257223563; eccentricity=flattening*(2-flattening)
                denominator=1-eccentricity*np.sin(angle)**2
                prime=6378137/np.sqrt(denominator);meridional=6378137*(1-eccentricity)/denominator**1.5
                unit=np.array([np.cos(angle)*prime,meridional])
                ground=spacing*unit*np.pi/180 if mode=='geographic' else spacing*unit*np.cos(angle)/np.array([6378137*np.cos(angle),6378137])
            cost=np.linalg.norm(ground*[horizontal,vertical])*(friction[row,column]+friction[other_row,other_column])/2
            edges[cell,code]=cost
            if np.isfinite(cost): graph.add_edge(cell,other_row*width+other_column,weight=cost)
    sources=[0, width*height-1, 4];source_costs=[7.,13.,0.]
    source_mask=np.zeros(width*height,dtype=int); source_mask[width*(height-1)]=int(source_kind!='list')
    source_count=0 if source_kind=='mask' else 2
    for cell,cost in zip(sources[:source_count],source_costs[:source_count]):graph.add_edge(-1,cell,weight=cost)
    for cell in np.flatnonzero(source_mask):graph.add_edge(-1,int(cell),weight=0)
    lengths=nx.single_source_dijkstra_path_length(graph,-1)
    full=np.array([lengths.get(cell,np.inf) for cell in range(width*height)])
    limit=float(np.float32(7319.25)) if bounded else np.inf
    costs=np.where(full<=limit,full,np.inf)
    breaks=np.array([1000.25,4000.25,10000.25,40000.25],dtype=np.float32)
    bands=np.searchsorted(breaks,costs,side='left').astype(np.uint32)
    bands[bands==len(breaks)]=NONE
    # No classification boundary ambiguity: nearest independent cost is > 32 ulps.
    finite=costs[np.isfinite(costs)]
    assert min(np.min(abs(finite-value)) for value in breaks)>32*np.spacing(np.float32(max(finite)))
    return dict(name=f'{mode}-{direction}-{source_kind}-'+('limited' if bounded else 'unlimited'),mode=mode,rowDirection=direction,
                width=width,height=height,cellSize=spacing,origin=origin,raw=raw.ravel(),validity=validity.ravel(),
                calibration=[1.25,.5],sourceCosts=source_costs,sources=sources,sourceCount=source_count,sourceKind=source_kind,sourceMask=source_mask,
                costLimit=limit,costs=costs,edges=edges.ravel(),breaks=breaks,bands=bands,
                bandCounts=[int(np.sum(bands==index)) for index in range(len(breaks))])

fixture=dict(reference='networkx.single_source_dijkstra_path_length; composite Snyder WGS84 differential metric; pyproj.Transformer',
 versions=dict(python=platform.python_version(),numpy=np.__version__,networkx=nx.__version__,pyproj=pyproj.__version__),
 cases=[make_case(mode,direction,bounded,source_kind) for mode in ['uniform','web-mercator','geographic'] for direction in ['north','south'] for bounded in [False,True] for source_kind in ['list','mask','union']])
output=HERE/'fixtures/parity-wp4-cost-modes.json';output.write_text(json.dumps(encode(fixture),separators=(',',':'),allow_nan=False)+'\n')
assert output.stat().st_size<1_500_000
print(output.name,output.stat().st_size)
