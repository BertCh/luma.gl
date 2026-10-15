# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""NetworkX line_graph on directed multigraphs; complete sorted CSR outputs.

Every original CSR edge is a distinct line-graph node, including parallel arcs
and loops. No undirected symmetrization. Full Chicago 77,140 arcs, input from
round-6 routing with explicit 1/8-second travel costs; AEQD f32 node coordinates
from round-3 snapping. Full cases use zero turn penalties so NetworkX topology
and following-edge cost are exact, with and without a reproducible list of bans.
Synthetic planar cross includes parallel arcs, loops, zero/negative weights and
an isolated node. Nonzero angular costs are a labelled composite reference:
NetworkX line_graph plus NumPy atan2(cross,dot), then the documented caller's
piecewise turn-penalty function. No standard library defines these custom costs.
Angles use f64 on f32 inputs and parameters. Exact adjacency, inclusive U-turn
cutoff and strict straight cutoff; no synthetic angles near either cutoff.
Numeric costs use S (1e-6*output scale + 1e-4 relative); default-following-edge
costs and all IDs/counts use E. All arrays stored in <=60,000-entry fixture shards.
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
from pathlib import Path

import networkx
import numpy

HERE = Path(__file__).resolve().parent
VERSIONS = {name:importlib.metadata.version(name) for name in ['networkx','numpy']}
VERSIONS['python'] = platform.python_version()


def write(output,name,value):
    value.update(versions=VERSIONS,reference='networkx.line_graph; composite NumPy atan2 turn-cost convention',
                 conventions=__doc__,generator=Path(__file__).name)
    path=output/f'parity-wp3-line-graph-{name}.json'
    path.write_text(json.dumps(value,separators=(',',':'),allow_nan=False)+'\n')
    assert path.stat().st_size <= 1_500_000
    print(path.name,path.stat().st_size,flush=True)


def generate(output,name,topology,positions,source):
    graph=networkx.MultiDiGraph()
    graph.add_nodes_from(range(topology['nodeCount']))
    edgeRows=[]
    for node in range(topology['nodeCount']):
        for edge in range(topology['offsets'][node],topology['offsets'][node+1]):
            destination=topology['neighbors'][edge]
            graph.add_edge(node,destination,key=edge)
            edgeRows.append((node,destination,edge))
    lineGraph=networkx.line_graph(graph)
    turns=sorted((first[2],second[2]) for first,second in lineGraph.edges())
    bans=turns[::max(3,len(turns)//31)]
    positions=numpy.asarray(positions,dtype=numpy.float32).astype(float).reshape(-1,2)
    write(output,name+'-input',dict(case=name,**topology,source=source))
    write(output,name+'-positions',dict(positions=positions.reshape(-1).tolist(),source=source))
    for caseName,parameters,activeBans in [('all',[0,0,0,0,0.5,0,2.9,0],[]),
            ('banned',[0,0,0,0,0.5,len(bans),2.9,0],bans)]+(
            [('turn-costs',[2,5,3,11,0.5,0,2.9,0],[]),('no-uturn',[2,5,3,-1,0.5,len(bans),2.9,0],bans)] if name=='synthetic' else []):
        parameters=numpy.asarray(parameters,dtype=numpy.float32).astype(float).tolist()
        bannedSet=set(activeBans)
        neighbors,weights,counts=[],[],[0]*len(edgeRows)
        for first,second in turns:
            if (first,second) in bannedSet:
                continue
            origin,pivot,_=edgeRows[first]
            _,destination,_=edgeRows[second]
            incoming=positions[pivot]-positions[origin]
            outgoing=positions[destination]-positions[pivot]
            angle=0 if not numpy.any(incoming) or not numpy.any(outgoing) else float(numpy.arctan2(
                incoming[0]*outgoing[1]-incoming[1]*outgoing[0],numpy.dot(incoming,outgoing)))
            if abs(angle)>=parameters[6]:
                penalty=parameters[3]
            else:
                penalty=parameters[0]*abs(angle)+(parameters[1] if angle>parameters[4] else parameters[2] if angle < -parameters[4] else 0)
            if penalty<0:
                continue
            neighbors.append(second)
            weight=topology['weights'][second]
            weights.append(weight+penalty if weight>=0 else -1)
            counts[first]+=1
        offsets=numpy.concatenate(([0],numpy.cumsum(counts))).tolist()
        write(output,name+'-'+caseName,dict(case=name+'-'+caseName,parameters=parameters,
            bans=[value for pair in reversed(activeBans) for value in pair],offsets=offsets,
            count=len(neighbors),costBound=1e-6*max(1,max(weights,default=0)),exactCosts=caseName in ['all','banned']))
        for start in range(0,len(neighbors),60000):
            write(output,name+'-'+caseName+'-'+str(start//60000),dict(neighbors=neighbors[start:start+60000],weights=weights[start:start+60000]))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output-dir',type=Path,default=HERE/'fixtures')
    output=parser.parse_args().output_dir
    output.mkdir(parents=True,exist_ok=True)
    edges=[(0,1,2),(0,1,4),(0,2,0),(0,3,-1),(0,4,3),(0,0,1),
           (1,0,1),(2,0,2),(3,0,3),(4,0,4),(4,1,2)]
    offsets,neighbors,weights=[0],[],[]
    for node in range(6):
        for first,second,weight in edges:
            if first==node:
                neighbors.append(second)
                weights.append(weight)
        offsets.append(len(neighbors))
    generate(output,'synthetic',dict(nodeCount=6,offsets=offsets,neighbors=neighbors,weights=weights),
             [[0,0],[4,0],[0,4],[-4,0],[0,-4],[20,20]],dict(description='directed cross, loops, parallel arcs, zero/closed edge, island'))
    topologyPath=HERE/'fixtures/parity-wp3-routing-chicago-forward-input.json'
    positionsPath=HERE/'fixtures/parity-wp3-snapping-chicago-positions.json'
    topology=json.loads(topologyPath.read_text())
    positions=json.loads(positionsPath.read_text())
    generate(output,'chicago',{key:topology[key] for key in ['nodeCount','offsets','neighbors','weights']},positions['positions'],
             {**topology['source'],'fixtureHashes':{path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in [topologyPath,positionsPath]}})


if __name__=='__main__':
    main()
