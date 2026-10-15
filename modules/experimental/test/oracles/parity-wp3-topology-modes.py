# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Additional showcase modes from independent NetworkX and GEOS operations.
Neighborhood: every integer radius 1..24 with the scene's maximumHops=24; all
integer masks and stable IDs use NetworkX shortest paths/subgraphs. Noding: zero,
unit and maximum scene tolerance, both sorting choices are GPU-only execution
options. The scene capacity choices are tested on the same complete GEOS output;
capacity exhaustion is not claimed as GEOS parity.
Modularity: every resolution in the airline (.2:.1:3) and Bixi (.25:.25:3)
schemas; f32 coefficient rounded before the NetworkX objective evaluation.
"""
import importlib.util
import json
from pathlib import Path
import networkx
import numpy
HERE=Path(__file__).resolve().parent

def loadModule(name):
    specification=importlib.util.spec_from_file_location(name,HERE/f'parity-wp3-{name}.py')
    module=importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module

def main():
    noding=loadModule('noding')
    coordinates=[[[0,0],[4,0]],[[4.25,.25],[8,0]],[[8.75,0],[12,0]],[[12,.75],[16,2]],[[16.25,2.25],[20,2]],[[20.125,3],[24,3]]]
    noding.generate(HERE/'fixtures','showcase-large-snap',coordinates,60,dict(description=__doc__))
    reference=loadModule('graph-communities')
    source=json.loads((HERE/'fixtures/parity-wp3-graph-communities-undirected.json').read_text())
    graph=networkx.MultiGraph()
    graph.add_nodes_from(range(source['nodeCount']))
    graph.add_weighted_edges_from(source['edges'])
    scoring=[]
    resolutions=sorted(set([round(.2+.1*index,2) for index in range(29)]+[.25*index for index in range(1,13)]))
    for partitionName,labels in [('mixed',[0,0,2,2,4,4,6]),('single',[0]*7),('singleton',list(range(7)))]:
        for resolution in resolutions:
            resolution=float(numpy.float32(resolution))
            scoring.append(dict(name=partitionName,resolution=resolution,labels=labels,**reference.evaluate(graph,labels,resolution)))
    reference.write('showcase',dict(source,scoring=scoring,source=dict(description=__doc__)))
    neighborhood=json.loads((HERE/'fixtures/parity-wp3-network-analysis-synthetic-directed-outgoing.json').read_text())
    original=json.loads((HERE/'fixtures/parity-wp3-network-analysis-synthetic-directed-input.json').read_text())
    graph=networkx.MultiDiGraph()
    graph.add_nodes_from(range(original['nodeCount']))
    edges=[(node,original['neighbors'][edge]) for node in graph for edge in range(original['offsets'][node],original['offsets'][node+1])]
    graph.add_edges_from(edges)
    cases=[]
    for hops in range(1,25):
        distances=networkx.single_source_shortest_path_length(graph,0,cutoff=hops)
        induced=graph.subgraph(distances)
        edgeMask=[int(induced.has_edge(source,target)) for source,target in edges]
        selectedNodes=[node for node in graph if node in distances]
        selectedEdges=[edge for edge,keep in enumerate(edgeMask) if keep]
        cases.append(dict(neighborhood,case=f'synthetic-directed/showcase-hops-{hops}',starts=[0,0,0],startCount=1,hops=hops,maximumHops=24,
            hopDistances=[distances.get(node,2**32-1) for node in graph],nodeMask=[int(node in distances) for node in graph],edgeMask=edgeMask,
            nodeIds=selectedNodes,edgeIds=selectedEdges,nodeCount=len(selectedNodes),edgeCount=len(selectedEdges),nodeOverflow=0,edgeOverflow=0))
    path=HERE/'fixtures/parity-wp3-neighborhood-showcase.json'
    path.write_text(json.dumps(cases,separators=(',',':'),allow_nan=False)+'\n')
    print(path.name,len(cases),path.stat().st_size)
if __name__=='__main__':main()
