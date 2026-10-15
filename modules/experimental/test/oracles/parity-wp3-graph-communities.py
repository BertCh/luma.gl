# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""NetworkX references for GPUGraphLabelPropagation and GPUGraphModularity.

OpenFlights: all 3257 airports and 18930 undirected source pairs; no subsampling.
Modularity uses count.bin airline-route weights (integers exactly representable
in f32); LPA ignores edge weights, as documented by the GPU API, and uses unit
weak-neighbor multiplicity. LPA reference is NetworkX asyn_lpa_communities with
seeds 3,17,41,97. The GPU is synchronous, self-voting, minimum-label tie breaking;
its seed runs instead permute vertex IDs with those seeds, preserving topology.
These are DIFFERENT heuristics: compare feasibility and independently evaluated
modularity quality ratios, never require identical labels or assert equal quality.
Small disjoint complete components + islands do have an unambiguous partition.

Modularity: networkx.community.modularity on MultiGraph/MultiDiGraph, preserving
parallel edges, reciprocal source occurrences, zero weights and self loops;
resolutions 0, 0.5, 1, 2. All f32 weights evaluated in f64. Per-community terms
are a composite reference: NetworkX subgraph.size and degree/out_degree/in_degree
in the published Newman definition (NetworkX modularity documentation,
https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.community.quality.modularity.html).
Their sum is checked against the established modularity function on every case.
Unused stable community IDs have zero contributions; islands are preserved.
All label arrays and contribution arrays are retained. Profile S, no enlarged
bound. Deterministic LPA partitions have exact canonical labels; heuristic ratios
are descriptive, and a poor ratio must be reported rather than hidden by a gate.
"""
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import networkx
import numpy
from sklearn.metrics import adjusted_rand_score

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[3] / 'examples/deck/spatial-analysis-showcase/public/data/openflights'
VERSIONS = {name: importlib.metadata.version(name) for name in ['networkx','numpy','scikit-learn']}
VERSIONS['python'] = platform.python_version()
SEEDS = [3,17,41,97]


def canonical(communities, nodeCount):
    labels = [0] * nodeCount
    for community in communities:
        for node in community:
            labels[node] = min(community)
    return labels


def evaluate(graph, labels, resolution):
    communities = [{node for node,label in enumerate(labels) if label == community} for community in sorted(set(labels))]
    score = networkx.community.modularity(graph, communities, weight='weight', resolution=resolution)
    totalWeight = graph.size(weight='weight')
    contributions = [0.0]*len(labels)
    for community in communities:
        internal = graph.subgraph(community).size(weight='weight')
        if graph.is_directed():
            outgoing = sum(value for _,value in graph.out_degree(community, weight='weight'))
            incoming = sum(value for _,value in graph.in_degree(community, weight='weight'))
            term = internal/totalWeight - resolution*outgoing*incoming/totalWeight**2
        else:
            volume = sum(value for _,value in graph.degree(community, weight='weight'))
            term = internal/totalWeight - resolution*(volume/(2*totalWeight))**2
        contributions[labels[next(iter(community))]] = term
    assert abs(sum(contributions)-score) < 1e-12
    return dict(score=score, contributions=contributions)


def write(name, fixture):
    fixture.update(case=name, generator=Path(__file__).name, versions=VERSIONS, conventions=__doc__,
                   reference='networkx.community.modularity; networkx.community.asyn_lpa_communities')
    path = HERE/'fixtures'/f'parity-wp3-graph-communities-{name}.json'
    path.write_text(json.dumps(fixture,separators=(',',':'),allow_nan=False)+'\n')
    assert path.stat().st_size <= 1_500_000
    print(path.name,path.stat().st_size,flush=True)


def generate(name,nodeCount,edges,directed,partitions,source,propagation=False,exactPartition=False):
    graph = networkx.MultiDiGraph() if directed else networkx.MultiGraph()
    graph.add_nodes_from(range(nodeCount))
    graph.add_weighted_edges_from(edges)
    scoring = []
    for partitionName,labels in partitions:
        for resolution in [0,0.5,1,2]:
            scoring.append(dict(name=partitionName,resolution=resolution,labels=labels,**evaluate(graph,labels,resolution)))
    runs = []
    if propagation:
        weak = networkx.MultiGraph()
        weak.add_nodes_from(graph.nodes)
        weak.add_edges_from((first,second) for first,second,_ in edges if first != second)
        for seed in SEEDS:
            communities = list(networkx.community.asyn_lpa_communities(weak,weight=None,seed=seed))
            labels = canonical(communities,nodeCount)
            unweighted = weak.copy()
            networkx.set_edge_attributes(unweighted,1,'weight')
            runs.append(dict(seed=seed,labels=labels,permutation=numpy.random.default_rng(seed).permutation(nodeCount).tolist(),
                             score=evaluate(unweighted,labels,1)['score']))
    stability = [[adjusted_rand_score(first['labels'], second['labels']) for second in runs] for first in runs]
    components = canonical(networkx.connected_components(graph.to_undirected()), nodeCount)
    write(name,dict(components=components, stability=stability, nodeCount=nodeCount,edges=edges,directed=directed,scoring=scoring,runs=runs,source=source,
                   exactPartition=exactPartition,tolerance=dict(absolute=1e-6,relative=1e-4)))


def main():
    edges=[(0,0,2.5),(0,1,1.25),(0,1,0.75),(1,0,3),(1,2,0),(2,3,4.5),(3,2,1),(3,3,0.25),(4,5,2)]
    for directed in [False,True]:
        generate('directed' if directed else 'undirected',7,edges,directed,
                 [('mixed',[0,0,2,2,4,4,6]),('single',[0]*7),('singleton',list(range(7)))],
                 dict(description='weighted loops, parallel/reciprocal occurrences, zero edge, island'))
    cliqueGraph = networkx.disjoint_union_all([networkx.complete_graph(5),networkx.complete_graph(7),networkx.empty_graph(2)])
    cliqueEdges = [(first,second,1) for first,second in cliqueGraph.edges]
    cliqueLabels = canonical(networkx.connected_components(cliqueGraph),14)
    generate('cliques',14,cliqueEdges,False,[('components',cliqueLabels)],dict(description='disjoint cliques and two islands'),True,True)
    sources=numpy.fromfile(DATA/'origin.bin',dtype='<u4')
    targets=numpy.fromfile(DATA/'destination.bin',dtype='<u4')
    weights=numpy.fromfile(DATA/'count.bin',dtype='<u2')
    nodeCount=len(numpy.fromfile(DATA/'position.bin',dtype='<f4'))//2
    edges=[(int(first),int(second),int(weight)) for first,second,weight in zip(sources,targets,weights)]
    graph=networkx.Graph()
    graph.add_nodes_from(range(nodeCount))
    graph.add_weighted_edges_from(edges)
    partition=canonical(networkx.community.louvain_communities(graph,weight='weight',seed=20261015),nodeCount)
    generate('openflights',nodeCount,edges,False,[('networkx-louvain',partition)],
             dict(hashes={name:hashlib.sha256((DATA/name).read_bytes()).hexdigest() for name in
                  ['origin.bin','destination.bin','count.bin','position.bin']},
                  attribution=json.loads((DATA/'manifest.json').read_text()).get('properties'),
                  vertexMapping='every original airport index preserved',referencePartitionSeed=20261015),True)


if __name__ == '__main__':
    main()
