# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""NetworkX references on every OpenFlights airport and source edge, plus adversarial multigraphs.

Core numbers: simple weak graph, self-loops removed, parallel/reciprocal edges
collapsed, exactly matching the public GPU contract. Degree: preserve all edge
occurrences; directed in/out degrees use MultiDiGraph. Undirected GPU adjacency
counts a loop once, so subtract nx.number_of_edges(v,v) from MultiGraph.degree.
No weighted-degree API exists: weights are uploaded to prove they are ignored.
PageRank: nx.pagerank(weight=None), uniform teleport and dangling redistribution,
f32-rounded alpha, stringent n*tol stopping in float64. MultiGraph loop adjacency
has one diagonal entry, matching the GPU. Compare stationary ranks after 256 GPU
iterations for alpha<=0.85: L1 truncation <=2*alpha^256 < 1.8e-18. Include alpha=0.
Every rank is checked, together with sum and independently evaluated L1 residual.
Real source weights are counts; no reindexing, invalid edges or graph filtering.
"""
import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import networkx as networkx
import numpy as numpy

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[3]/'examples/deck/spatial-analysis-showcase/public/data/openflights'
VERSIONS = {name: importlib.metadata.version(name) for name in ['numpy','scipy','networkx']}
VERSIONS['python'] = platform.python_version()


def write(output, name, value):
    value.update(case=name, versions=VERSIONS, conventions=__doc__, generator=Path(__file__).name,
                 reference='networkx.core_number; MultiGraph.degree; MultiDiGraph.in_degree/out_degree; networkx.pagerank')
    path = output/f'parity-wp3-graph-metrics-{name}.json'
    path.write_text(json.dumps(value,separators=(',',':'),allow_nan=False)+'\n')
    assert path.stat().st_size <= 1_500_000
    print(path.name,path.stat().st_size,flush=True)


def generate(output, name, nodeCount, edges, directed, source):
    graph = networkx.MultiDiGraph() if directed else networkx.MultiGraph()
    graph.add_nodes_from(range(nodeCount))
    graph.add_weighted_edges_from(edges)
    simple = networkx.Graph(graph)
    simple.remove_edges_from(list(networkx.selfloop_edges(simple)))
    cores = networkx.core_number(simple)
    outgoing = [graph.out_degree(node) if directed else graph.degree(node)-graph.number_of_edges(node,node) for node in graph]
    incoming = [graph.in_degree(node) if directed else outgoing[node] for node in graph]
    rankCases = []
    for alpha in [0, 0.5, float(numpy.float32(0.85))]:
        ranks = networkx.pagerank(graph,alpha=alpha,weight=None,tol=1e-15/nodeCount,max_iter=4096)
        rankCases.append(dict(damping=alpha,iterations=256 if alpha else 2,ranks=[ranks[node] for node in graph]))
    write(output,name,dict(nodeCount=nodeCount,edges=edges,directed=directed,source=source,
          cores=[cores[node] for node in graph],degeneracy=max(cores.values()),outgoing=outgoing,incoming=incoming,
          coreIterations=128,rankCases=rankCases))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output-dir',type=Path,default=HERE/'fixtures')
    output=parser.parse_args().output_dir
    output.mkdir(parents=True,exist_ok=True)
    synthetic=[(0,0,2.5),(0,1,0),(0,1,3),(1,0,7),(1,2,0.125),(2,0,40),(2,3,2),(3,4,1),(4,5,5),(5,3,1),(5,6,2),(8,8,0)]
    sources=numpy.fromfile(DATA/'origin.bin',dtype='<u4')
    targets=numpy.fromfile(DATA/'destination.bin',dtype='<u4')
    weights=numpy.fromfile(DATA/'count.bin',dtype='<u2')
    edges=[(int(first),int(second),int(weight)) for first,second,weight in zip(sources,targets,weights)]
    nodeCount=len(numpy.fromfile(DATA/'position.bin',dtype='<f4'))//2
    source=dict(hashes={name:hashlib.sha256((DATA/name).read_bytes()).hexdigest() for name in
        ['origin.bin','destination.bin','count.bin','position.bin']},
        attribution=json.loads((DATA/'manifest.json').read_text()).get('properties'),
        vertexMapping='all original airport IDs, no remapping')
    for directed in [False,True]:
        suffix='directed' if directed else 'undirected'
        generate(output,'synthetic-'+suffix,10,synthetic,directed,
                 dict(description='loops, parallel/reciprocal edges, zero/nonunit weights, dangling chain, island'))
        generate(output,'openflights-'+suffix,nodeCount,edges,directed,source)
    generate(output,'isolates',5,[],True,dict(description='all dangling, uniform distribution, core and degree zero'))


if __name__ == '__main__':
    main()
