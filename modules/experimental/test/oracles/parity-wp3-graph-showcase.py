# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Finite-iteration PageRank from powers of networkx.google_matrix, not a port of
GPU iteration code. The row-stochastic Google matrix has uniform teleportation
and dangling distribution; edge weights ignored. Uniform initial mass multiplied
by numpy.linalg.matrix_power(M,k) is the closed Markov-chain definition (Page et
al. 1999). f32 damping is promoted to f64 before NetworkX constructs M. Store
all ranks, previous ranks, and L1 iteration change. Tests keep the S budget for
finite iterates rather than loosening a stationary-vector tolerance.
Directed/undirected synthetic loops, multiedges and islands use the existing
NetworkX metric fixture. All 5 x 5 street-centrality damping/iteration settings,
plus airline damping and iteration endpoints. Community references use NetworkX
asynchronous LPA and modularity: descriptive heuristic quality, not label equality.
"""
import importlib.metadata
import json
from pathlib import Path
import networkx
import numpy
HERE=Path(__file__).resolve().parent

def main():
    fixtures=[]
    for name in ['synthetic-directed','synthetic-undirected']:
        original=json.loads((HERE/f'fixtures/parity-wp3-graph-metrics-{name}.json').read_text())
        count=original['nodeCount']
        graph=networkx.MultiDiGraph()
        graph.add_nodes_from(range(count))
        for source,target,_ in original['edges']:
            graph.add_edge(source,target)
            if not original['directed'] and source!=target:graph.add_edge(target,source)
        runs=[]
        for damping in [.05,.5,.7,.85,.95,.99]:
            damping=float(numpy.float32(damping))
            transition=numpy.asarray(networkx.google_matrix(graph,alpha=damping,weight=None))
            for iterations in [5,10,20,40,100,200]:
                ranks=numpy.ones(count)/count@numpy.linalg.matrix_power(transition,iterations)
                previous=numpy.ones(count)/count@numpy.linalg.matrix_power(transition,iterations-1)
                runs.append(dict(damping=damping,iterations=iterations,ranks=ranks.tolist(),previous=previous.tolist(),residual=float(numpy.abs(ranks-previous).sum())))
        weak=networkx.Graph(graph)
        weak.remove_edges_from(networkx.selfloop_edges(weak))
        components={node:min(component) for component in networkx.connected_components(weak) for node in component}
        communities=list(networkx.community.asyn_lpa_communities(weak,seed=3))
        communityLabels={node:min(community) for community in communities for node in community}
        ordered=sorted(graph.edges())
        offsets=numpy.r_[0,numpy.cumsum(numpy.bincount([source for source,target in ordered],minlength=count))].tolist()
        reverse=sorted((target,source) for source,target in ordered)
        reverseOffsets=numpy.r_[0,numpy.cumsum(numpy.bincount([source for source,target in reverse],minlength=count))].tolist()
        fixture=dict(original,case='showcase-'+name,runs=runs,reference='networkx.google_matrix; numpy.linalg.matrix_power (Page et al. 1999)',
            versions={name:importlib.metadata.version(name) for name in ['networkx','numpy']},generator=Path(__file__).name,conventions=__doc__,
            offsets=offsets,neighbors=[target for source,target in ordered],reverseOffsets=reverseOffsets,reverseNeighbors=[target for source,target in reverse],
            components=[components[node] for node in graph],communityLabels=[communityLabels[node] for node in graph],communityEdges=[[source,target,1] for source,target in weak.edges()],
            communityScore=networkx.community.modularity(weak,communities))
        fixtures.append(fixture)
    output=HERE/'fixtures/parity-wp3-graph-showcase.json'
    output.write_text(json.dumps(fixtures,separators=(',',':'),allow_nan=False)+'\n')
    print(output.name,output.stat().st_size)
if __name__=='__main__':main()
