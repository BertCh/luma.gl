# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""NetworkX references for analytics columns and directed k-hop neighborhoods.

All original Chicago nodes/arcs, preserving CSR occurrence order and original IDs.
No geometry/CRS arithmetic or weighted routing: both APIs ignore supplied weights.
Analytics: MultiDiGraph in/out degrees, uniform PageRank (f32 damping, dangling
redistribution), simple weak core numbers (remove loops/parallel adjacency), weak
components labeled by minimum row. Undirected synthetic input is symmetric CSR;
loops count once per CSR occurrence. Normalize complete columns by (x-min)/(max-min),
zero for constant columns. No betweenness/closeness or heuristic communities claim.
Neighborhood: networkx.multi_source_dijkstra_path_length(weight=None,cutoff=k),
outgoing arcs; reverse CSR separately tests incoming arcs. Integer hops are exact.
Invalid/duplicate starts, zero active starts, clamped radius, loops, parallel arcs,
islands, stable nonmonotone IDs, and bounded compaction included. Induced edge sets
come from NetworkX subgraph, retaining every original occurrence; compact order is
CSR row order, count is truncated to capacity, overflow reports untruncated count.
Every complete distance/mask/column array is stored, split only to meet file limits.
"""
import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import networkx
import numpy

HERE=Path(__file__).resolve().parent
DATA=HERE.parents[3]/'examples/deck/spatial-analysis-showcase/public/data/chicago-roads'
VERSIONS={name:importlib.metadata.version(name) for name in ['numpy','networkx']}
VERSIONS['python']=platform.python_version()
NONE=2**32-1

def write(output,name,value):
    value.update(versions=VERSIONS,conventions=__doc__,generator=Path(__file__).name)
    path=output/f'parity-wp3-network-analysis-{name}.json'
    path.write_text(json.dumps(value,separators=(',',':'),allow_nan=False)+'\n')
    assert path.stat().st_size<=1_500_000,(path,path.stat().st_size)
    print(path.name,path.stat().st_size,flush=True)

def generate(output,name,count,edges,directed,source,nodeIds=None):
    # Stable source ordering preserves parallel-arc occurrence IDs.
    ordered=sorted(enumerate(edges),key=lambda entry:entry[1][0])
    sources=[edge[0] for _,edge in ordered]
    neighbors=[edge[1] for _,edge in ordered]
    edgeIds=[index for index,_ in ordered]
    offsets=numpy.r_[0,numpy.cumsum(numpy.bincount(sources,minlength=count))].tolist()
    reverse=sorted(zip(neighbors,sources,edgeIds),key=lambda edge:edge[0])
    reverseOffsets=numpy.r_[0,numpy.cumsum(numpy.bincount([edge[0] for edge in reverse],minlength=count))].tolist()
    reverseNeighbors=[edge[1] for edge in reverse]
    nodeIds=nodeIds or list(range(count))
    write(output,name+'-input',dict(case=name,nodeCount=count,directed=directed,offsets=offsets,neighbors=neighbors,
        edgeIds=edgeIds,nodeIds=nodeIds,source=source,reference='networkx.MultiDiGraph'))
    write(output,name+'-reverse',dict(case=name,offsets=reverseOffsets,neighbors=reverseNeighbors,edgeIds=[edge[2] for edge in reverse],reference='networkx.MultiDiGraph.reverse'))
    graph=networkx.MultiDiGraph()
    graph.add_nodes_from(range(count))
    graph.add_edges_from(zip(sources,neighbors))
    weak=networkx.Graph(graph)
    weak.remove_edges_from(networkx.selfloop_edges(weak))
    cores=networkx.core_number(weak)
    ranks=networkx.pagerank(graph,alpha=float(numpy.float32(.85)),weight=None,tol=1e-15/count,max_iter=4096)
    components={node:min(component) for component in networkx.connected_components(weak) for node in component}
    metrics={'degree':[graph.out_degree(node) for node in graph],
        'inDegree':[graph.in_degree(node) for node in graph],
        'coreNumber':[cores[node] for node in graph],
        'pageRank':[ranks[node] for node in graph],
        'components':[components[node] for node in graph]}
    references={'degree':'networkx.MultiDiGraph.out_degree','inDegree':'networkx.MultiDiGraph.in_degree',
        'coreNumber':'networkx.core_number','pageRank':'networkx.pagerank','components':'networkx.connected_components'}
    for metric,values in metrics.items():
        minimum,maximum=min(values),max(values)
        normalized=[(value-minimum)/(maximum-minimum) if maximum>minimum else 0 for value in values]
        write(output,name+'-'+metric,dict(case=name,metric=metric,values=values,extent=[minimum,maximum],
            normalized=normalized,reference=references[metric]))
    scenarios=[('zero',[0,0,count+3],2,0,False,None),('outgoing',[0,count//2,count+3],3,4,False,None),
        ('clamped',[0,count//2,count+3],3,100,False,None),('incoming',[count//3,count-1,count+3],3,4,True,None),
        ('empty',[0,count//2,count+3],0,4,False,None),('bounded',[0,count//2,count+3],3,4,False,(2,3))]
    for scenario,starts,active,hops,incoming,capacities in scenarios:
        maximumHops=12
        activeGraph=graph.reverse(copy=False) if incoming else graph
        seeds=set(node for node in starts[:active] if node<count)
        distances=networkx.multi_source_dijkstra_path_length(activeGraph,seeds,cutoff=min(hops,maximumHops),weight=None) if seeds else {}
        selected=set(distances)
        induced=activeGraph.subgraph(selected)
        activeOffsets=reverseOffsets if incoming else offsets
        activeNeighbors=reverseNeighbors if incoming else neighbors
        mask=[int(induced.has_edge(node,activeNeighbors[edge])) for node in range(count)
              for edge in range(activeOffsets[node],activeOffsets[node+1])]
        selectedNodes=[nodeIds[node] for node in range(count) if node in selected]
        activeEdgeIds=[edge[2] for edge in reverse] if incoming else edgeIds
        selectedEdges=[activeEdgeIds[edge] for edge,value in enumerate(mask) if value]
        nodeCapacity,edgeCapacity=capacities or (count,len(neighbors))
        write(output,name+'-'+scenario,dict(case=name+'/'+scenario,reference='networkx.multi_source_dijkstra_path_length; MultiDiGraph.subgraph',
            starts=starts,startCount=active,hops=hops,maximumHops=maximumHops,incoming=incoming,
            nodeCapacity=nodeCapacity,edgeCapacity=edgeCapacity,
            hopDistances=[distances.get(node,NONE) for node in range(count)],nodeMask=[int(node in selected) for node in range(count)],
            edgeMask=mask,nodeIds=selectedNodes[:nodeCapacity],edgeIds=selectedEdges[:edgeCapacity],
            nodeCount=min(len(selectedNodes),nodeCapacity),edgeCount=min(len(selectedEdges),edgeCapacity),
            nodeOverflow=int(len(selectedNodes)>nodeCapacity),edgeOverflow=int(len(selectedEdges)>edgeCapacity)))

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output-dir',type=Path,default=HERE/'fixtures')
    output=parser.parse_args().output_dir
    output.mkdir(parents=True,exist_ok=True)
    edges=[(0,1),(0,1),(1,2),(2,0),(2,2),(2,3),(4,5),(5,4),(6,6)]
    generate(output,'synthetic-directed',8,edges,True,dict(description='multiplicity, loops, sink, disconnected component and island'),[91,17,83,42,66,10,99,3])
    symmetric=edges+[(second,first) for first,second in edges if first!=second]
    generate(output,'synthetic-undirected',8,symmetric,False,dict(description='symmetric occurrence CSR, loops count once'),[91,17,83,42,66,10,99,3])
    generate(output,'isolates',5,[],False,dict(description='constant degree/core/rank columns; no edges'))
    manifest=json.loads((DATA/'manifest.json').read_text())
    edges=list(zip(numpy.fromfile(DATA/'edgeSource.bin',dtype='<u4').tolist(),numpy.fromfile(DATA/'edgeTarget.bin',dtype='<u4').tolist()))
    generate(output,'chicago',manifest['nodeCount'],edges,True,dict(attribution=manifest['properties'],
        hashes={name:hashlib.sha256((DATA/name).read_bytes()).hexdigest() for name in ['edgeSource.bin','edgeTarget.bin','nodes.bin','manifest.json']},
        selection='full original graph, every node and edge, original vertex IDs'))

if __name__=='__main__': main()
