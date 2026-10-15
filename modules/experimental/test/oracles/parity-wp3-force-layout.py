# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Composite energy reference for GPUGraphForceLayout, with external optimizers.

NetworkX spring_layout(method='force',weight=None,scale=None) uses Fruchterman-
Reingold's quadratic attraction force, unlike GPU Hooke attraction. Coordinates
are NOT equal-answer references. Evaluate both layouts in the SAME explicitly
specified generalized spring/log-repulsion energy. Center NetworkX output and
optimize its global scale in that energy (preserving shape) to remove arbitrary
FR coordinate scale as a confounder. Use scipy.optimize.minimize
(L-BFGS-B) as an additional model-aligned quality comparator, not a global optimum.
Citation: Noack, Energy Models for Graph Clustering, JGAA 11(2), 2007,
https://jgaa.info/index.php/jgaa/article/view/paper154 (general energy framework).
E(X)=A/2 tr(X^T L X)+G/2 ||X||^2-R sum_{i<j} log(||X_i-X_j||).
L is scipy.sparse.csgraph.laplacian on the unweighted occurrence adjacency.
The log potential continues quadratically below d^2=1e-4, continuously in energy
and gradient. This defines the public softened force model, without reproducing
GPU stepping, tiling, damping, clipping, initialization or convergence logic.
All supplied positions and coefficients are rounded f32. Parallel edges strengthen
springs, loops have zero energy, directed edges are symmetrized by occurrence.
Real input: deterministic 512-airport BFS prefix in the largest OpenFlights weak
component; induced source edges, original IDs and full-data hashes retained.
This is subset O(V^2) layout coverage, NOT full-airport layout coverage. Three seeds.
Every reference edge length, energy component and gradient coordinate is stored.
"""
import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import networkx
import numpy
from scipy import optimize, sparse
from scipy.sparse.csgraph import laplacian
from scipy.spatial.distance import pdist, squareform

HERE=Path(__file__).resolve().parent
DATA=HERE.parents[3]/'examples/deck/spatial-analysis-showcase/public/data/openflights'
VERSIONS={name:importlib.metadata.version(name) for name in ['numpy','scipy','networkx']}
VERSIONS['python']=platform.python_version()
MINIMUM_SQUARED_DISTANCE=1e-4


def evaluate(positions, edges, parameters, matrix, includeGeometry=True):
    # NetworkX's sparse FR path returns f32; evaluate ALL references in f64, not
    # a mixture of scipy f64 distances and numpy f32 norms/gravity reductions.
    positions=numpy.asarray(positions,dtype=float)
    repulsion, attraction, gravity=(parameters[name] for name in ['repulsion','attraction','gravity'])
    squared=squareform(pdist(positions,'sqeuclidean'))
    upper=squared[numpy.triu_indices(len(positions),1)]
    potential=numpy.log(numpy.maximum(upper,MINIMUM_SQUARED_DISTANCE)) + numpy.minimum(upper/MINIMUM_SQUARED_DISTANCE-1,0)
    inverse=1/numpy.maximum(squared,MINIMUM_SQUARED_DISTANCE)
    numpy.fill_diagonal(inverse,0)
    gradient=attraction*(matrix@positions)+gravity*positions-repulsion*(inverse.sum(axis=1)[:,None]*positions-inverse@positions)
    components=[float(attraction/2*numpy.sum(positions*(matrix@positions))),float(gravity/2*numpy.sum(positions**2)),float(-repulsion/2*potential.sum())]
    lengths=[float(numpy.linalg.norm(positions[first]-positions[second])) for first,second,_ in edges] if includeGeometry else []
    return dict(positions=positions.reshape(-1).tolist(),energy=sum(components),components=components,gradient=gradient.reshape(-1).tolist(),edgeLengths=lengths,
                rootMeanSquareGradient=float(numpy.sqrt(numpy.mean(numpy.sum(gradient**2,axis=1)))))


def generate(output,name,nodeCount,edges,source,parameterOverrides=None):
    graph=networkx.MultiGraph()
    graph.add_nodes_from(range(nodeCount))
    graph.add_edges_from((first,second) for first,second,_ in edges)
    adjacency=networkx.to_scipy_sparse_array(graph,weight=None,dtype=float)
    matrix=laplacian(adjacency)
    parameters={key:float(numpy.float32(value)) for key,value in dict(repulsion=0.2,attraction=0.1,gravity=0.02,damping=0.85,timeStep=0.05,maxVelocity=2).items()}
    if parameterOverrides:
        parameters.update({key:float(numpy.float32(value)) for key,value in parameterOverrides.items()})
    if name=='equilibrium':
        parameters.update(repulsion=2,attraction=0.5,gravity=0)
    runs=[]
    for seed in ([3] if name=='equilibrium' else [3,17,41]):
        initial=(numpy.array([[-1,0],[1,0]]) if name=='equilibrium' else numpy.random.default_rng(seed).uniform(-4,4,(nodeCount,2))).astype(numpy.float32).astype(float)
        def objective(flat):
            # Model derivative from sparse Laplacian and pair-distance energy, not GPU integrator.
            values=evaluate(flat.reshape(-1,2),edges,parameters,matrix,includeGeometry=False)
            return values['energy'],numpy.array(values['gradient'])
        optimized=optimize.minimize(objective,initial.reshape(-1),method='L-BFGS-B',jac=True,
            options=dict(maxiter=400,gtol=1e-7,ftol=1e-12,maxls=40))
        spring=networkx.spring_layout(graph,pos=dict(enumerate(initial)),seed=seed,iterations=256,
            weight=None,scale=None,method='force',threshold=1e-7)
        springPositions=numpy.array([spring[node] for node in graph],dtype=float)
        springPositions-=springPositions.mean(axis=0)
        scaling=optimize.minimize_scalar(lambda logScale:evaluate(springPositions*numpy.exp(logScale),edges,parameters,matrix,includeGeometry=False)['energy'],
            bounds=(-8,8),method='bounded',options=dict(xatol=1e-12))
        springPositions*=numpy.exp(scaling.x)
        runs.append(dict(seed=seed,initial=evaluate(initial,edges,parameters,matrix),
            optimized=evaluate(optimized.x.reshape(-1,2),edges,parameters,matrix),
            networkx=evaluate(springPositions,edges,parameters,matrix),
            optimizer=dict(success=bool(optimized.success),message=str(optimized.message),iterations=int(optimized.nit))))
    value=dict(case=name,nodeCount=nodeCount,edges=edges,directed=False,parameters=parameters,runs=runs,
        iterationsPerFrame=32,frames=8,exactEquilibrium=name=='equilibrium',source=source,
        reference='networkx.spring_layout; scipy.optimize.minimize; composite generalized spring/log energy',
        versions=VERSIONS,conventions=__doc__,generator=Path(__file__).name)
    # Full arrays exceed the ceiling when three real runs share a JSON; split by seed.
    fixtures = [{**value, 'case': name+'-seed-'+str(run['seed']), 'runs': [run]} for run in runs] if name=='openflights' else [value]
    for fixture in fixtures:
        path=output/f"parity-wp3-force-layout-{fixture['case']}.json"
        path.write_text(json.dumps(fixture,separators=(',',':'),allow_nan=False)+'\n')
        assert path.stat().st_size <= 1_500_000
        print(path.name,path.stat().st_size,[(run['seed'],run['optimizer']) for run in fixture['runs']],flush=True)



def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output-dir',type=Path,default=HERE/'fixtures')
    output=parser.parse_args().output_dir
    output.mkdir(parents=True,exist_ok=True)
    generate(output,'equilibrium',2,[(0,1,7)],dict(description='exact stationary pair, nonunit edge weight ignored'))
    generate(output,'synthetic',8,[(0,1,2),(0,1,0),(1,2,4),(2,0,1),(3,4,9),(4,5,1),(5,3,1),(6,6,5)],dict(description='parallel edges, loop, disconnected triangles and two islands'))
    sources=numpy.fromfile(DATA/'origin.bin',dtype='<u4')
    targets=numpy.fromfile(DATA/'destination.bin',dtype='<u4')
    weights=numpy.fromfile(DATA/'count.bin',dtype='<u2')
    nodeCount=len(numpy.fromfile(DATA/'position.bin',dtype='<f4'))//2
    graph=networkx.Graph()
    graph.add_nodes_from(range(nodeCount))
    graph.add_edges_from(zip(sources.tolist(),targets.tolist()))
    component=max(networkx.connected_components(graph),key=len)
    root=min(component,key=lambda node:(-graph.degree(node),node))
    selected=[root]+[target for _,target in networkx.bfs_edges(graph,root,sort_neighbors=sorted)]
    selected=selected[:512]
    mapping={node:index for index,node in enumerate(selected)}
    edges=[(mapping[int(first)],mapping[int(second)],int(weight)) for first,second,weight in zip(sources,targets,weights) if first in mapping and second in mapping]
    source=dict(hashes={name:hashlib.sha256((DATA/name).read_bytes()).hexdigest() for name in ['origin.bin','destination.bin','count.bin','position.bin']},
        attribution=json.loads((DATA/'manifest.json').read_text()).get('properties'),originalVertexIds=selected,
        fullNodeCount=nodeCount,fullEdgeCount=len(sources),selection='512 BFS prefix from highest-degree vertex in largest weak component, neighbors in ID order')
    generate(output,'openflights',len(selected),edges,source)


if __name__=='__main__':
    main()
