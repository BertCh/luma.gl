# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Pinned spatial-force quality against independent constrained optimizers.

Same generalized quadratic-spring/log-repulsion energy as round 4 (Noack 2007,
https://jgaa.info/index.php/jgaa/article/view/paper154), evaluated by SciPy pair
 distances and Laplacian; reuse the WP3 reference definition, never GPU stepping.
Two f32 initial coordinates are fixed exactly; scipy.optimize.minimize L-BFGS-B
optimizes only free coordinates. networkx.spring_layout(method='force', fixed=...,
weight=None,scale=None) supplies a different-force-law quality comparator, never
position equality. No centering/rescaling is permitted with fixed anchors.
Grid theta=0 is exact force evaluation; theta=.6 is flat-grid monopole, not a new
spatial anchoring objective. Compare both outputs under the same independent exact
energy and free-vertex gradient, with complete edge lengths and exact anchor checks.
Three initial seeds, same 512-airport BFS subset from round 4 plus synthetic loops,
parallel edges, disconnected components, islands and analytic stationary pair.
Spatial bounds [-32,32]^2 contain initial coordinates plus the analytic maximum
256*dt*maxVelocity displacement. Inputs/coefficients are rounded to f32; unweighted.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import networkx
import numpy
from scipy import optimize, stats
from scipy.sparse.csgraph import laplacian

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('energy_reference',HERE/'parity-wp3-force-layout.py')
energy=importlib.util.module_from_spec(spec)
spec.loader.exec_module(energy)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output-dir',type=Path,default=HERE/'fixtures')
    parser.add_argument('--names',nargs='+',default=['equilibrium','synthetic','openflights-seed-3','openflights-seed-17','openflights-seed-41'])
    parser.add_argument('--grid-size',type=int,default=32)
    parser.add_argument('--bound',type=float,default=32)
    arguments=parser.parse_args()
    output=arguments.output_dir
    gridSize=arguments.grid_size
    bound=arguments.bound
    output.mkdir(parents=True,exist_ok=True)
    for name in arguments.names:
        source=json.loads((HERE/f'fixtures/parity-wp3-force-layout-{name}.json').read_text())
        graph=networkx.MultiGraph()
        graph.add_nodes_from(range(source['nodeCount']))
        graph.add_edges_from((first,second) for first,second,_ in source['edges'])
        matrix=laplacian(networkx.to_scipy_sparse_array(graph,weight=None,dtype=float))
        pinned=[0] if name=='equilibrium' else [0,source['nodeCount']//3]
        free=[node for node in graph if node not in pinned]
        def evaluate(positions,includeGeometry=True):
            return energy.evaluate(positions,source['edges'],source['parameters'],matrix,includeGeometry)
        runs=[]
        for run in source['runs']:
            initial=numpy.array(run['initial']['positions']).reshape(-1,2)
            def expand(flat):
                positions=initial.copy()
                positions[free]=flat.reshape(-1,2)
                return positions
            def objective(flat):
                values=evaluate(expand(flat),False)
                return values['energy'],numpy.array(values['gradient']).reshape(-1,2)[free].reshape(-1)
            optimized=optimize.minimize(objective,initial[free].reshape(-1),method='L-BFGS-B',jac=True,
                options=dict(maxiter=400,gtol=1e-7,ftol=1e-12,maxls=40))
            spring=networkx.spring_layout(graph,pos=dict(enumerate(initial)),fixed=pinned,seed=run['seed'],
                iterations=256,weight=None,scale=None,method='force',threshold=1e-7)
            # SciPy rectangular-bin statistics provide an independent index reference.
            binned=stats.binned_statistic_2d(initial[:,0],initial[:,1],None,statistic='count',bins=gridSize,
                range=[[-bound,bound],[-bound,bound]],expand_binnumbers=True)
            counts=binned.statistic.T.reshape(-1).astype(int)
            centers=[]
            for axis in range(2):
                sums=stats.binned_statistic_2d(initial[:,0],initial[:,1],initial[:,axis],statistic='sum',bins=gridSize,
                    range=[[-bound,bound],[-bound,bound]]).statistic.T.reshape(-1)
                centers.append(numpy.divide(sums,counts,out=numpy.zeros_like(sums),where=counts>0))
            cells=(binned.binnumber[1]-1)*gridSize+binned.binnumber[0]-1
            gridReference=dict(offsets=numpy.r_[0,numpy.cumsum(counts)].tolist(),
                vertexIds=numpy.argsort(cells,kind='stable').tolist(),centers=numpy.array(centers).T.reshape(-1).tolist())
            runs.append(dict(seed=run['seed'],gridReference=gridReference,initial=evaluate(initial),optimized=evaluate(expand(optimized.x)),
                networkx=evaluate(numpy.array([spring[node] for node in graph],dtype=float)),
                optimizer=dict(success=bool(optimized.success),message=str(optimized.message),iterations=int(optimized.nit))))
        value={**source,'case':'pinned-'+name,'runs':runs,'pinned':[int(node in pinned) for node in graph],
            'gridSize':[gridSize,gridSize],'bounds':[-bound,-bound,bound,bound],
            'reference':'networkx.spring_layout(fixed); scipy.optimize.minimize with fixed anchors; composite Noack energy',
            'conventions':__doc__,'generator':Path(__file__).name,'versions':energy.VERSIONS}
        if name == 'showcase':
            # 32*1*.045 plus initial |coordinate|<4 remains inside [-6,6].
            value['frames']=1
        path=output/f'parity-wp3-spatial-force-layout-{name}.json'
        path.write_text(json.dumps(value,separators=(',',':'),allow_nan=False)+'\n')
        assert path.stat().st_size<=1_500_000
        print(path.name,path.stat().st_size,[run['optimizer'] for run in runs],flush=True)

if __name__=='__main__': main()
