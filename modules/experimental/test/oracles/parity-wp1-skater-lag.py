# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""spopt Skater + libpysal lag_spatial on all Chicago tracts and synthetic forests.

Full 791 tract geometry from supplied f32 EPSG:4326 vertices. libpysal Queen
constructs shared-vertex adjacency; no projection needed for topology. All
105808 place assignments supply counts (unassigned 65535 excluded explicitly).
Lag uses three f32 columns: place counts, population, poverty150; raw and
row-standardized weights, and a deterministic selection mask removing BOTH
query and neighbor rows before restandardization. Islands are retained as zero.
Synthetic negative/zero weights, cancellation, masks and island included.

SKATER receives an explicit independent SciPy minimum spanning forest of squared
Euclidean dissimilarities of population-standardized f32 covariates. spopt Skater
is run on exactly that tree: this isolates the public tree-cut API from MST tie
selection. spopt defaults to Manhattan cost; override with squared Euclidean,
mean center and sum reduction to match documented within-region SSD. Islands
='ignore' means requested count includes original components. No RNG or seeds
(algorithm deterministic); compare canonical labels only on these same-tree
runs, independent per-region SSD and connectivity, and cut gains. Floor is row
count, not population. Exact partitions are an additional deterministic H gate;
objective quality ratio is reported even on mismatches, never a loosened bound.
https://pysal.org/spopt/generated/spopt.region.Skater.html
https://pysal.org/libpysal/generated/libpysal.weights.lag_spatial.html
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import warnings
import numpy as numpy
import pandas
import geopandas
import scipy
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components
from scipy.spatial.distance import cdist
import shapely
from shapely.geometry import Polygon
import libpysal
from libpysal.weights import W, Queen, lag_spatial
import spopt
from spopt.region import Skater
from spopt.region.skater import SpanningForest

ROOT=Path(__file__).resolve().parents[4]
DATA=ROOT/'examples/deck/spatial-analysis-showcase/public/data'


def makeWeights(matrix):
    matrix=matrix.tocsr()
    return W({row:matrix.indices[matrix.indptr[row]:matrix.indptr[row+1]].tolist() for row in range(matrix.shape[0])},
             {row:matrix.data[matrix.indptr[row]:matrix.indptr[row+1]].tolist() for row in range(matrix.shape[0])},silence_warnings=True)


def packMatrix(matrix):
    matrix=matrix.tocsr(); matrix.sort_indices()
    return {'offsets':matrix.indptr.tolist(),'neighbors':matrix.indices.tolist(),'weights':matrix.data.tolist()}


def canonicalize(labels):
    minimum={label:int(numpy.flatnonzero(labels==label)[0]) for label in numpy.unique(labels)}
    return [minimum[label] for label in labels]


def makeSkater(name, adjacency, values, targets, floor, provenance):
    values=numpy.asarray(values,dtype=numpy.float32).astype(float)
    source,target=adjacency.nonzero()
    costs=numpy.sum((values[source]-values[target])**2,axis=1)
    assert numpy.all(costs>0), 'spopt treats zero-cost links as absent'
    directedTree=minimum_spanning_tree(csr_matrix((costs,(source,target)),shape=adjacency.shape))
    tree=(directedTree+directedTree.T).tocsr(); tree.sort_indices()
    componentCount,components=connected_components(tree)
    componentLabels=canonicalize(components)
    binary=tree.copy(); binary.data[:]=1
    frame=geopandas.GeoDataFrame({f'value{column}':values[:,column] for column in range(values.shape[1])})
    runs=[]
    for targetCount in targets:
        model=Skater(frame, makeWeights(binary), list(frame.columns), n_clusters=targetCount,
                     floor=floor, islands='ignore', spanning_forest_kwds={
                         'dissimilarity':lambda first,second=None: cdist(first,first if second is None else second,'sqeuclidean'),
                         'center':numpy.mean,'reduction':numpy.sum})
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model.solve()
        # Retain the independent library's complete cut trace as well as the
        # public Skater wrapper result. No GPU algorithm is mirrored here.
        traced=SpanningForest(**model.spanning_forest_kwds)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            traced.fit(targetCount,makeWeights(binary),values,quorum=floor,trace=True,islands='ignore')
        assert canonicalize(traced.current_labels_)==canonicalize(model.labels_)
        traceScores=[float(traced.score(values,labels=labels)) for labels,_ in traced._trace]
        cutGains=(-numpy.diff(traceScores)).tolist()
        cutGains += [0.]*(targetCount-1-len(cutGains))
        labels=numpy.array(canonicalize(model.labels_))
        regionIds=numpy.unique(labels)
        sizes=[]; scores=[]
        for region in regionIds:
            members=numpy.flatnonzero(labels==region)
            assert connected_components(binary[members][:,members])[0]==1
            sizes.append(len(members)); scores.append(float(numpy.sum((values[members]-values[members].mean(axis=0))**2)))
        runs.append({'target':targetCount,'labels':labels.tolist(),'regionIds':regionIds.tolist(),'sizes':sizes,'scores':scores,'objective':sum(scores),'cutGains':cutGains})
    packed=packMatrix(binary)
    source=numpy.repeat(numpy.arange(len(values)),numpy.diff(binary.indptr))
    initialObjective=sum(float(numpy.sum((values[components==component]-values[components==component].mean(axis=0))**2)) for component in range(componentCount))
    return {'name':name,'reference':'spopt.region.Skater; scipy.sparse.csgraph.minimum_spanning_tree','n':len(values),'columns':values.shape[1],
            **packed,'values':values.ravel().tolist(),'treeFlags':(source<binary.indices).astype(int).tolist(),'components':componentLabels,
            'componentCount':int(componentCount),'initialObjective':initialObjective,'floor':floor,'runs':runs,'provenance':provenance,
            'inputFloat32Sha256':hashlib.sha256(values.astype('<f4').tobytes()).hexdigest()}


def makeLag(name, matrix, values, mask, normalize, provenance):
    values=numpy.asarray(values,dtype=numpy.float32).astype(float)
    matrix=matrix.astype(numpy.float32).astype(float).tocsr(); matrix.sort_indices()
    selected=matrix.multiply(numpy.asarray(mask)[:,None]).multiply(numpy.asarray(mask)[None,:]).tocsr()
    selected.eliminate_zeros()
    weights=makeWeights(selected)
    if normalize:
        # libpysal R cannot normalize a nonempty signed row with sum=0. The GPU
        # explicitly defines that output as zero: remove just those rows first.
        sums=numpy.asarray(selected.sum(axis=1)).ravel()
        selected=selected.multiply((sums!=0)[:,None]).tocsr(); selected.eliminate_zeros()
        weights=makeWeights(selected); weights.transform='R'
    expected=lag_spatial(weights,values)
    return {'name':name,'reference':'libpysal.weights.lag_spatial','n':len(values),'columns':values.shape[1],**packMatrix(matrix),
            'values':values.ravel().tolist(),'mask':mask,'normalize':normalize,'expected':expected.ravel().tolist(),'provenance':provenance,
            'inputFloat32Sha256':hashlib.sha256(values.astype('<f4').tobytes()).hexdigest()}


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--output-dir',type=Path,default=Path(__file__).parent/'fixtures'); arguments=parser.parse_args()
    directory=DATA/'chicago-tracts'
    vertices=numpy.fromfile(directory/'vertices.bin',dtype='<f4').reshape(-1,2)
    ringOffsets=numpy.fromfile(directory/'ringOffsets.bin',dtype='<u4')
    polygons=[Polygon(vertices[start:end]) for start,end in zip(ringOffsets[:-1],ringOffsets[1:])]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        adjacency=Queen.from_iterable(polygons).sparse
    assignments=numpy.fromfile(DATA/'chicago-places/tract.bin',dtype='<u2')
    counts=numpy.bincount(assignments[assignments<len(polygons)],minlength=len(polygons))
    population=numpy.fromfile(directory/'population.bin',dtype='<f4')
    poverty=numpy.fromfile(directory/'poverty150.bin',dtype='<f4')
    values=numpy.column_stack((counts,population,poverty))
    assert numpy.all(numpy.isfinite(values))
    files=['chicago-tracts/'+name for name in ['manifest.json','vertices.bin','ringOffsets.bin','population.bin','poverty150.bin']]+['chicago-places/manifest.json','chicago-places/tract.bin']
    provenance={'source':'chicago-tracts; chicago-places','sourceTracts':len(polygons),'sourcePlaces':len(assignments),'unassignedPlaces':int((assignments>=len(polygons)).sum()),'sourceSha256':{name:hashlib.sha256((DATA/name).read_bytes()).hexdigest() for name in files}}
    standardized=((values-values.mean(axis=0))/values.std(axis=0)).astype(numpy.float32)
    skater=[makeSkater('chicago-all-tracts',adjacency,standardized,[8,16],1,provenance)]
    # Distinct attributes on two paths and an isolated row; floor=1 preserves island.
    edges=[(0,1),(1,2),(2,3),(3,4),(5,6),(6,7),(7,8)]
    small=csr_matrix(([1]*(2*len(edges)),([row for pair in edges for row in pair],[row for pair in edges for row in pair[::-1]])),shape=(10,10))
    smallValues=numpy.array([[-4,1],[-3,0],[-2,1],[2,0],[5,2],[-4,-3],[-2,-2],[2,-3],[6,-2],[0,5]])/4
    skater.append(makeSkater('two-paths-island',small,smallValues,[3,5,7],1,{'source':'synthetic'}))
    path=csr_matrix(([1]*18,([row for index in range(9) for row in [index,index+1]],[row for index in range(9) for row in [index+1,index]])),shape=(10,10))
    skater.append(makeSkater('binding-quorum',path,smallValues,[2,4],3,{'source':'synthetic'}))
    lag=[]
    for masked in [False,True]:
        mask=[int(not masked or row%11!=0) for row in range(len(values))]
        for normalize in [False,True]:
            lag.append(makeLag(f'chicago-raw-mask-{masked}-normalize-{normalize}',adjacency,values,mask,normalize,provenance))
    small=csr_matrix(numpy.array([[0,1,-1,0,0],[.5,0,2,0,0],[0,0,0,0,0],[0,1,0,0,0],[0,0,0,0,0]]))
    for normalize in [False,True]:
        lag.append(makeLag(f'signed-islands-normalize-{normalize}',small,[[1,2],[3,4],[7,8],[9,10],[11,12]],[1,1,1,0,1],normalize,{'source':'synthetic'}))
    fixture={'generator':Path(__file__).name,'versions':{'python':platform.python_version(),'numpy':numpy.__version__,'scipy':scipy.__version__,'spopt':spopt.__version__,'libpysal':libpysal.__version__,'shapely':shapely.__version__,'geopandas':geopandas.__version__,'pandas':pandas.__version__},'skater':skater,'lag':lag}
    output=arguments.output_dir/'parity-wp1-skater-lag.json'; serialized=json.dumps(fixture,separators=(',',':'),allow_nan=False)+'\n'
    assert len(serialized)<1500000
    output.write_text(serialized)
    print(output,len(serialized),'bytes',[(case['name'],case['componentCount'],[(run['target'],len(run['sizes'])) for run in case['runs']]) for case in skater])

if __name__=='__main__': main()
