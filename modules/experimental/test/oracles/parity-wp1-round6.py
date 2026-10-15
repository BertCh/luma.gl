# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Independent sparse algebra, transpose and minimum-spanning-forest references.

Run reference Python -I; no random seeds. All 791 Chicago tract shells define
libpysal Queen adjacency. GEOS centroids projected EPSG:26916 define a second
4-neighbor scenario (scipy cKDTree); the topology is supplied to, not generated
by, the three tested GPU contributors. Queen weights are neighbor poverty150
rounded to quarters plus 1; kNN weights are inverse kilometers rounded to 1/1024.
All weights and attributes are rounded to f32 BEFORE float64 references.

Algebra: scipy sparse add/multiply/maximum and pattern arithmetic implement
set operations with explicit left/right/sum/min/max/product/binary rules.
Canonical sorted duplicate-free inputs are required by the public API; duplicate
slots are deliberately tested only for transpose. scipy csgraph.shortest_path
(unweighted, directed) supplies higher-order reachability. scipy diagonal
multiplication supplies subgraphs, identity supplies self weights, sparse group
indicators supply block contiguity. Empty rows and explicit zero-weight slots
retain structural membership. Overflow is the documented sorted CSR prefix.

Transpose: scipy csr.transpose().tocsr() preserves every duplicate stored slot,
including zero weights, and carries distances independently. Out-of-range
columns are removed by scipy sparse slicing. Full 105808 place->tract assignments
exercise a large rectangular graph; all source rows remain, including unassigned
tract IDs that the GPU must discard. Symmetry counts distinguish absent entries
from explicitly stored zeros. No distance or weight is inferred by the GPU.

MST: scipy.spatial.distance.cdist(metric='sqeuclidean') supplies all stored-slot
costs; scipy csgraph.minimum_spanning_tree supplies the optimum forest and
connected_components supplies minimum-row component labels. Standardization is
population ddof=0 over f32 inputs; constant columns become zero. SciPy treats
zero-cost edges as absent, so every stored edge cost is shifted by exactly 1
before solving. Every spanning forest has n-components edges; the same additive
shift preserves the minimizing forest, including genuine zero dissimilarities.
Costs reported in the fixture are the ORIGINAL costs. Exact edge IDs are required
only in the small distinct-cost case; tied/standardized cases compare feasibility,
every selected edge cost, full components, edge count and optimal total, not one
library's tie ordering. The attribute metric ignores supplied CSR weight values.

Sources: scipy sparse/csgraph docs; libpysal.weights.Queen; SciPy cdist; NumPy std.
"""
import argparse
import base64
import hashlib
import json
import platform
from pathlib import Path
import warnings
import zlib
import numpy as numpy
import scipy
from scipy.sparse import csr_matrix, diags, eye
from scipy.sparse.csgraph import shortest_path, connected_components, minimum_spanning_tree
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
import shapely
import pyproj
import libpysal
from libpysal.weights import Queen

ROOT = Path(__file__).resolve().parents[4]
DATA = ROOT / 'examples/deck/spatial-analysis-showcase/public/data'
VERSIONS = dict(python=platform.python_version(), numpy=numpy.__version__, scipy=scipy.__version__, shapely=shapely.__version__, pyproj=pyproj.__version__, libpysal=libpysal.__version__)


def rounded(values):
    return numpy.asarray(values, dtype=numpy.float32).astype(float)


def clean(value):
    if isinstance(value, numpy.ndarray): return value.tolist()
    if isinstance(value, numpy.generic): return value.item()
    if isinstance(value, dict): return {key: clean(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)): return [clean(item) for item in value]
    return value


def writeFixture(directory, topic, payload):
    raw = json.dumps(clean(payload), separators=(',', ':'), allow_nan=False).encode()
    wrapper = dict(generator=Path(__file__).name, versions=VERSIONS, encoding='zlib-json', sha256=hashlib.sha256(raw).hexdigest(), data=base64.b64encode(zlib.compress(raw, 9)).decode())
    serialized = json.dumps(wrapper, separators=(',', ':')) + '\n'
    assert len(serialized.encode()) <= 1500000
    (directory / f'parity-wp1-round6-{topic}.json').write_text(serialized)
    print(topic, len(serialized), 'bytes', flush=True)


def pack(matrix, distances=None):
    matrix = matrix.tocsr()
    return dict(offsets=matrix.indptr, neighbors=matrix.indices, weights=matrix.data,
                distances=[] if distances is None else distances.data)


def pattern(matrix):
    result = matrix.copy()
    result.data = numpy.ones(len(result.data))
    return result


def onPattern(values, support):
    # scipy arithmetic may drop explicit zeros. Restore the declared structural
    # support after arithmetic without inventing values or collapsing zero slots.
    rowIndices = numpy.repeat(numpy.arange(support.shape[0]), numpy.diff(support.indptr))
    data = numpy.asarray(values[rowIndices, support.indices]).ravel()
    return csr_matrix((data, support.indices.copy(), support.indptr.copy()), shape=support.shape)


def floatHash(*arrays):
    return hashlib.sha256(b''.join(numpy.asarray(array, dtype='<f4').tobytes() for array in arrays)).hexdigest()


def makeAlgebraDataset(name, left, right, groups, metadata):
    left.sort_indices(); right.sort_indices()
    assert left.has_canonical_format and right.has_canonical_format
    rows = left.shape[0]
    mask = numpy.arange(rows) % 5 != 0
    return dict(name=name, n=rows, left=pack(left), right=pack(right), mask=mask.astype(int), groups=groups,
        groupCount=int(max(groups))+1, provenance=metadata, inputFloat32Sha256=floatHash(left.data,right.data)), mask


def algebraCases(datasetIndex, dataset, left, right, mask):
    leftPattern, rightPattern = pattern(left), pattern(right)
    overlap = leftPattern.multiply(rightPattern)
    union = leftPattern.maximum(rightPattern)
    leftOnly, rightOnly = leftPattern - overlap, rightPattern - overlap
    cases = []
    def add(name, operation, result, **parameters):
        result = result.tocsr(); result.sort_indices()
        cases.append(dict(name=dataset['name']+'/'+name, reference='scipy.sparse arithmetic; scipy.sparse.csgraph.shortest_path',
            n=dataset['n'], datasetIndex=datasetIndex, operation=operation, **parameters,
            expected=pack(result), rowSums=numpy.asarray(result.sum(axis=1)).ravel()))
    for operation, support in [('union',union),('intersection',overlap),('difference',leftOnly),('symmetricDifference',leftOnly+rightOnly)]:
        rules = ['left','right','sum','min','max','product','binary'] if operation in ['union','intersection'] else ['left','binary']
        for rule in rules:
            if rule == 'binary': values = support
            elif rule == 'left': values = left + right.multiply(rightOnly)
            elif rule == 'right': values = right + left.multiply(leftOnly)
            else:
                both = {'sum':lambda:left+right, 'min':lambda:left.minimum(right), 'max':lambda:left.maximum(right), 'product':lambda:left.multiply(right)}[rule]()
                values = both.multiply(overlap) + left.multiply(leftOnly) + right.multiply(rightOnly)
            add(operation+'-'+rule,operation,onPattern(values,support),weightRule=rule)
    restriction = diags(mask.astype(float))
    add('subgraph','subgraph',onPattern(restriction@left@restriction,restriction@leftPattern@restriction))
    diagonal = left.copy(); diagonal.setdiag(0.5)
    add('self-weight','selfWeight',diagonal,selfWeight=0.5)
    distances = shortest_path(leftPattern,directed=True,unweighted=True)
    for order, cumulative in [(2,False),(3,False),(3,True)]:
        support = (distances > 0) & (distances <= order if cumulative else distances == order)
        add(f'order-{order}-cumulative-{cumulative}','higherOrder',csr_matrix(support.astype(float)),order=order,cumulative=cumulative,workCapacity=int(numpy.count_nonzero((distances > 0) & (distances <= order)))+5)
    groups = numpy.array(dataset['groups'])
    indicators = csr_matrix((numpy.ones(dataset['n']),(numpy.arange(dataset['n']),groups)),shape=(dataset['n'],dataset['groupCount']))
    block = indicators@indicators.T-eye(dataset['n'])
    block.eliminate_zeros()
    add('block','block',block)
    return cases


def makeTranspose(name, matrix, columns, metadata, distances=None, checkSymmetry=False):
    rows = matrix.shape[0]
    clipped = matrix[:, :columns]
    transposed = clipped.transpose().tocsr()
    transposedDistances = distances[:, :columns].transpose().tocsr() if distances is not None else None
    asymmetric = None
    if checkSymmetry:
        support = pattern(clipped)
        missing = (support-support.T) != 0
        unequal = (clipped-clipped.T) != 0
        asymmetric = int(missing.maximum(unequal).count_nonzero())
    return dict(name=name,reference='scipy.sparse.csr_matrix.transpose/tocsr; sparse slicing',n=rows,columnCount=columns,
        input=pack(matrix,distances),expected=pack(transposed,transposedDistances),asymmetricSlots=asymmetric,
        rowSums=numpy.asarray(transposed.sum(axis=1)).ravel(),provenance=metadata,
        inputFloat32Sha256=floatHash(matrix.data,[] if distances is None else distances.data))


def makeForest(name, adjacency, values, standardize, metadata, exactEdges=False):
    values=rounded(values)
    rows, columns=values.shape
    transformed=values.copy()
    if standardize:
        deviation=values.std(axis=0,ddof=0)
        transformed=(values-values.mean(axis=0))/numpy.where(deviation>0,deviation,1)
    rowIndices=numpy.repeat(numpy.arange(rows),numpy.diff(adjacency.indptr))
    costs=cdist(transformed,transformed,metric='sqeuclidean')[rowIndices,adjacency.indices]
    shifted=csr_matrix((costs+1,adjacency.indices,adjacency.indptr),shape=adjacency.shape)
    forest=minimum_spanning_tree(shifted)
    count, labels=connected_components(adjacency,directed=False)
    canonical=numpy.array([numpy.flatnonzero(labels==label).min() for label in labels])
    edgeRows,edgeColumns=forest.nonzero()
    edgePairs={(min(first,second),max(first,second)) for first,second in zip(edgeRows,edgeColumns)}
    flags=numpy.array([int(first<second and (first,second) in edgePairs) for first,second in zip(rowIndices,adjacency.indices)])
    if exactEdges:
        candidates=costs[rowIndices<adjacency.indices]
        assert len(numpy.unique(candidates))==len(candidates)
    assert len(edgePairs)==rows-count
    # Cross-check the shifted SciPy optimum with NetworkX only when zero edges
    # are present, to ensure the shift did not silently omit those stored links.
    if numpy.any(costs==0):
        import networkx
        VERSIONS['networkx']=networkx.__version__
        graph=networkx.Graph()
        graph.add_nodes_from(range(rows))
        graph.add_weighted_edges_from((int(first),int(second),float(cost)) for first,second,cost in zip(rowIndices,adjacency.indices,costs) if first<second)
        independent=networkx.minimum_spanning_tree(graph)
        assert abs(independent.size(weight='weight')-costs[flags.astype(bool)].sum())<1e-9
    return dict(name=name,reference='scipy.sparse.csgraph.minimum_spanning_tree/connected_components; scipy.spatial.distance.cdist',
        n=rows,input=pack(adjacency),values=values.ravel(),columnCount=columns,standardize=standardize,exactEdges=exactEdges,
        expected=dict(flags=flags,componentLabels=canonical,componentCount=count,edgeCount=rows-count,
            slotCosts=costs,totalCost=float(costs[flags.astype(bool)].sum()),standardizedValues=transformed.ravel()),
        inputFloat32Sha256=floatHash(adjacency.data,values),provenance=metadata)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output-dir',type=Path,default=Path(__file__).parent/'fixtures')
    directory=parser.parse_args().output_dir; directory.mkdir(parents=True,exist_ok=True)
    filenames=['chicago-tracts/'+name for name in ['manifest.json','vertices.bin','ringOffsets.bin','population.bin','poverty150.bin']]+['chicago-places/manifest.json','chicago-places/tract.bin']
    source=dict(sourceSha256={name:hashlib.sha256((DATA/name).read_bytes()).hexdigest() for name in filenames})
    positions=numpy.fromfile(DATA/'chicago-tracts/vertices.bin',dtype='<f4').reshape(-1,2)
    ringOffsets=numpy.fromfile(DATA/'chicago-tracts/ringOffsets.bin',dtype='<u4')
    polygons=[shapely.Polygon(positions[first:last]) for first,last in zip(ringOffsets[:-1],ringOffsets[1:])]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        adjacency=Queen.from_iterable(polygons).sparse.tocsr()
    rows=len(polygons)
    poverty=numpy.fromfile(DATA/'chicago-tracts/poverty150.bin',dtype='<f4').astype(float)
    population=numpy.fromfile(DATA/'chicago-tracts/population.bin',dtype='<f4').astype(float)
    assignments=numpy.fromfile(DATA/'chicago-places/tract.bin',dtype='<u2')
    counts=numpy.bincount(assignments[assignments<rows],minlength=rows)
    source.update(sourceTracts=rows,sourcePlaces=len(assignments),queenSlots=adjacency.nnz)
    centers=shapely.get_coordinates(shapely.centroid(polygons))
    transformer=pyproj.Transformer.from_crs('EPSG:4326','EPSG:26916',always_xy=True)
    projected=numpy.column_stack(transformer.transform(centers[:,0],centers[:,1]))
    neighborDistances,neighborIndices=cKDTree(projected).query(projected,k=5)
    right=csr_matrix((rounded(numpy.maximum(1,numpy.rint(1024/(1+neighborDistances[:,1:].ravel()/1000)))/1024),neighborIndices[:,1:].ravel(),numpy.arange(0,rows*4+1,4)),shape=(rows,rows))
    right.sort_indices()
    left=adjacency.copy(); left.data=rounded(1+numpy.rint(poverty[left.indices]*4)/4)
    synthetic=csr_matrix((numpy.array([0.,2.,1.,3.,.5,4.,2.]),numpy.array([1,3,0,2,0,1,3]),numpy.array([0,2,4,5,7,7,7])),shape=(6,6))
    other=csr_matrix((numpy.array([1.,4.,2.,0.,3.,2.]),numpy.array([1,2,0,3,1,4]),numpy.array([0,2,3,4,5,6,6])),shape=(6,6))
    datasets=[]; cases=[]
    for index,(name,first,second,groups,metadata) in enumerate([
        ('chicago-all-tracts',left,right,numpy.arange(rows)//20,source),
        ('zero-weights-islands-self',synthetic,other,numpy.array([0,0,1,1,2,3]),{'source':'synthetic'})]):
        dataset,mask=makeAlgebraDataset(name,first,second,groups,metadata)
        datasets.append(dataset); cases+=algebraCases(index,dataset,first,second,mask)
    writeFixture(directory,'algebra',dict(datasets=datasets,cases=cases))
    # Duplicates and unsorted rows are supported by transpose, not algebra.
    repeated=csr_matrix((numpy.array([2.,0.,3.,7.,4.,1.,6.]),numpy.array([2,0,2,8,1,1,0]),numpy.array([0,4,4,6,7,7])),shape=(5,9))
    repeatedDistances=repeated.copy(); repeatedDistances.data=numpy.arange(1,8)/4
    placeMatrix=csr_matrix((numpy.ones(len(assignments)),assignments,numpy.arange(len(assignments)+1)),shape=(len(assignments),65536))
    transposeCases=[makeTranspose('chicago-weighted-queen',left,rows,source,checkSymmetry=True),
        makeTranspose('chicago-directed-knn-extra-empty-columns',csr_matrix((right.data,right.indices,right.indptr),shape=(rows,rows+5)),rows+5,source),
        makeTranspose('all-places-to-tracts',placeMatrix,rows,source),
        makeTranspose('unsorted-duplicates-zero-invalid-islands',repeated,4,{'source':'synthetic'},repeatedDistances),
        makeTranspose('square-self-zero-absent',synthetic,6,{'source':'synthetic'},checkSymmetry=True)]
    writeFixture(directory,'transpose',dict(cases=transposeCases))
    small=csr_matrix(numpy.array([[0,1,1,1,0,0,0],[1,0,1,1,0,0,0],[1,1,0,1,0,0,0],[1,1,1,0,0,0,0],[0,0,0,0,0,1,0],[0,0,0,0,1,0,0],[0,0,0,0,0,0,0]],float))
    forests=[makeForest('chicago-raw-dyadic-attributes',adjacency,numpy.column_stack([counts/128,numpy.rint(poverty*8)/8]),False,source),
        makeForest('chicago-standardized-three-attributes',adjacency,numpy.column_stack([counts,population,poverty]),True,source),
        makeForest('distinct-cost-disconnected-island',small,numpy.array([0,1,4,10,22,47,100])[:,None],False,{'source':'synthetic'},True),
        makeForest('zero-cost-ties-constant-column',small,numpy.column_stack([[0,0,1,1,4,4,10],[3]*7]),True,{'source':'synthetic'})]
    writeFixture(directory,'forest',dict(cases=forests))
    print('case counts',len(cases),len(transposeCases),len(forests),flush=True)

if __name__=='__main__': main()
