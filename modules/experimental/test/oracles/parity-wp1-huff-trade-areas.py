# luma.gl
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
"""Independent Huff probabilities from scipy.special.softmax and sparse products.

Inputs are A (facility attractiveness) and w (already evaluated distance decay),
NOT raw travel costs: softmax(alpha*log(A)+log(w)) over each reachable row.
Zero attractiveness/weight has probability zero; an all-zero or empty row is
unserved, with modal ID UINT32_MAX and probability zero. Equal positive scores
select the lowest facility ID, with ascending unique CSR neighbor IDs (shared weights API invariant).
Expected demand is scipy.sparse probability-matrix transpose dot demand.
No access.fca reference is needed: GPUHuffTradeAreas has no FCA/accessibility
branch; those belong to other contributors. No RNG, permutation or missing data.

Real case: all 1,255 grocery facilities from Chicago places; 48 evenly spaced
source row IDs from the full 105,808 places. Demand is the full-data count in
that origin's tract (0 for unassigned); attractiveness is facility confidence
percent (a reproducible test covariate, not a capacity claim). pyproj EPSG:26916
kilometres, cutoff 5 km, cost floor .001 km explicitly resolves co-location;
w=1/max(distance,.001)^2 rounded to f32. Every admitted pair is retained.
The oracle operates in f64 on f32 uploads, including alpha=1.5. Profile S.
Synthetic cases test zero weight/attractiveness, empty/unserved rows, co-location
policy through bounded decay, and equal-score ties in valid ascending CSR rows.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform

import numpy as numpy
import pyproj
import scipy
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist
from scipy.special import softmax

ROOT = Path(__file__).resolve().parents[4]
DATA = ROOT/'examples/deck/spatial-analysis-showcase/public/data/chicago-places'


def makeCase(name, offsets, neighbors, weights, attractiveness, demand, exponent, provenance):
    weights = numpy.asarray(weights, dtype=numpy.float32).astype(float)
    attractiveness = numpy.asarray(attractiveness, dtype=numpy.float32).astype(float)
    demand = numpy.asarray(demand, dtype=numpy.float32).astype(float)
    exponent = float(numpy.float32(exponent))
    probabilities = numpy.zeros(len(weights))
    tradeArea = numpy.full(len(demand), 4294967295, dtype=numpy.uint32)
    tradeAreaProbability = numpy.zeros(len(demand))
    for row in range(len(demand)):
        slots = numpy.arange(offsets[row], offsets[row+1])
        facilities = numpy.array(neighbors, dtype=int)[slots]
        assert numpy.all(numpy.diff(facilities) > 0), facilities
        eligible = (weights[slots] > 0) & (attractiveness[facilities] > 0)
        valid = slots[eligible]
        if len(valid):
            scores = exponent*numpy.log(attractiveness[facilities[eligible]])+numpy.log(weights[valid])
            probabilities[valid] = softmax(scores)
            winners = numpy.array(neighbors)[valid[scores == scores.max()]]
            tradeArea[row] = winners.min()
            tradeAreaProbability[row] = probabilities[valid].max()
    probabilityMatrix = csr_matrix((probabilities, neighbors, offsets), shape=(len(demand), len(attractiveness)))
    transpose = csr_matrix((weights, neighbors, offsets), shape=probabilityMatrix.shape).T.tocsr()
    return {'name': name, 'reference': 'scipy.special.softmax; scipy.sparse.csr_matrix.T.dot',
            'n': len(demand), 'offsets': list(offsets), 'neighbors': list(neighbors), 'weights': weights.tolist(),
            'facilityOffsets': transpose.indptr.tolist(), 'facilityNeighbors': transpose.indices.tolist(),
            'facilityWeights': transpose.data.tolist(), 'attractiveness': attractiveness.tolist(),
            'demand': demand.tolist(), 'exponent': exponent,
            'expected': {'probabilities': probabilities.tolist(), 'tradeArea': tradeArea.tolist(),
                         'tradeAreaProbability': tradeAreaProbability.tolist(),
                         'expectedDemand': probabilityMatrix.T.dot(demand).tolist(),
                         'served': (tradeArea != 4294967295).astype(int).tolist()},
            'inputFloat32Sha256': {name: hashlib.sha256(numpy.asarray(values, dtype='<f4').tobytes()).hexdigest()
                                  for name, values in [('weights', weights), ('attractiveness', attractiveness), ('demand', demand)]},
            'provenance': provenance}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', dest='outputDirectory', type=Path, default=Path(__file__).parent/'fixtures')
    arguments = parser.parse_args()
    positions = numpy.fromfile(DATA/'position.bin', dtype='<f4').reshape(-1, 2)
    categories = numpy.fromfile(DATA/'category.bin', dtype='u1')
    confidence = numpy.fromfile(DATA/'confidence.bin', dtype='u1')
    tracts = numpy.fromfile(DATA/'tract.bin', dtype='<u2')
    facilityRows = numpy.flatnonzero(categories == 2)
    originRows = numpy.linspace(0, len(positions)-1, 48, dtype=int)
    transformer = pyproj.Transformer.from_crs('EPSG:4326', 'EPSG:26916', always_xy=True)
    projected = numpy.column_stack(transformer.transform(positions[:, 0], positions[:, 1]))/1000
    costs = cdist(projected[originRows], projected[facilityRows])
    matrix = numpy.where(costs <= 5, 1/numpy.maximum(costs, .001)**2, 0).astype(numpy.float32)
    sparse = csr_matrix(matrix)
    tractCounts = numpy.bincount(tracts, minlength=65536)
    tractCounts[65535] = 0
    provenance = {'source': str(DATA.relative_to(ROOT)), 'sourceCount': len(positions),
                  'sourceSha256': {name: hashlib.sha256((DATA/name).read_bytes()).hexdigest() for name in ['manifest.json', 'position.bin', 'category.bin', 'confidence.bin', 'tract.bin']},
                  'originRows': originRows.tolist(), 'facilityRows': facilityRows.tolist(),
                  'selection': '48 evenly spaced origins against ALL grocery facilities; closed 5 km catchment',
                  'crs': 'EPSG:26916', 'costUnits': 'kilometers', 'costFloor': .001,
                  'attribution': 'Overture Maps Foundation; CDLA-Permissive-2.0/Apache-2.0/CC0 by source'}
    cases = [makeCase('chicago-places-all-groceries', sparse.indptr.tolist(), sparse.indices.tolist(), sparse.data,
                      confidence[facilityRows], tractCounts[tracts[originRows]], 1.5, provenance)]
    for exponent in [1., 2.]:
        cases.append(makeCase(f'zeros-empty-ties-alpha-{exponent}',
                              [0, 3, 5, 5, 7, 10], [0, 1, 2, 0, 1, 0, 2, 0, 1, 2],
                              [1, 1, 1, 0, 0, 1000000, 1, 1, 4, 4],
                              [0, 2, 2], [10, 20, 30, 40, 50], exponent, {'source': 'synthetic'}))
    fixture = {'generator': Path(__file__).name,
               'versions': {'python': platform.python_version(), 'numpy': numpy.__version__, 'scipy': scipy.__version__, 'pyproj': pyproj.__version__},
               'cases': cases}
    output = arguments.outputDirectory/'parity-wp1-huff-trade-areas.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(fixture, separators=(',', ':'), allow_nan=False)+'\n'
    assert len(serialized.encode()) <= 1500000
    output.write_text(serialized)
    print(output, len(serialized), 'bytes;', len(facilityRows), 'facilities;', sparse.nnz, 'pairs')


if __name__ == '__main__':
    main()
