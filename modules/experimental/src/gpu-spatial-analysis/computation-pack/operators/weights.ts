// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {SpatialFamilyOperators} from '../types';

/** Spatial weights operators, with formulas scoped to the variants demonstrated by their cards. */
export const WEIGHTS_OPERATORS: SpatialFamilyOperators = {
  GPUContiguityWeights: {
    family: 'weights',
    descriptor: {scope: 'pairwise', reduction: 'scan-sort'},
    variables: {weightsValues: 'measure', neighborCounts: 'count'},
    stages: ['neighbourhood', 'operator'],
    datumKinds: [{kind: 'region', label: 'tract', indexSpace: 'csr'}],
    representation: {from: 'object', to: 'network'},
    guards: ['islands-disclosed'],
    choices: [
      {name: 'criterion', stage: 'neighbourhood', sweepable: true, alternatives: ['queen', 'rook']},
      {name: 'snapTolerance', stage: 'neighbourhood', sweepable: true}
    ],
    formula: {
      id: 'contiguity-weights',
      derivation: 'curated',
      tex: String.raw`\term{weight}{w_{ij}}=\mathbf{1}[i\ne j\ \land\ \term{keys}{K_i}\cap K_j\ne\varnothing]`,
      ascii:
        'Binary contiguity: w_ij = 1 iff i != j and K_i intersects K_j. Queen K contains snapped vertex keys; rook K contains undirected nondegenerate edge keys. All rings and parts participate; invalid vertices are ignored; only unique pairs within capacity are emitted.',
      symbols: [
        {id: 'weight', role: 'output', binding: 'wgsl:weights'},
        {id: 'keys', role: 'intermediate', binding: 'wgsl:groupId'}
      ]
    }
  },
  GPULatticeWeights: {
    family: 'weights',
    descriptor: {scope: 'focal', reduction: 'select'},
    variables: {weightsValues: 'measure', neighborCounts: 'count', elevation: 'measure'},
    stages: ['neighbourhood', 'operator'],
    datumKinds: [{kind: 'cell', label: 'raster cell', indexSpace: 'csr'}],
    representation: {from: 'field', to: 'network'},
    guards: ['islands-disclosed'],
    choices: [
      {name: 'criterion', stage: 'neighbourhood', sweepable: true, alternatives: ['rook', 'queen']},
      {name: 'radius', stage: 'neighbourhood', sweepable: true}
    ],
    formula: {
      id: 'lattice-weights',
      derivation: 'curated',
      tex: String.raw`\term{weight}{w_{ij}}=\mathbf{1}[0<|\term{horizontal}{d_x}|+|\term{vertical}{d_y}|\le\term{radius}{r}]`,
      ascii:
        'Rook: w_ij = 1 for 0 < abs(dx)+abs(dy) <= radius; queen uses 0 < max(abs(dx),abs(dy)) <= radius. Both cells must be in bounds and unmasked. Capacity overflow is reported.',
      symbols: [
        {id: 'weight', role: 'output', binding: 'wgsl:weights'},
        {id: 'horizontal', role: 'intermediate', binding: 'wgsl:dx'},
        {id: 'vertical', role: 'intermediate', binding: 'wgsl:dy'},
        {id: 'radius', role: 'parameter', binding: 'wgsl:RADIUS'}
      ]
    }
  },
  GPUSpatialWeightsTransform: {
    family: 'weights',
    descriptor: {scope: 'focal', reduction: 'sum'},
    variables: {weightsValues: 'measure', sourceWeightsValues: 'measure', neighborCounts: 'count'},
    stages: ['neighbourhood', 'operator'],
    datumKinds: [{kind: 'region', label: 'tract', indexSpace: 'csr'}],
    guards: ['row-standardisation', 'islands-disclosed'],
    choices: [
      {
        name: 'operation',
        stage: 'neighbourhood',
        sweepable: true,
        alternatives: ['row', 'binary', 'kernel', 'symmetrize', 'double', 'variance']
      }
    ],
    formula: {
      id: 'spatial-weights-transform',
      derivation: 'curated',
      tex: String.raw`\term{output}{w\prime_{ij}}=\term{input}{w_{ij}}\term{scale}{s_i},\quad\term{scale}{s_i}=\begin{cases}1/\term{sum}{S_i}&S_i>0\text{ and finite}\\0&\text{otherwise}\end{cases},\quad\term{sum}{S_i}=\sum_j\term{input}{w_{ij}}`,
      ascii:
        'Row variant: S_i = sum_j w_ij; scale = 1/S_i for positive finite S_i, otherwise 0; output = input * scale. The CSR pattern is unchanged. Binary, kernel, symmetrize, double and variance are separate variants.',
      symbols: [
        {id: 'output', role: 'output', binding: 'wgsl:values'},
        {id: 'input', role: 'input', binding: 'wgsl:source'},
        {id: 'scale', role: 'intermediate', binding: 'wgsl:scale'},
        {id: 'sum', role: 'intermediate', binding: 'wgsl:sum'}
      ]
    }
  },
  GPUSpatialLag: {
    family: 'weights',
    descriptor: {scope: 'focal', reduction: 'sum'},
    variables: {values: 'rate', lag: 'rate', weightsValues: 'measure', neighborCounts: 'count'},
    stages: ['neighbourhood', 'operator'],
    datumKinds: [{kind: 'region', label: 'tract', indexSpace: 'csr'}],
    guards: ['row-standardisation', 'islands-disclosed'],
    choices: [
      {name: 'normalize', stage: 'neighbourhood', sweepable: true, alternatives: [false, true]},
      {name: 'columnCount', stage: 'neighbourhood', sweepable: true}
    ],
    formula: {
      id: 'spatial-lag',
      derivation: 'curated',
      tex: String.raw`\term{lag}{[Wx]_i}=\sum_j\term{weight}{w_{ij}}\term{value}{x_j}`,
      ascii:
        'Without normalize: [Wx]_i = sum_j w_ij x_j over included neighbors in the value range; masked rows output zero. With normalize, divide by the included weight sum, returning zero when that sum is nonpositive. This example uses row weights and normalize=false.',
      symbols: [
        {id: 'lag', role: 'output', binding: 'wgsl:output'},
        {id: 'weight', role: 'input', binding: 'wgsl:weight'},
        {id: 'value', role: 'input', binding: 'wgsl:values'}
      ]
    }
  },
  GPUNeighborSearch: {
    family: 'weights',
    descriptor: {scope: 'pairwise', reduction: 'select'},
    variables: {weightsValues: 'measure', distances: 'measure', neighborCounts: 'count'},
    stages: ['neighbourhood', 'operator'],
    datumKinds: [{kind: 'region', label: 'tract', indexSpace: 'csr'}],
    representation: {from: 'object', to: 'network'},
    guards: ['coordinate-model', 'islands-disclosed'],
    choices: [
      {name: 'mode', stage: 'neighbourhood', sweepable: true, alternatives: ['knn', 'radius']},
      {name: 'k', stage: 'neighbourhood', sweepable: true},
      {
        name: 'weightKind',
        stage: 'neighbourhood',
        sweepable: true,
        alternatives: ['binary', 'inverse-distance', 'kernel']
      },
      {name: 'rowStandardize', stage: 'neighbourhood', sweepable: true, alternatives: [false, true]}
    ],
    formula: {
      id: 'neighbor-search',
      derivation: 'curated',
      tex: String.raw`\term{weight}{w_{ij}}=\mathbf{1}[j\in\operatorname{first}_{\term{count}{k}}\operatorname{sort}_{(\term{distance}{d_{ij}},j)}\{j:j\ne i\}]`,
      ascii:
        'Binary k-NN self join: retain the k smallest (distance, target ID) pairs among valid unmasked targets within bounds and any finite radius cap, excluding self. This example uses great-circle distance in meters from longitude/latitude degrees, binary weights, and no row standardization; CSR IDs are sorted after selection.',
      symbols: [
        {id: 'weight', role: 'output', binding: 'wgsl:weight'},
        {id: 'count', role: 'parameter', binding: 'wgsl:K'},
        {id: 'distance', role: 'intermediate', binding: 'wgsl:distance'}
      ]
    }
  },
  GPUSpatialWeightsAlgebra: {
    family: 'weights',
    descriptor: {scope: 'focal', reduction: 'scan-sort'},
    variables: {weightsValues: 'measure', sourceWeightsValues: 'measure', neighborCounts: 'count'},
    stages: ['neighbourhood', 'operator'],
    datumKinds: [{kind: 'region', label: 'tract', indexSpace: 'csr'}],
    guards: ['islands-disclosed'],
    choices: [
      {
        name: 'operation',
        stage: 'neighbourhood',
        sweepable: true,
        alternatives: [
          'union',
          'intersection',
          'difference',
          'symmetricDifference',
          'higherOrder',
          'selfWeight',
          'subgraph',
          'block'
        ]
      },
      {name: 'order', stage: 'neighbourhood', sweepable: true},
      {name: 'cumulative', stage: 'neighbourhood', sweepable: true, alternatives: [false, true]}
    ],
    formula: {
      formula: 'none',
      reason:
        'Higher-order weights expand sparse frontiers, remove already reached rows and self links, then sort and compact the requested shortest-path shell.'
    }
  },
  GPUSpatialWeightsSummary: {
    family: 'weights',
    descriptor: {scope: 'global', reduction: 'sum'},
    variables: {statistics: 'measure', counts: 'count', neighborCounts: 'count'},
    stages: ['neighbourhood', 'operator'],
    datumKinds: [{kind: 'region', label: 'tract', indexSpace: 'csr'}],
    guards: ['islands-disclosed'],
    choices: [{name: 'symmetryTolerance', stage: 'neighbourhood', sweepable: true}],
    formula: {
      id: 'spatial-weights-summary',
      derivation: 'curated',
      tex: String.raw`\term{sum}{S_0}=\sum_{ij}\term{weight}{w_{ij}},\quad\term{pairs}{S_1}=\frac12\sum_{ij}(\term{weight}{w_{ij}}+\term{reverse}{w_{ji}})^2,\quad\term{rows}{S_2}=\sum_i(\term{row}{r_i}+\term{column}{c_i})^2`,
      ascii:
        'S0 = sum_ij w_ij; S1 = (1/2) sum_ij (w_ij+w_ji)^2 over the union of both patterns, with absent reverse weights zero; S2 = sum_i (rowSum_i+columnSum_i)^2. Self slots enter moments. Cardinality and isolates instead count only positive, in-range, nonself slots. Fixed-order reductions.',
      symbols: [
        {id: 'sum', role: 'output', binding: 'wgsl:s0'},
        {id: 'pairs', role: 'output', binding: 'wgsl:s1'},
        {id: 'rows', role: 'output', binding: 'wgsl:s2'},
        {id: 'weight', role: 'input', binding: 'wgsl:weight'},
        {id: 'reverse', role: 'intermediate', binding: 'wgsl:reverse'},
        {id: 'row', role: 'intermediate', binding: 'wgsl:rowSums'},
        {id: 'column', role: 'intermediate', binding: 'wgsl:columnSums'}
      ]
    }
  },
  GPUSpatialWeightsTranspose: {
    family: 'weights',
    descriptor: {scope: 'global', reduction: 'scan-sort'},
    variables: {weightsValues: 'measure', sourceWeightsValues: 'measure', neighborCounts: 'count'},
    stages: ['neighbourhood', 'operator'],
    datumKinds: [{kind: 'region', label: 'tract', indexSpace: 'csr'}],
    guards: ['islands-disclosed'],
    choices: [
      {name: 'columnCount', stage: 'neighbourhood', sweepable: true},
      {name: 'symmetryTolerance', stage: 'neighbourhood', sweepable: true}
    ],
    formula: {
      formula: 'none',
      reason:
        'Transposition stable-sorts slots by target column, locates CSR row boundaries by binary search and gathers reversed links.'
    }
  },
  GPUSpatialWeightsValidation: {
    family: 'weights',
    descriptor: {scope: 'focal', reduction: 'count'},
    variables: {flags: 'category', invalidCount: 'count', neighborCounts: 'count'},
    stages: ['neighbourhood', 'operator'],
    datumKinds: [{kind: 'region', label: 'tract', indexSpace: 'csr'}],
    guards: ['islands-disclosed'],
    choices: [],
    formula: {
      formula: 'none',
      reason:
        'Validation checks CSR offsets and each row’s neighbor order, range, diagonal and weights, then counts rows with violation flags.'
    }
  },
  GPUMapColoring: {
    family: 'weights',
    descriptor: {scope: 'frontier', reduction: 'select'},
    variables: {
      colors: 'category',
      colorCount: 'count',
      conflictCount: 'count',
      iterationCount: 'count',
      converged: 'category'
    },
    stages: ['neighbourhood', 'operator'],
    datumKinds: [{kind: 'region', label: 'tract', indexSpace: 'csr'}],
    guards: [],
    choices: [
      {
        name: 'priority',
        stage: 'presentation',
        sweepable: true,
        alternatives: ['largest-degree-first', 'random']
      },
      {name: 'maximumIterations', stage: 'presentation', sweepable: true},
      {name: 'seed', stage: 'presentation', sweepable: true}
    ],
    formula: {
      formula: 'none',
      reason:
        'Jones–Plassmann rounds greedily color independent priority maxima, using largest degree first with seeded-hash and row-ID tie breaks.'
    }
  }
};
