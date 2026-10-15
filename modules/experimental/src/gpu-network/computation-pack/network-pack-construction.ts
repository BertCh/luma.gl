// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUComputationFormulaRef} from '@luma.gl/gpgpu';
import type {NetworkPackPart} from './network-pack-part';

const CONSTRUCTION_FORMULA_REFS: Record<string, GPUComputationFormulaRef> = {
  'gpu-network.cost-table.admission': {
    id: 'gpu-network.cost-table.admission',
    derivation: 'curated',
    tex: String.raw`\term{isIncluded}{I}=\mathbf{1}\bigl[\operatorname{finite}(\term{cost}{c})\wedge \term{cost}{c}\ge 0\bigr]`,
    ascii:
      'Table admission, after the cutoff search of one source row: isIncluded = isFiniteBits(cost) && cost >= 0, where cost is the f32 reachability fixpoint searched with the cost limit. Unreached nodes are +Infinity and are not stored; the source itself is stored at cost 0. Included targets are packed in ascending node order by an inclusive scan, so slot order is fixed; entries past the capacity are dropped and reported by the overflow status. This is the admission rule, not a second shortest-path formula.',
    symbols: [
      {id: 'isIncluded', role: 'output', label: 'entry is stored', binding: 'wgsl:isIncluded'},
      {id: 'cost', role: 'input', label: 'network cost from the source', binding: 'wgsl:cost'}
    ]
  },
  'gpu-network.cost-weights.decay': {
    id: 'gpu-network.cost-weights.decay',
    derivation: 'curated',
    tex: String.raw`\term{weights}{w}=e^{-\term{beta}{\beta}\,\term{cost}{c}}\quad\text{for finite }0\le c\le L`,
    ascii:
      "Exponential decay, the branch chosen by the decay parameter: weights = exp(-beta * cost) for every finite matrix cost with 0 <= cost <= costLimit; other entries (unreached +Infinity, negative, beyond the limit) are not stored. Slots follow ascending neighbor order within each row. The 'none' branch stores 1, 'power' stores max(cost, minimumCost)^-beta and 'gaussian' stores the truncated, rescaled (exp(-0.5 (c/L)^2) - exp(-0.5)) / (1 - exp(-0.5)); they are alternative branches of the same kernel.",
    symbols: [
      {id: 'weights', role: 'output', label: 'stored decay weight', binding: 'wgsl:weights'},
      {id: 'beta', role: 'parameter', label: 'decay rate', binding: 'wgsl:beta'},
      {id: 'cost', role: 'input', label: 'matrix cost', binding: 'wgsl:cost'}
    ]
  },
  'gpu-network.k-function.k': {
    id: 'gpu-network.k-function.k',
    derivation: 'curated',
    tex: String.raw`\term{k}{K(d_k)}=2\,\term{cumulative}{P(d_k)}\,\term{networkLength}{L}\,/\,\term{count}{n}^{2}\qquad(n\ge 2,\ \text{else }0)`,
    ascii:
      'K(d_k) = 2 * cumulative * networkLength / (count * count) for count >= 2 snapped events, else 0. cumulative is the number of unordered event pairs whose network distance is strictly below the band threshold d_k = maximumDistance * k / (bandCount - 1) (a prefix sum of the per-band pair histogram); networkLength is the length of every road counted once. The denominator is count squared (the spaghetti GlobalAutoK normalisation), not count * (count - 1). With zero simulations only the observed row exists.',
    symbols: [
      {id: 'k', role: 'output', label: 'K value of the band', binding: 'wgsl:k'},
      {
        id: 'cumulative',
        role: 'intermediate',
        label: 'pairs closer than the threshold',
        binding: 'wgsl:cumulative'
      },
      {
        id: 'networkLength',
        role: 'parameter',
        label: 'total network length',
        binding: 'wgsl:networkLength'
      },
      {id: 'count', role: 'input', label: 'snapped event count', binding: 'wgsl:count'}
    ]
  },
  'gpu-network.flow-aggregation.pair-sum': {
    id: 'gpu-network.flow-aggregation.pair-sum',
    derivation: 'curated',
    tex: String.raw`\term{rowWeights}{W}=\sum_{x\in\text{pair}}\term{sortedContributions}{x}`,
    ascii:
      "rowWeights[row] = fixed_tree_sum(sortedContributions[begin:end]): one workgroup per stored origin-destination pair sums the pair's contributions (the weights of its records, in source row order, 0 for rejected or non-finite weights) with a strided partial sum and a fixed binary tree, so the float sum is bitwise reproducible. It is a sum of the weights (trips), not a mean and not the record count; flowCounts counts the records of the pair separately.",
    symbols: [
      {
        id: 'rowWeights',
        role: 'output',
        label: 'weight sum of the pair',
        binding: 'wgsl:rowWeights'
      },
      {
        id: 'sortedContributions',
        role: 'input',
        label: 'record weights of the pair, sorted',
        binding: 'wgsl:sortedContributions'
      }
    ]
  }
};

/** Network pack part: sparse costs, construction, inference and flows (batch C). */
export const NETWORK_CONSTRUCTION_PACK: NetworkPackPart = {
  formulaRefs: CONSTRUCTION_FORMULA_REFS,
  formulas: {
    GPUNetworkCostTable: CONSTRUCTION_FORMULA_REFS['gpu-network.cost-table.admission'],
    GPUNetworkCostWeights: CONSTRUCTION_FORMULA_REFS['gpu-network.cost-weights.decay'],
    GPUNetworkNoding: {
      formula: 'none',
      reason:
        'segment-intersection splitting, tolerance-cell grouping of end points and CSR packing are procedural; the graph rung shows split, snap, sort, edge and CSR passes. Tolerance is a grid cell, not a transitive Euclidean clustering'
    },
    GPUMapMatching: {
      formula: 'none',
      reason:
        'bounded A* route searches feeding a Viterbi decode with restarts and overflow flags are procedural; the graph rung shows candidates, routes, forward, backward and decode'
    },
    GPUNetworkKFunction: CONSTRUCTION_FORMULA_REFS['gpu-network.k-function.k'],
    GPUEdgeBundling: {
      formula: 'none',
      reason:
        'fixed-point density splatting, normalised gradient advection, arc-length resampling and smoothing form a procedural per-iteration update; the iterations rung shows the path states and the graph rung the repeated clear, splat and update passes'
    },
    GPUFlowAggregation: CONSTRUCTION_FORMULA_REFS['gpu-network.flow-aggregation.pair-sum']
  },
  captureStages: {
    tableOffsets: 'representation',
    tableTargets: 'search',
    tableCosts: 'search',
    rowCosts: 'search',
    weightOffsets: 'representation',
    weightRetained: 'impedance',
    weightValues: 'impedance',
    weightDistances: 'impedance',
    pieceSourceIds: 'representation',
    nodingNodeCount: 'summary',
    matchedEdges: 'assignment',
    matchSnapDistances: 'assignment',
    matchBreaks: 'summary',
    pairCounts: 'summary',
    kValues: 'summary',
    bundledPaths: 'presentation',
    flowWeights: 'summary',
    flowCounts: 'summary',
    flowOriginZones: 'assignment',
    flowDestinationZones: 'assignment'
  },
  variableKinds: {
    tableOffsets: {level: 'ratio', extent: 'extensive'},
    tableCosts: {level: 'ratio', extent: 'intensive'},
    rowCosts: {level: 'ratio', extent: 'intensive'},
    weightOffsets: {level: 'ratio', extent: 'extensive'},
    weightValues: {level: 'ratio', extent: 'intensive'},
    weightDistances: {level: 'ratio', extent: 'intensive'},
    pieceSourceIds: {level: 'nominal'},
    matchedEdges: {level: 'nominal'},
    matchSnapDistances: {level: 'ratio', extent: 'intensive'},
    matchBreaks: {level: 'nominal'},
    pairCounts: {level: 'ratio', extent: 'extensive'},
    kValues: {level: 'ratio', extent: 'intensive'},
    flowWeights: {level: 'ratio', extent: 'extensive'},
    flowCounts: {level: 'ratio', extent: 'extensive'},
    flowOriginZones: {level: 'nominal'},
    flowDestinationZones: {level: 'nominal'}
  }
};
