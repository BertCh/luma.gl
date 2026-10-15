// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {SpatialFamilyOperators} from '../types';

/**
 * Spatial computation-pack entries for the `accessibility` family, keyed by exported name. Every entry
 * must use `family: 'accessibility'`.
 */
export const ACCESSIBILITY_OPERATORS: SpatialFamilyOperators = {
  GPUCatchmentAccessibility: {
    family: 'accessibility',
    descriptor: {scope: 'focal', reduction: 'sum'},
    variables: {supply: 'count', demand: 'count', ratios: 'rate', accessibility: 'rate'},
    stages: ['support', 'neighbourhood', 'operator'],
    datumKinds: [{kind: 'demandLocation', label: 'demand location', indexSpace: '1d'}],
    guards: ['coordinate-model'],
    choices: [
      {stage: 'operator', name: 'method', alternatives: ['2sfca', '3sfca'], sweepable: false},
      {
        stage: 'neighbourhood',
        name: 'radius',
        label: 'catchment radius',
        unit: 'm',
        sweepable: true
      },
      {
        stage: 'neighbourhood',
        name: 'weightKind',
        alternatives: ['binary', 'inverseDistance', 'kernel'],
        sweepable: false
      }
    ],
    // gpu-catchment-accessibility.ts `ratios` and `scores` kernels, method '2sfca' (3SFCA squares the
    // weights and divides by the selection total T_i).
    formula: {
      id: 'gpu-spatial-analysis.catchment-accessibility',
      tex: String.raw`\term{accessibility}{A_i}=\sum_{j}\term{weight}{w_{ij}}\,\term{ratio}{R_j},\qquad \term{ratio}{R_j}=\frac{\term{supply}{S_j}}{\sum_{i}\term{demand}{P_i}\,\term{weight}{w_{ji}}}`,
      ascii:
        'R_j = S_j / sum_i P_i w_ji over the demand row of facility j (0 when the sum is 0); ' +
        'A_i = sum_j w_ij R_j over the facility row of demand i. Weights are 1 inside the catchment for a binary band.',
      symbols: [
        {id: 'accessibility', role: 'output', label: 'A_i', binding: 'wgsl:accessibility'},
        {id: 'ratio', role: 'intermediate', label: 'R_j', binding: 'wgsl:ratios'},
        {id: 'supply', role: 'input', label: 'S_j', binding: 'wgsl:supply'},
        {id: 'demand', role: 'input', label: 'P_i', binding: 'wgsl:demand'},
        {id: 'weight', role: 'parameter', label: 'w', binding: 'wgsl:weights'}
      ],
      derivation: 'curated'
    },
    metadata: {coordinateModel: 'projected', metricUnit: 'm'}
  },
  GPUHuffTradeAreas: {
    family: 'accessibility',
    descriptor: {scope: 'focal', reduction: 'weighted-mean'},
    variables: {
      attractiveness: 'measure',
      probabilities: 'probability',
      tradeAreaProbability: 'probability',
      tradeArea: 'category',
      expectedDemand: 'count'
    },
    stages: ['support', 'neighbourhood', 'operator'],
    datumKinds: [{kind: 'demandLocation', label: 'demand location', indexSpace: '1d'}],
    guards: ['coordinate-model'],
    choices: [
      {
        stage: 'neighbourhood',
        name: 'distanceDecay',
        alternatives: ['binary', 'inverse-distance', 'kernel'],
        sweepable: false
      },
      {stage: 'neighbourhood', name: 'power', label: 'distance exponent', sweepable: true},
      {stage: 'neighbourhood', name: 'radius', label: 'cut-off radius', unit: 'm', sweepable: true},
      {stage: 'operator', name: 'exponent', label: 'attractiveness exponent', sweepable: true}
    ],
    // gpu-huff-trade-areas.ts `trade-area` kernel (getAttraction, denominators) and `probabilities` kernel.
    formula: {
      id: 'gpu-spatial-analysis.huff-trade-areas',
      tex: String.raw`\term{probability}{p_{ij}}=\frac{\term{attractiveness}{A_j}^{\term{exponent}{\alpha}}\,\term{weight}{w_{ij}}}{\term{denominator}{\sum_k A_k^{\alpha}\,w_{ik}}}`,
      ascii:
        'p_ij = A_j^alpha w_ij / sum_k A_k^alpha w_ik over the facility row of demand i; zero attractiveness gives 0 and ' +
        'a row with denominator 0 gives 0. The decay and cut-off are carried by the weights w_ij.',
      symbols: [
        {id: 'probability', role: 'output', label: 'p_ij', binding: 'wgsl:probabilities'},
        {id: 'attractiveness', role: 'input', label: 'A_j', binding: 'wgsl:attractiveness'},
        {id: 'exponent', role: 'parameter', label: 'alpha', binding: 'wgsl:exponent'},
        {id: 'weight', role: 'parameter', label: 'w_ij', binding: 'wgsl:weights'},
        {id: 'denominator', role: 'intermediate', label: 'row total', binding: 'wgsl:denominators'}
      ],
      derivation: 'curated'
    },
    metadata: {coordinateModel: 'projected', metricUnit: 'm'}
  },
  GPUArealInterpolation: {
    family: 'accessibility',
    descriptor: {scope: 'zonal', reduction: 'sum'},
    variables: {population: 'count', transferredFractions: 'rate', areaPopulation: 'count'},
    stages: ['representation', 'support', 'operator'],
    datumKinds: [{kind: 'targetZone', label: 'target zone', indexSpace: '1d'}],
    guards: ['coordinate-model', 'conservation'],
    choices: [
      {stage: 'operator', name: 'mode', alternatives: ['extensive', 'intensive'], sweepable: false},
      {
        stage: 'operator',
        name: 'denominator',
        alternatives: ['zone', 'overlap'],
        sweepable: false
      },
      {stage: 'support', name: 'cellSize', label: 'raster cell size', unit: 'm', sweepable: true}
    ],
    // gpu-areal-interpolation.ts `weights` kernel, mode 'extensive' with the 'zone' denominator; the
    // transfer sum over s is a GPUSpatialLag over the weights.
    formula: {
      id: 'gpu-spatial-analysis.areal-interpolation',
      tex: String.raw`\term{value}{y_t}=\sum_{s}\term{weight}{w_{ts}}\,\term{source}{x_s},\qquad \term{weight}{w_{ts}}=\frac{\term{overlap}{a_{st}}}{\term{sourceArea}{A_s}}`,
      ascii:
        'y_t = sum_s w_ts x_s, w_ts = a_st / A_s: a_st is the rasterized overlap of source s and target t (cell count), ' +
        'A_s the whole raster area of source s. Mass of a source outside every target is lost.',
      symbols: [
        {id: 'value', role: 'output', label: 'y_t'},
        {id: 'source', role: 'input', label: 'x_s'},
        {id: 'weight', role: 'intermediate', label: 'w_ts', binding: 'wgsl:weights'},
        {id: 'overlap', role: 'intermediate', label: 'a_st', binding: 'wgsl:areas'},
        {id: 'sourceArea', role: 'intermediate', label: 'A_s', binding: 'wgsl:sourceTotals'}
      ],
      derivation: 'curated'
    },
    metadata: {coordinateModel: 'projected', metricUnit: 'm', mappedVariableExtent: 'extensive'}
  },
  GPUPycnophylactic: {
    family: 'accessibility',
    descriptor: {scope: 'solver', reduction: 'solve'},
    variables: {population: 'count', mass: 'count'},
    stages: ['representation', 'support', 'operator'],
    datumKinds: [{kind: 'cell', label: 'raster cell', indexSpace: '2d'}],
    representation: {from: 'object', to: 'field'},
    guards: ['coordinate-model', 'conservation'],
    choices: [
      {stage: 'neighbourhood', name: 'kernel', alternatives: ['rook', 'box'], sweepable: false},
      {stage: 'operator', name: 'maximumIterations', sweepable: true},
      {stage: 'operator', name: 'tolerance', sweepable: true},
      {stage: 'support', name: 'cellSize', label: 'raster cell size', unit: 'm', sweepable: true}
    ],
    // gpu-pycnophylactic.ts per-iteration kernels: smooth, restore-and-clamp, rescale (planar raster,
    // cell area 1, so mass equals density).
    formula: {
      id: 'gpu-spatial-analysis.pycnophylactic',
      tex: String.raw`\term{density}{d_c^{(n+1)}}=\frac{T_z}{M'_z}\,\max\!\Big(0,\ \term{smooth}{\bar d_c^{(n)}}+\frac{\term{total}{T_z}-\term{mass}{M_z}}{\term{area}{A_z}}\Big)`,
      ascii:
        "d_c(n+1) = (T_z / M'_z) * max(0, mean_c(d(n)) + (T_z - M_z) / A_z) for every cell c of zone z. mean_c is the focal mean of " +
        "in-area neighbours (rook or box), M_z the zone mass of the smoothed surface, M'_z the zone mass after the clamp, " +
        'A_z the zone cell count. Initial density is T_z / A_z.',
      symbols: [
        {id: 'density', role: 'output', label: 'd_c', binding: 'wgsl:destination'},
        {id: 'smooth', role: 'intermediate', label: 'focal mean', binding: 'wgsl:source'},
        {id: 'total', role: 'input', label: 'T_z', binding: 'wgsl:totals'},
        {id: 'mass', role: 'intermediate', label: 'M_z', binding: 'wgsl:zoneSums'},
        {id: 'area', role: 'intermediate', label: 'A_z', binding: 'wgsl:zoneAreas'}
      ],
      derivation: 'curated'
    },
    metadata: {coordinateModel: 'projected', metricUnit: 'm', mappedVariableExtent: 'extensive'}
  },
  GPUSegregation: {
    family: 'accessibility',
    descriptor: {scope: 'global', reduction: 'sum'},
    variables: {groupCounts: 'count', localTheil: 'rate', localEntropy: 'measure', indices: 'rate'},
    stages: ['representation', 'neighbourhood', 'operator'],
    datumKinds: [{kind: 'unit', label: 'census tract', indexSpace: '1d'}],
    guards: [],
    choices: [
      {stage: 'operator', name: 'groupCount', sweepable: false},
      {
        stage: 'neighbourhood',
        name: 'spatialScales',
        alternatives: ['aspatial', 'kernel bandwidths'],
        sweepable: false
      }
    ],
    // gpu-segregation.ts `local` kernel (localTheil) and the finalize kernel (indices[entropy]); aspatial scale.
    formula: {
      id: 'gpu-spatial-analysis.segregation',
      tex: String.raw`\term{theil}{h_i}=\frac{\term{population}{t_i}\,(\term{diversity}{E}-\term{entropy}{E_i})}{E\,\term{total}{T}},\qquad H=\sum_i h_i`,
      ascii:
        'h_i = t_i (E - E_i) / (E T), H = sum_i h_i. E = sum_m P_m ln(1 / P_m) is the diversity of all residents, ' +
        'E_i the same for tract i, t_i its population, T the total. Undefined (NaN) when E = 0 or the total is 0.',
      symbols: [
        {id: 'theil', role: 'output', label: 'h_i', binding: 'wgsl:localTheil'},
        {id: 'population', role: 'input', label: 't_i', binding: 'wgsl:unitPopulation'},
        {id: 'diversity', role: 'intermediate', label: 'E', binding: 'wgsl:diversity'},
        {id: 'entropy', role: 'intermediate', label: 'E_i', binding: 'wgsl:entropy'},
        {id: 'total', role: 'intermediate', label: 'T', binding: 'wgsl:populationTotal'}
      ],
      derivation: 'curated'
    }
  }
};
