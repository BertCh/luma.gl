// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {SpatialFamilyOperators} from '../types';

/**
 * Spatial computation-pack entries for the `density` family, keyed by exported name. Every entry
 * must use `family: 'density'`.
 */
export const DENSITY_OPERATORS: SpatialFamilyOperators = {
  GPUKernelDensity: {
    family: 'density',
    descriptor: {scope: 'focal', reduction: 'weighted-mean'},
    variables: {positions: 'measure', weights: 'count', density: 'rate'},
    stages: ['representation', 'neighbourhood', 'operator'],
    datumKinds: [{kind: 'cell', label: 'output cell', indexSpace: '2d'}],
    representation: {from: 'event', to: 'field'},
    guards: ['coordinate-model'],
    choices: [
      {stage: 'neighbourhood', name: 'bandwidth', label: 'bandwidth h', unit: 'm', sweepable: true},
      {
        stage: 'neighbourhood',
        name: 'kernel',
        alternatives: ['gaussian', 'epanechnikov', 'quartic'],
        sweepable: false
      },
      {
        stage: 'support',
        name: 'edgeCorrection',
        alternatives: ['uniform', 'diggle', 'none'],
        sweepable: false
      },
      {stage: 'support', name: 'cellSize', label: 'cell size', unit: 'm', sweepable: true}
    ],
    // Exact gather, gaussian kernel truncated at 5h, 'uniform' edge correction, 'intensity'
    // normalization: kernel-density-wgsl.ts getKernelValue / getWindowMass / getKernelDensityResultWGSL.
    formula: {
      id: 'gpu-spatial-analysis.kernel-density',
      tex: String.raw`\term{density}{\hat\lambda(u)} = \frac{1}{\term{mass}{e_h(u)}} \sum_{i\,:\,\lVert u - x_i \rVert < 5h} \term{weight}{w_i}\, \term{kernel}{K_h}\big(\lVert u - \term{position}{x_i} \rVert\big), \qquad K_h(d) = \frac{e^{-d^2 / (2\term{bandwidth}{h}^2)}}{2\pi h^2}`,
      ascii:
        'density(u) = (1 / e_h(u)) * sum over points i with |u - x_i| < 5h of w_i * K_h(|u - x_i|), ' +
        'K_h(d) = exp(-d^2 / (2 h^2)) / (2 pi h^2), e_h(u) = kernel mass of cell centre u inside the window',
      symbols: [
        {id: 'density', role: 'output', label: 'λ̂(u)', binding: 'wgsl:valuesOut'},
        {id: 'position', role: 'input', label: 'x_i', binding: 'wgsl:positions'},
        {id: 'weight', role: 'input', label: 'w_i', binding: 'wgsl:pointWeights'},
        {id: 'kernel', role: 'intermediate', label: 'K_h', binding: 'wgsl:getKernelValue'},
        {id: 'mass', role: 'intermediate', label: 'e_h(u)', binding: 'wgsl:getWindowMass'},
        {id: 'bandwidth', role: 'parameter', label: 'h', binding: 'wgsl:bandwidth'}
      ],
      derivation: 'curated'
    },
    metadata: {coordinateModel: 'projected', metricUnit: 'm'}
  },
  GPUPointDensity: {
    family: 'density',
    descriptor: {scope: 'block', reduction: 'count'},
    variables: {observations: 'measure', counts: 'count'},
    stages: ['representation', 'support', 'operator'],
    datumKinds: [{kind: 'cell', label: 'count cell', indexSpace: '2d'}],
    representation: {from: 'event', to: 'field'},
    guards: ['coordinate-model', 'count-vs-rate'],
    choices: [
      {stage: 'support', name: 'gridSize', sweepable: false},
      {stage: 'support', name: 'binning', alternatives: ['grid', 'hexagon'], sweepable: false},
      {
        stage: 'operator',
        name: 'statistic',
        alternatives: ['count', 'sum', 'mean'],
        sweepable: false
      },
      {
        stage: 'operator',
        name: 'sumAccumulation',
        alternatives: ['auto', 'workgroup', 'atomic'],
        sweepable: false
      }
    ],
    // gpu-point-density.ts grid aggregation and point-density-kernels.ts createPointDensityFinalizeNode, count mode.
    formula: {
      id: 'gpu-spatial-analysis.point-density',
      tex: String.raw`\term{field}{v_c}=\term{count}{n_c}=\sum_{\term{point}{i}\,\in\,\term{cell}{C_c}}1`,
      ascii:
        'v_c = n_c = number of accepted points in cell C_c; inclusive domain maxima map to the last cell. No smoothing or division by area.',
      symbols: [
        {id: 'field', role: 'output', label: 'v_c', binding: 'wgsl:values'},
        {id: 'count', role: 'intermediate', label: 'n_c', binding: 'wgsl:counts'},
        {id: 'point', role: 'input', label: 'point i', binding: 'wgsl:positions'},
        {id: 'cell', role: 'parameter', label: 'cell C_c', binding: 'wgsl:gridSize'}
      ],
      derivation: 'curated'
    },
    metadata: {coordinateModel: 'projected', metricUnit: 'm', mappedVariableExtent: 'extensive'}
  },
  GPUStudyWindowPreparation: {
    family: 'density',
    descriptor: {scope: 'global', reduction: 'none'},
    variables: {preparedWindow: 'measure', summary: 'measure'},
    stages: ['representation', 'support'],
    datumKinds: [{kind: 'summaryField', label: 'window summary field', indexSpace: '1d'}],
    guards: ['coordinate-model'],
    choices: [{stage: 'support', name: 'samplingGrid', sweepable: false}],
    formula: {
      formula: 'none',
      reason:
        'Preparation orients polygon edges, reduces a window summary, builds edge slabs and compacts sampling cells.'
    },
    metadata: {coordinateModel: 'projected', metricUnit: 'm'}
  },
  GPUInverseDistanceWeighting: {
    family: 'density',
    descriptor: {scope: 'focal', reduction: 'weighted-mean'},
    variables: {stations: 'measure', precipitation: 'rate', rainfall: 'rate', counts: 'count'},
    stages: ['support', 'neighbourhood', 'operator'],
    datumKinds: [{kind: 'cell', label: 'rainfall cell', indexSpace: '2d'}],
    representation: {from: 'event', to: 'field'},
    guards: ['coordinate-model'],
    choices: [
      {stage: 'neighbourhood', name: 'searchRadius', unit: 'm', sweepable: true},
      {stage: 'neighbourhood', name: 'neighborCount', sweepable: true},
      {stage: 'neighbourhood', name: 'minimumNeighborCount', sweepable: true},
      {stage: 'operator', name: 'power', sweepable: true},
      {
        stage: 'operator',
        name: 'taper',
        alternatives: ['none', 'modified-shepard'],
        sweepable: false
      }
    ],
    // gpu-inverse-distance-weighting.ts addWeightedSample/getGatherBody, taper='none', nearest-k path.
    formula: {
      id: 'gpu-spatial-analysis.inverse-distance-weighting',
      tex: String.raw`\term{result}{\hat z(u)}=\frac{\sum_{i\in\term{neighbors}{N(u)}}\term{sample}{z_i}\,\term{distance}{d_i}^{-\term{power}{p}}}{\sum_{i\in\term{neighbors}{N(u)}}\term{distance}{d_i}^{-\term{power}{p}}}`,
      ascii:
        'z(u) = sum(z_i / d_i^p) / sum(1 / d_i^p), taper=none. N contains up to k nearest unmasked non-NaN samples within inclusive radius R and index bounds, ordered by (distance², row). An exact hit returns the smallest-row sample; without an exact hit, fewer than max(minimumNeighborCount, 1) contributors gives NaN. Log-space accumulation rescales weights without changing the normalized mean.',
      symbols: [
        {id: 'result', role: 'output', label: 'predicted rainfall', binding: 'wgsl:valuesOut'},
        {id: 'sample', role: 'input', label: 'station rainfall', binding: 'wgsl:sampleValues'},
        {
          id: 'distance',
          role: 'intermediate',
          label: 'distance to cell centre',
          binding: 'wgsl:distanceSquared'
        },
        {id: 'power', role: 'parameter', label: 'distance power', binding: 'wgsl:power'},
        {
          id: 'neighbors',
          role: 'intermediate',
          label: 'nearest samples within radius',
          binding: 'wgsl:nearRows'
        }
      ],
      derivation: 'curated'
    },
    metadata: {coordinateModel: 'projected', metricUnit: 'm', mappedVariableExtent: 'intensive'}
  },
  GPUKriging: {
    family: 'density',
    descriptor: {scope: 'solver', scopeSecondary: 'focal', reduction: 'solve'},
    variables: {
      stations: 'measure',
      precipitation: 'rate',
      rainfall: 'rate',
      variance: 'score',
      invalidCount: 'count',
      uncertainCount: 'count'
    },
    stages: ['neighbourhood', 'operator', 'inference'],
    datumKinds: [{kind: 'cell', label: 'rainfall cell', indexSpace: '2d'}],
    representation: {from: 'event', to: 'field'},
    guards: ['coordinate-model'],
    choices: [
      {stage: 'neighbourhood', name: 'searchRadius', unit: 'm', sweepable: true},
      {stage: 'neighbourhood', name: 'neighborCount', sweepable: true},
      {stage: 'neighbourhood', name: 'minimumNeighborCount', sweepable: true},
      {
        stage: 'operator',
        name: 'model',
        alternatives: ['spherical', 'exponential', 'gaussian'],
        sweepable: false
      },
      {stage: 'operator', name: 'nugget', sweepable: true},
      {stage: 'operator', name: 'sill', sweepable: true},
      {stage: 'operator', name: 'range', unit: 'm', sweepable: true}
    ],
    // gpu-kriging.ts evaluateVariogram, solveSystem and krige: normalized ordinary system and restored variance.
    formula: {
      id: 'gpu-spatial-analysis.kriging',
      tex: String.raw`\begin{bmatrix}\term{matrix}{\bar\Gamma}&1\\1^T&0\end{bmatrix}\begin{bmatrix}\term{weights}{w}\\\term{multiplier}{\mu}\end{bmatrix}=\begin{bmatrix}\term{target}{\bar\gamma_u}\\1\end{bmatrix},\quad \term{prediction}{\hat z}=\term{base}{z_0}+\sum_i\term{weights}{w_i}(\term{values}{\bar z_i}-\term{base}{z_0}),\quad \term{variance}{\sigma^2}=(\term{nugget}{a}+\term{sill}{s})\max(0,\term{multiplier}{\mu}+\sum_i\term{weights}{w_i}\term{target}{\bar\gamma_{iu}})`,
      ascii:
        '[GammaBar 1; 1^T 0] [w; mu] = [gammaBar(target); 1]. gammaBar(0)=0; gammaBar(h)=(nugget+sill*shape(h/range))/(nugget+sill). Spherical shape(t)=1.5t-0.5t³ for t<1, else 1. Prediction=z0+sum(w_i*(meanValue_i-z0)); variance=(nugget+sill)*max(0,mu+sum(w_i*gammaBar_i)). Up to k nearest rows inside the radius, then duplicate positions coalesced; an exact hit returns their mean with zero variance. Too few distinct neighbors or rejected pivots/variance produce NaN; only normalized variance >= -1e-4 is clamped.',
      symbols: [
        {
          id: 'matrix',
          role: 'intermediate',
          label: 'normalized variogram matrix',
          binding: 'wgsl:system'
        },
        {id: 'weights', role: 'intermediate', label: 'kriging weights', binding: 'wgsl:weight'},
        {
          id: 'multiplier',
          role: 'intermediate',
          label: 'Lagrange multiplier',
          binding: 'wgsl:system'
        },
        {
          id: 'target',
          role: 'intermediate',
          label: 'normalized target variogram',
          binding: 'wgsl:targetGamma'
        },
        {id: 'prediction', role: 'output', label: 'rainfall prediction', binding: 'wgsl:valuesOut'},
        {
          id: 'base',
          role: 'intermediate',
          label: 'nearest distinct value',
          binding: 'wgsl:baseValue'
        },
        {
          id: 'values',
          role: 'input',
          label: 'mean value at distinct position',
          binding: 'wgsl:uniqueValueSums'
        },
        {id: 'variance', role: 'output', label: 'kriging variance', binding: 'wgsl:varianceOut'},
        {id: 'nugget', role: 'parameter', label: 'nugget', binding: 'wgsl:nugget'},
        {id: 'sill', role: 'parameter', label: 'partial sill', binding: 'wgsl:sill'}
      ],
      derivation: 'curated'
    },
    metadata: {coordinateModel: 'projected', metricUnit: 'm', mappedVariableExtent: 'intensive'}
  },
  GPUDotDensity: {
    family: 'density',
    descriptor: {scope: 'zonal', reduction: 'none'},
    variables: {
      population: 'count',
      points: 'measure',
      sourceIds: 'category',
      slotCounts: 'count',
      count: 'count',
      failedCount: 'count'
    },
    stages: ['representation', 'operator', 'presentation'],
    datumKinds: [{kind: 'feature', label: 'census tract', indexSpace: '1d'}],
    representation: {from: 'object', to: 'event'},
    guards: ['coordinate-model', 'count-vs-rate'],
    choices: [
      {stage: 'operator', name: 'seed', sweepable: false},
      {stage: 'presentation', name: 'dotsPerUnit', sweepable: true},
      {stage: 'operator', name: 'maximumAttempts', sweepable: false}
    ],
    formula: {
      formula: 'none',
      reason:
        'Seeded stochastic rounding allocates dots, then GPU triangulation and area-weighted sampling place them inside each feature.'
    },
    metadata: {coordinateModel: 'projected', metricUnit: 'm', mappedVariableExtent: 'extensive'}
  },
  GPURandomPointsInPolygon: {
    family: 'density',
    descriptor: {scope: 'zonal', reduction: 'none'},
    variables: {
      counts: 'count',
      points: 'measure',
      sourceIds: 'category',
      slotCounts: 'count',
      count: 'count',
      failedCount: 'count'
    },
    stages: ['support', 'operator'],
    datumKinds: [{kind: 'feature', label: 'community area', indexSpace: '1d'}],
    representation: {from: 'object', to: 'event'},
    guards: ['coordinate-model'],
    choices: [
      {stage: 'operator', name: 'seed', sweepable: false},
      {stage: 'operator', name: 'maximumAttempts', sweepable: false}
    ],
    formula: {
      formula: 'none',
      reason:
        'GPU triangulation, prefix allocation and seeded area-weighted triangle draws place the requested points per polygon.'
    },
    metadata: {coordinateModel: 'projected', metricUnit: 'm'}
  },
  GPURandomPointsOnLine: {
    family: 'density',
    descriptor: {scope: 'zonal', reduction: 'none'},
    variables: {
      counts: 'count',
      points: 'measure',
      sourceIds: 'category',
      fractions: 'rate',
      count: 'count'
    },
    stages: ['support', 'operator'],
    datumKinds: [{kind: 'point', label: 'sample slot', indexSpace: '1d'}],
    representation: {from: 'network', to: 'event'},
    guards: ['coordinate-model'],
    choices: [{stage: 'operator', name: 'seed', sweepable: false}],
    formula: {
      formula: 'none',
      reason:
        'Seeded arc-length draws locate segments by cumulative length and interpolate positions before compact publication.'
    },
    metadata: {coordinateModel: 'projected', metricUnit: 'm'}
  }
};
