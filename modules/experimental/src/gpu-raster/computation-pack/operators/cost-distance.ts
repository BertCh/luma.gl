// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {RasterOperatorTable} from '../types';

const CELL = {kind: 'cell', label: 'grid cell', indexSpace: '2d'} as const;

/** Raster operators of the cost-distance family, with formulas for the recorded settings. */
export const RASTER_COST_DISTANCE_OPERATORS: RasterOperatorTable = {
  GPUCostDistance: {
    family: 'cost-distance',
    descriptor: {scope: 'frontier', reduction: 'argmin'},
    variables: {
      landscapeElevation: 'elevation',
      costFriction: 'rate',
      costSourceMask: 'category',
      costs: 'measure',
      costSettledAt: 'round',
      costConverged: 'category',
      costIterationCount: 'round'
    },
    stages: ['support', 'neighbourhood', 'operator', 'iteration'],
    datumKinds: [CELL],
    neighbourhood: 'stencil-3x3',
    guards: ['metric-cell-size', 'convergence'],
    choices: [
      {stage: 'support', name: 'cellSizeMode', sweepable: false},
      {stage: 'iteration', name: 'maximumIterations', sweepable: true}
    ],
    formula: {
      id: 'gpu-raster.cost-distance.rule',
      derivation: 'curated',
      tex: String.raw`\term{cost}{C(c)}=\min\!\left(C(c),\min_{n\in N_8(c)} C(n)+\term{edge}{e(n,c)}\right),\quad e(n,c)=\term{distance}{\ell(n,c)}\frac{\term{friction}{f(n)+f(c)}}{2}`,
      ascii:
        'C(c) <- min(C(c), min over passable D8 neighbors n of C(n) + length(n,c)*(f(n)+f(c))/2), accepting candidates <= costLimit; sources start at 0, others +Infinity; axial length 120 m, diagonal sqrt(2)*120 m here; invalid, NaN, infinite or negative friction is impassable; diagonal corner cutting allowed; at the fixpoint costs are least accumulated costs, unreached/barriers +Infinity; if capped before convergence every passable cost becomes NaN.',
      symbols: [
        {id: 'cost', role: 'output', label: 'C', binding: 'wgsl:costs'},
        {id: 'friction', role: 'input', label: 'f', binding: 'wgsl:auxiliary'},
        {id: 'edge', role: 'intermediate', label: 'e', binding: 'wgsl:getEdgeCost'},
        {id: 'distance', role: 'intermediate', label: 'length', binding: 'wgsl:getD8Distance'}
      ]
    },
    sources: [
      'modules/experimental/src/gpu-raster/cost-distance/gpu-cost-distance.ts',
      'modules/experimental/src/gpu-raster/cost-distance/raster-relaxation.ts',
      'modules/experimental/src/gpu-raster/cost-distance/raster-grid-utils.ts'
    ]
  },
  GPUCostDistancePath: {
    family: 'cost-distance',
    descriptor: {scope: 'global', reduction: 'select'},
    variables: {
      landscapeElevation: 'elevation',
      costFriction: 'rate',
      costSourceMask: 'category',
      costs: 'measure',
      costBackLinks: 'category',
      costConverged: 'category',
      costIterationCount: 'round',
      pathTarget0: 'category',
      pathCells0: 'category',
      pathCount0: 'measure',
      pathOverflow0: 'category',
      pathMask0: 'category',
      pathTarget1: 'category',
      pathCells1: 'category',
      pathCount1: 'measure',
      pathOverflow1: 'category',
      pathMask1: 'category',
      pathTarget2: 'category',
      pathCells2: 'category',
      pathCount2: 'measure',
      pathOverflow2: 'category',
      pathMask2: 'category'
    },
    stages: ['support', 'operator', 'iteration'],
    datumKinds: [CELL],
    neighbourhood: 'path',
    guards: ['convergence'],
    choices: [{stage: 'iteration', name: 'maximumIterations', sweepable: true}],
    formula: {
      formula: 'none',
      reason:
        'A bounded pointer chase follows D8 back-links from each target to a source; invalid, cyclic or unresolved walks report an empty path.'
    },
    sources: [
      'modules/experimental/src/gpu-raster/cost-distance/gpu-cost-distance-path.ts',
      'modules/experimental/src/gpu-raster/cost-distance/gpu-cost-distance.ts'
    ]
  },
  GPUDistanceField: {
    family: 'cost-distance',
    descriptor: {scope: 'global', reduction: 'argmin'},
    variables: {landscapeElevation: 'elevation', waterSeeds: 'category', waterDistance: 'measure'},
    stages: ['support', 'operator'],
    datumKinds: [CELL],
    neighbourhood: 'none',
    guards: ['metric-cell-size'],
    choices: [
      {stage: 'operator', name: 'mode', sweepable: false},
      {stage: 'support', name: 'cellSizeMode', sweepable: false}
    ],
    formula: {
      id: 'gpu-raster.distance-field.rule',
      derivation: 'curated',
      tex: String.raw`\term{distance}{d(c)}=\term{cellSize}{h}\min_{\term{seed}{s}\in S}\lVert\term{cell}{c}-s\rVert_2`,
      ascii:
        'd(c) = 120 * correctlyRoundedSqrt(min over water seed cells s of (column(c)-column(s))^2 + (row(c)-row(s))^2) in metres; exact separable Felzenszwalb-Huttenlocher EDT on this isotropic grid (integer squared-distance keys); no seed or distance > maximumDistance gives +Infinity; distances are cell-centre to cell-centre, ties prefer smallest seed ID. This run uses exact mode, not jump flooding.',
      symbols: [
        {id: 'distance', role: 'output', label: 'd', binding: 'wgsl:distanceBits'},
        {id: 'cell', role: 'input', label: 'c', binding: 'wgsl:index'},
        {id: 'seed', role: 'intermediate', label: 's', binding: 'wgsl:nearestCell'},
        {id: 'cellSize', role: 'parameter', label: 'h', binding: 'wgsl:settings'}
      ]
    },
    sources: [
      'modules/experimental/src/gpu-raster/distance-field/gpu-distance-field.ts',
      'modules/experimental/src/gpu-raster/distance-field/distance-field-kernels.ts'
    ]
  }
};
