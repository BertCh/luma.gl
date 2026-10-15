// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {RasterOperatorTable} from '../../../gpu-raster/computation-pack/types';

const CELL = {kind: 'cell', label: 'grid cell', indexSpace: '2d'} as const;

/** Terrain pack entries of the `hydrology` family, keyed by exported class name. */
export const TERRAIN_HYDROLOGY_OPERATORS: RasterOperatorTable = {
  GPUTerrainFlow: {
    family: 'hydrology',
    descriptor: {scope: 'frontier', reduction: 'sum'},
    variables: {
      dem: 'elevation',
      flowFilledElevation: 'elevation',
      flowFillRelaxation: 'elevation',
      flowDirections: 'category',
      flowCellClasses: 'category',
      flowAccumulation: 'measure',
      flowStreams: 'category',
      flowFillConverged: 'category',
      flowFillIterationCount: 'round',
      flowAccumulationConverged: 'category',
      flowAccumulationIterationCount: 'round'
    },
    stages: ['support', 'neighbourhood', 'operator', 'iteration'],
    datumKinds: [CELL],
    neighbourhood: 'stencil-3x3',
    guards: ['metric-cell-size', 'convergence'],
    choices: [
      {
        stage: 'support',
        name: 'cellSizeMode',
        alternatives: ['uniform', 'web-mercator', 'geographic'],
        sweepable: false
      },
      {stage: 'operator', name: 'fillEpsilon', label: 'fill epsilon', unit: 'm', sweepable: true},
      {
        stage: 'operator',
        name: 'streamThreshold',
        label: 'stream threshold',
        unit: 'cells',
        sweepable: true
      },
      {
        stage: 'operator',
        name: 'accumulationUnits',
        alternatives: ['cells', 'area'],
        sweepable: false
      },
      {stage: 'iteration', name: 'maximumFillIterations', sweepable: true},
      {stage: 'iteration', name: 'maximumAccumulationIterations', sweepable: true}
    ],
    // As implemented: fill is the epsilon relaxation, routing is steepest D8 descent on the
    // filled surface over ground distance, and accumulation with a runoff view is the
    // deterministic pull (a thread finalizes a cell once every donor is final, then walks
    // downstream). Without a runoff view D8 'cells' accumulation is Euler-tour list ranking.
    formula: {
      id: 'gpu-terrain.terrain-flow.d8',
      tex: String.raw`\term{receiver}{r}(c) = \operatorname*{arg\,max}_{k}\ \frac{\term{surface}{\tilde z_c} - \term{neighborSurface}{\tilde z_k}}{\term{groundDistance}{d_k}},\quad \tilde z_c = \max\!\bigl(z_c,\ \min_n \tilde z_n + \term{epsilon}{\varepsilon}\bigr),\quad \term{accumulation}{A}(c) = \term{weight}{w_c} + \sum_{u:\ r(u) = c} A(u)`,
      ascii:
        'r(c) = argmax over the 8 neighbours k of (zf(c) - zf(k)) / d_k where zf is the filled surface, d_k = cell size (cardinal) or sqrt(2) * cell size (diagonal), only drops > 0 route, ties keep the lowest direction index; zf(c) = max(z(c), min over neighbours of zf(n) + epsilon) with boundary cells keeping z; A(c) = w(c) + sum of A(u) over donors u with r(u) = c, w = runoff (1 here), solved by pull rounds: a thread finalizes a cell once every donor is final and then walks downstream (up to 1024 steps); streams = A >= threshold',
      derivation: 'curated',
      symbols: [
        {id: 'receiver', role: 'output', label: 'r', binding: 'wgsl:bestNeighbor'},
        {id: 'surface', role: 'input', label: 'z̃_c', binding: 'wgsl:centerSurface'},
        {id: 'neighborSurface', role: 'input', label: 'z̃_k', binding: 'wgsl:neighborSurface'},
        {id: 'groundDistance', role: 'parameter', label: 'd_k', binding: 'wgsl:getD8Distance'},
        {id: 'epsilon', role: 'parameter', label: 'ε', binding: 'wgsl:epsilon'},
        {id: 'accumulation', role: 'output', label: 'A', binding: 'wgsl:acc'},
        {id: 'weight', role: 'parameter', label: 'w_c', binding: 'wgsl:getWeight'}
      ]
    },
    sources: [
      'modules/experimental/src/gpu-terrain/hydrology/gpu-terrain-flow.ts',
      'modules/experimental/src/gpu-terrain/hydrology/terrain-flow-passes.ts',
      'modules/experimental/src/gpu-terrain/hydrology/terrain-flow-spill-fill.ts'
    ]
  },
  GPUTerrainHeightAboveDrainage: {
    family: 'hydrology',
    descriptor: {scope: 'frontier', reduction: 'select'},
    variables: {
      dem: 'elevation',
      flowFilledElevation: 'elevation',
      flowDirections: 'category',
      flowStreams: 'category',
      heightAboveDrainage: 'measure',
      drainageCells: 'category',
      handConverged: 'category',
      handIterationCount: 'round'
    },
    stages: ['operator', 'iteration'],
    datumKinds: [CELL],
    neighbourhood: 'path',
    guards: ['convergence'],
    choices: [{stage: 'iteration', name: 'maximumIterations', sweepable: true}],
    // HAND as implemented: pointer jumping finds the first stream cell on the D8 path, then one
    // subtraction of two float32 elevations. No cell size enters.
    formula: {
      id: 'gpu-terrain.terrain-height-above-drainage.hand',
      tex: String.raw`\term{hand}{\mathrm{HAND}}(c) = \term{cellElevation}{z_c} - \term{drainageElevation}{z_{d(c)}},\quad \term{drainage}{d}(c) = \text{first stream cell on the D8 path from } c`,
      ascii:
        'HAND(c) = z(c) - z(d(c)); d(c) = first stream cell on the D8 path from c (c itself on a stream cell), found by pointer jumping p = p[p]; 0 on stream cells; NaN when the path ends without a stream, meets an unresolved word, or the elevations are not finite',
      derivation: 'curated',
      symbols: [
        {id: 'hand', role: 'output', label: 'HAND', binding: 'wgsl:heightOut'},
        {id: 'cellElevation', role: 'input', label: 'z_c', binding: 'wgsl:cellElevation'},
        {
          id: 'drainageElevation',
          role: 'input',
          label: 'z_d(c)',
          binding: 'wgsl:drainageElevation'
        },
        {id: 'drainage', role: 'intermediate', label: 'd', binding: 'wgsl:drainage'}
      ]
    },
    sources: [
      'modules/experimental/src/gpu-terrain/hydrology/gpu-terrain-height-above-drainage.ts',
      'modules/experimental/src/gpu-terrain/hydrology/terrain-flow-tracing.ts'
    ]
  },
  GPUTerrainWatersheds: {
    family: 'hydrology',
    descriptor: {scope: 'frontier', reduction: 'select'},
    variables: {
      dem: 'elevation',
      flowDirections: 'category',
      watershedLabels: 'category',
      watershedsConverged: 'category',
      watershedsIterationCount: 'round'
    },
    stages: ['operator', 'iteration'],
    datumKinds: [CELL],
    neighbourhood: 'path',
    guards: ['convergence'],
    choices: [{stage: 'iteration', name: 'maximumIterations', sweepable: true}],
    formula: {
      formula: 'none',
      reason:
        'Each cell follows its D8 receivers to a terminal cell by pointer jumping; the label is that fixpoint, with no closed form.'
    },
    sources: [
      'modules/experimental/src/gpu-terrain/hydrology/gpu-terrain-watersheds.ts',
      'modules/experimental/src/gpu-terrain/hydrology/terrain-flow-tracing.ts'
    ]
  },
  GPUTerrainStreamOrder: {
    family: 'hydrology',
    descriptor: {scope: 'frontier', reduction: 'select'},
    variables: {
      dem: 'elevation',
      flowDirections: 'category',
      flowStreams: 'category',
      streamOrder: 'rank',
      streamOrderConverged: 'category',
      streamOrderIterationCount: 'round'
    },
    stages: ['operator', 'iteration'],
    datumKinds: [CELL],
    neighbourhood: 'path',
    guards: ['convergence'],
    choices: [{stage: 'iteration', name: 'maximumIterations', sweepable: true}],
    formula: {
      formula: 'none',
      reason:
        'Strahler order is evaluated by levels on the Euler tour of the stream forest (list ranking, then scatter, scan and gather per level), not by a closed form.'
    },
    sources: [
      'modules/experimental/src/gpu-terrain/hydrology/gpu-terrain-stream-order.ts',
      'modules/experimental/src/gpu-terrain/hydrology/terrain-flow-euler-tour.ts'
    ]
  },
  GPUTerrainHydrologicIndices: {
    family: 'hydrology',
    descriptor: {scope: 'focal', reduction: 'select'},
    variables: {
      dem: 'elevation',
      flowFilledElevation: 'elevation',
      flowAccumulation: 'measure',
      indicesSlope: 'rate',
      specificCatchmentArea: 'measure',
      wetnessIndex: 'rate',
      streamPowerIndex: 'rate'
    },
    stages: ['support', 'neighbourhood', 'operator'],
    datumKinds: [CELL],
    neighbourhood: 'stencil-3x3',
    guards: ['metric-cell-size'],
    choices: [
      {
        stage: 'support',
        name: 'cellSizeMode',
        alternatives: ['uniform', 'web-mercator', 'geographic'],
        sweepable: false
      },
      {stage: 'operator', name: 'minimumSlope', label: 'minimum slope', sweepable: true}
    ],
    // As implemented: tan(beta) is the steepest D8 descent (not a Horn slope) raised to
    // minimumSlope; the contour width is sqrt(groundX * groundY) at the cell's row.
    formula: {
      id: 'gpu-terrain.terrain-hydrologic-indices.twi',
      tex: String.raw`\term{wetness}{\mathrm{TWI}} = \ln\frac{\term{catchment}{a}}{\term{tangent}{\tan\beta}},\quad a = \frac{\term{area}{A}}{\sqrt{\Delta x\,\Delta y}},\quad \tan\beta = \max\bigl(\term{steepest}{s},\ \term{minimumSlope}{s_{\min}}\bigr),\quad s = \max_k \frac{z_c - z_k}{d_k},\quad \mathrm{SPI} = a \tan\beta`,
      ascii:
        'TWI = ln(a / tanB); a = A / sqrt(groundX * groundY) at the cell row, A = contributing area in m2 incl. the cell; tanB = max(s, minimumSlope), s = max over valid neighbours of (z - zk) / d_k, floor 0; SPI = a * tanB; NaN where the cell or A is not finite; a <= 0 gives -Infinity',
      derivation: 'curated',
      symbols: [
        {id: 'wetness', role: 'output', label: 'TWI', binding: 'wgsl:wetness'},
        {id: 'catchment', role: 'intermediate', label: 'a', binding: 'wgsl:catchment'},
        {id: 'tangent', role: 'intermediate', label: 'tan β', binding: 'wgsl:tangent'},
        {id: 'area', role: 'input', label: 'A', binding: 'wgsl:area'},
        {id: 'steepest', role: 'intermediate', label: 's', binding: 'wgsl:steepest'},
        {id: 'minimumSlope', role: 'parameter', label: 's_min', binding: 'wgsl:minimumSlope'}
      ]
    },
    sources: ['modules/experimental/src/gpu-terrain/hydrology/gpu-terrain-hydrologic-indices.ts']
  }
};
