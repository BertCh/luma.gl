// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {RasterOperatorTable} from '../../../gpu-raster/computation-pack/types';

/** Terrain pack entries of the `flow-field` family, keyed by exported class name. */
export const TERRAIN_FLOW_FIELD_OPERATORS: RasterOperatorTable = {
  GPUTerrainFlowField: {
    family: 'flow-field',
    descriptor: {scope: 'focal', reduction: 'none'},
    variables: {dem: 'elevation', windVelocities: 'rate'},
    stages: ['support', 'neighbourhood', 'operator'],
    datumKinds: [{kind: 'cell', label: 'grid cell', indexSpace: '2d'}],
    neighbourhood: 'stencil-3x3',
    guards: ['metric-cell-size'],
    choices: [
      {
        stage: 'support',
        name: 'cellSizeMode',
        alternatives: ['uniform', 'web-mercator', 'geographic'],
        sweepable: false
      },
      {stage: 'operator', name: 'wind', unit: 'm/s', sweepable: true},
      {stage: 'operator', name: 'verticalExaggeration', sweepable: true}
    ],
    // Tangent-plane projection of the uniform wind: central differences in ground metres
    // (one-sided at the border) give the surface gradient, then v = w - (w . g) g / (1 + |g|^2).
    formula: {
      id: 'gpu-terrain.terrain-flow-field.tangent-projection',
      tex: String.raw`\term{velocity}{\mathbf{v}} = \term{wind}{\mathbf{w}} - \term{deflection}{\lambda}\,\term{gradient}{\mathbf{g}},\quad \lambda = \frac{\mathbf{w}\cdot\mathbf{g}}{1 + \lVert\mathbf{g}\rVert^2},\quad \mathbf{g} = \term{exaggeration}{\varepsilon}\left(\frac{z_{c+1} - z_{c-1}}{\Delta x\,(c_1 - c_0)},\ \frac{z_{r+1} - z_{r-1}}{\Delta y}\right)`,
      ascii:
        'g = exaggeration * ((z[c+1] - z[c-1]) / (dx * (c1 - c0)), (z[r+1] - z[r-1]) / dy) with one-sided differences at the border and the row spacing summed from the D8 distances; v = w - (w . g) g / (1 + g . g); w = (windX, windY) in the raster frame, x along columns, y along rows (south); NaN for both components when the cell or one of its four gradient neighbours is invalid',
      derivation: 'curated',
      symbols: [
        {id: 'velocity', role: 'output', label: 'v', binding: 'wgsl:velocity'},
        {id: 'wind', role: 'parameter', label: 'w', binding: 'wgsl:wind'},
        {id: 'deflection', role: 'intermediate', label: 'λ', binding: 'wgsl:deflection'},
        {id: 'gradient', role: 'intermediate', label: 'g', binding: 'wgsl:gradient'},
        {id: 'exaggeration', role: 'parameter', label: 'ε', binding: 'wgsl:exaggeration'}
      ]
    },
    sources: ['modules/experimental/src/gpu-terrain/terrain-flow-field/gpu-terrain-flow-field.ts']
  }
};
