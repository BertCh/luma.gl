// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {RasterOperatorTable} from '../../../gpu-raster/computation-pack/types';

const CELL = {kind: 'cell', label: 'grid cell', indexSpace: '2d'} as const;

const CELL_SIZE_MODE = {
  stage: 'support',
  name: 'cellSizeMode',
  alternatives: ['uniform', 'web-mercator', 'geographic'],
  sweepable: false
} as const;

/** Terrain pack entries of the `relief` family, keyed by exported class name. */
export const TERRAIN_RELIEF_OPERATORS: RasterOperatorTable = {
  GPUSimpleLocalRelief: {
    family: 'relief',
    descriptor: {scope: 'focal', reduction: 'weighted-mean'},
    variables: {dem: 'elevation', simpleLocalRelief: 'score'},
    stages: ['neighbourhood', 'operator'],
    datumKinds: [CELL],
    neighbourhood: 'window',
    guards: [],
    choices: [
      {stage: 'neighbourhood', name: 'radius', unit: 'cell', sweepable: true},
      {stage: 'operator', name: 'verticalExaggeration', sweepable: true}
    ],
    // RVT SLRM as implemented: the (2r + 1)^2 mean replicates edge pixels and skips nodata, and the
    // deviation is formed without a large absolute sum, so centimetre relief survives float32.
    formula: {
      id: 'gpu-terrain.simple-local-relief.slrm',
      tex: String.raw`\term{relief}{\mathrm{SLRM}} = \term{exaggeration}{v}\left(z_c - \frac{1}{\term{count}{N}}\sum_{j \in W_{\term{radius}{r}}} z_j\right)`,
      ascii:
        'SLRM = verticalExaggeration * (z_centre - mean of z over the (2r+1)^2 window), r = radius in pixels (no cell size involved). Window coordinates outside the raster replicate the edge pixel (numpy pad edge, so an edge sample may count several times); invalid samples are excluded from both the sum and the count N; an invalid centre gives NaN. Computed as -total/N with total = sum(z_j - z_c), from an exact 64-bit summed-area table (quantum 2^-16) or the separable row/column filter on very large grids.',
      derivation: 'curated',
      symbols: [
        {id: 'relief', role: 'output', label: 'SLRM', binding: 'wgsl:deviationValue'},
        {id: 'exaggeration', role: 'parameter', label: 'v', binding: 'wgsl:radiusSettings'},
        {id: 'count', role: 'intermediate', label: 'N', binding: 'wgsl:count'},
        {id: 'radius', role: 'parameter', label: 'radius', binding: 'wgsl:radius'}
      ]
    },
    sources: [
      'modules/experimental/src/gpu-terrain/relief-visualization/gpu-simple-local-relief.ts',
      'modules/experimental/src/gpu-terrain/relief-visualization/relief-visualization-utils.ts'
    ]
  },
  GPUMultiScaleRelief: {
    family: 'relief',
    descriptor: {scope: 'focal', reduction: 'weighted-mean'},
    variables: {dem: 'elevation', multiScaleRelief: 'score'},
    stages: ['support', 'neighbourhood', 'operator'],
    datumKinds: [CELL],
    neighbourhood: 'window',
    guards: ['metric-cell-size'],
    choices: [
      CELL_SIZE_MODE,
      {stage: 'neighbourhood', name: 'featureMinimum', unit: 'm', sweepable: true},
      {stage: 'neighbourhood', name: 'featureMaximum', unit: 'm', sweepable: true},
      {stage: 'neighbourhood', name: 'scalingFactor', sweepable: false},
      {stage: 'operator', name: 'verticalExaggeration', sweepable: true}
    ],
    // RVT MSRM as implemented: the sum of consecutive mean-filter differences telescopes, so only the
    // coarsest and finest filters run; each is a precise deviation z - mean (radius 0 is z itself).
    formula: {
      id: 'gpu-terrain.multi-scale-relief.telescoped',
      tex: String.raw`\term{relief}{\mathrm{MSRM}} = \term{exaggeration}{v}\,\frac{\term{coarse}{d_n} - \term{fine}{d_i}}{\term{scaleCount}{n - i}},\quad d_k = z_c - \overline{z}_{k^s}`,
      ascii:
        'Indices i = floor(((featureMinimum - res)/(2 res))^(1/s)) and n = ceil(((featureMaximum - res)/(2 res))^(1/s)), mean-filter radii k^s pixels. d_k = z - mean of the (2 k^s + 1)^2 edge-replicated, nodata-skipping window (d = 0 for radius 0, so a first radius of 0 contributes nothing). MSRM = verticalExaggeration * (d_n - d_i) / (n - i), which equals the mean of consecutive filter differences (mean_i - mean_n)/(n - i). Invalid centres give NaN. In geographic or web-mercator mode each radius k^s becomes k^s * resolution ground metres, converted to a rectangle of at least one pixel per axis.',
      derivation: 'curated',
      symbols: [
        {id: 'relief', role: 'output', label: 'MSRM', binding: 'wgsl:value'},
        {id: 'exaggeration', role: 'parameter', label: 'v', binding: 'wgsl:settings'},
        {
          id: 'coarse',
          role: 'intermediate',
          label: 'coarsest deviation',
          binding: 'wgsl:coarseValue'
        },
        {id: 'fine', role: 'intermediate', label: 'finest deviation', binding: 'wgsl:fineValue'},
        {id: 'scaleCount', role: 'constant', label: 'n - i', binding: 'wgsl:SCALE_COUNT'}
      ]
    },
    sources: [
      'modules/experimental/src/gpu-terrain/relief-visualization/gpu-multi-scale-relief.ts',
      'modules/experimental/src/gpu-terrain/relief-visualization/relief-visualization-utils.ts'
    ]
  },
  GPULocalDominance: {
    family: 'relief',
    descriptor: {scope: 'focal', reduction: 'sum'},
    variables: {dem: 'elevation', localDominance: 'rate'},
    stages: ['neighbourhood', 'operator'],
    datumKinds: [CELL],
    neighbourhood: 'window',
    guards: [],
    choices: [
      {stage: 'neighbourhood', name: 'minimumRadius', unit: 'cell', sweepable: true},
      {stage: 'neighbourhood', name: 'maximumRadius', unit: 'cell', sweepable: true},
      {stage: 'neighbourhood', name: 'angularResolution', unit: 'deg', sweepable: false},
      {stage: 'operator', name: 'observerHeight', unit: 'm', sweepable: true}
    ],
    // RVT local dominance as implemented: rise is formed from the exaggerated height difference plus the
    // observer height, only positive rises count, and ring-area weights make a flat surface score 1.
    formula: {
      id: 'gpu-terrain.local-dominance.rise-sum',
      tex: String.raw`\term{dominance}{D} = \frac{1}{\term{observerHeight}{h}\;\term{normalization}{\mathcal{N}}}\sum_{s}\term{factor}{f_s}\,\max\!\left(0,\ \term{rise}{\rho_s}\right),\quad \rho_s = \term{exaggeration}{v}\left(z_c - z_s\right) + h`,
      ascii:
        'Baked offsets (distance d, angle a): d = minimumRadius + radiusIncrement*k, a = angularResolution*j, row shift round(sin a * d), column shift round(cos a * d), rounding half to even, samples edge-clamped at (row - rowShift, column - columnShift). rise = verticalExaggeration*(z_c - z_s) + observerHeight; offsets with rise > 0 (an invalid sample gives NaN, which never passes) add rise * (2d + radiusIncrement)/d. D = sum / (observerHeight * N) with N = sum over offsets of (2d + radiusIncrement)/d, so a flat surface scores 1. Invalid samples stay in N; an invalid centre gives NaN. Pixel units: no cell size.',
      derivation: 'curated',
      symbols: [
        {id: 'dominance', role: 'output', label: 'D', binding: 'wgsl:dominance'},
        {id: 'observerHeight', role: 'parameter', label: 'h', binding: 'wgsl:observerHeight'},
        {id: 'normalization', role: 'constant', label: 'N', binding: 'wgsl:NORMALIZATION'},
        {id: 'factor', role: 'constant', label: 'ring weight', binding: 'wgsl:FACTORS'},
        {id: 'rise', role: 'intermediate', label: 'rise', binding: 'wgsl:rise'},
        {
          id: 'exaggeration',
          role: 'parameter',
          label: 'v',
          binding: 'wgsl:verticalExaggeration'
        }
      ]
    },
    sources: ['modules/experimental/src/gpu-terrain/relief-visualization/gpu-local-dominance.ts']
  },
  GPUReliefBlend: {
    family: 'relief',
    descriptor: {scope: 'local', reduction: 'none'},
    variables: {
      dem: 'elevation',
      hillshade: 'fraction',
      slope: 'rate',
      reliefBlend: 'fraction'
    },
    stages: ['operator', 'presentation'],
    datumKinds: [CELL],
    neighbourhood: 'none',
    guards: [],
    choices: [
      {stage: 'presentation', name: 'minimum', sweepable: false},
      {stage: 'presentation', name: 'maximum', sweepable: false},
      {
        stage: 'presentation',
        name: 'blendMode',
        alternatives: ['normal', 'multiply', 'screen', 'overlay', 'soft-light', 'luminosity'],
        sweepable: false
      },
      {stage: 'presentation', name: 'opacity', sweepable: true}
    ],
    // RVT BlenderCombination as implemented: layers are stretched to [0, 1], optionally inverted, and
    // composited bottom to top with the documented opacity applied to every blend mode.
    formula: {
      id: 'gpu-terrain.relief-blend.composite',
      tex: String.raw`\term{background}{b_j} = \term{opacity}{\alpha_j}\,\term{mode}{f_{m_j}}\!\left(\term{normalized}{n_j},\ b_{j-1}\right) + \left(1 - \alpha_j\right) b_{j-1},\quad n_j = \operatorname{clamp}\!\left(\frac{x_j - \term{minimum}{x_{\min}}}{\term{maximum}{x_{\max}} - x_{\min}},\ 0,\ 1\right)`,
      ascii:
        'Layers are ordered bottom to top. n_j = clamp((x_j - min_j)/(max_j - min_j), 0, 1), replaced by 1 - n_j when the layer is inverted. The bottom layer is used as is: b_0 = n_0. For j >= 1: b_j = opacity_j * mode(n_j, b_{j-1}) + (1 - opacity_j) * b_{j-1}. Modes with foreground a = n_j and background b: normal and luminosity a; multiply a*b; screen 1-(1-a)(1-b); overlay (switches on the background) 2ab if b <= 0.5 else 1-(1-2(b-0.5))(1-a); soft-light (switches on the foreground) 2ba + b^2(1-2a) if a < 0.5 else 2b(1-a) + sqrt(b)(2a-1). Any non-finite layer value gives NaN.',
      derivation: 'curated',
      symbols: [
        {id: 'background', role: 'output', label: 'blended value', binding: 'wgsl:background'},
        {id: 'opacity', role: 'parameter', label: 'opacity', binding: 'wgsl:opacity'},
        {id: 'mode', role: 'intermediate', label: 'blend mode', binding: 'wgsl:blendLayer'},
        {
          id: 'normalized',
          role: 'intermediate',
          label: 'stretched layer',
          binding: 'wgsl:normalized'
        },
        {id: 'minimum', role: 'parameter', label: 'stretch minimum', binding: 'wgsl:minimum'},
        {id: 'maximum', role: 'parameter', label: 'stretch maximum', binding: 'wgsl:maximum'}
      ]
    },
    sources: ['modules/experimental/src/gpu-terrain/relief-visualization/gpu-relief-blend.ts']
  }
};
