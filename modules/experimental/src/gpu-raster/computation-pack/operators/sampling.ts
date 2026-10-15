// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {RasterOperatorTable} from '../types';

const POINT = {kind: 'point', label: 'sample point', indexSpace: '1d'} as const;
const SAMPLE = {kind: 'sample', label: 'profile sample', indexSpace: '1d'} as const;

/** Raster pack entries of the `sampling` family, keyed by exported class name. */
export const RASTER_SAMPLING_OPERATORS: RasterOperatorTable = {
  GPURasterSampling: {
    family: 'sampling',
    descriptor: {scope: 'focal', reduction: 'weighted-mean'},
    variables: {
      samplingDem: 'elevation',
      samplingPoints: 'measure',
      samplingSampledPoints: 'measure',
      samplingValues: 'elevation'
    },
    stages: ['representation', 'support', 'neighbourhood', 'operator'],
    datumKinds: [POINT],
    neighbourhood: 'window',
    guards: [],
    choices: [
      {
        stage: 'operator',
        name: 'method',
        alternatives: ['nearest', 'bilinear', 'bicubic'],
        sweepable: true
      },
      {
        stage: 'operator',
        name: 'noDataPolicy',
        alternatives: ['strict', 'renormalize'],
        sweepable: false
      },
      {stage: 'support', name: 'localOrigin', sweepable: false}
    ],
    formula: {
      id: 'gpu-raster.raster-sampling.bilinear',
      derivation: 'curated',
      tex: String.raw`\term{value}{z}=\sum_{i,j\in\{0,1\}}\bigl(1-|i-f_x|\bigr)\bigl(1-|j-f_y|\bigr)\,\term{raster}{r}_{\lfloor u\rfloor+i,\ \lfloor v\rfloor+j},\quad (u,v)=\left(\frac{\term{position}{p_x}-x_{\min}}{\term{cellSize}{\Delta_x}}-\tfrac12,\ \frac{p_y-y_0}{\Delta_y}-\tfrac12\right)`,
      ascii:
        'Bilinear (the default method): u = (px - minX) / cellWidth - 0.5, v = (py - y0) / rowStep - 0.5 with y0 the y of the outer edge of row 0 and rowStep the signed row step (negative for north-up grids), so integer u, v are cell centres for either row direction; fx = u - floor(u), fy = v - floor(v); z = sum of the four surrounding centres weighted (1-|i-fx|)(1-|j-fy|), indices clamped to the edge cell. A centre takes part only when both its weights are nonzero, so a point on a cell centre returns that cell. Outside the closed extent or NaN position gives NaN. Under strict a nodata (NaN, sentinel or validity 0) participating centre gives NaN; renormalize divides the valid weights by their sum. nearest returns cell floor(u + 0.5), floor(v + 0.5) clamped to the raster (the cell containing the point; the maxX and maxY edge belong to the last cell); bicubic is Catmull-Rom over 4x4 and falls back to the bilinear rule next to nodata.',
      symbols: [
        {id: 'value', role: 'output', label: 'sampled value', binding: 'wgsl:valuesOut'},
        {id: 'raster', role: 'input', label: 'raster', binding: 'wgsl:raster'},
        {id: 'position', role: 'input', label: 'point', binding: 'wgsl:positions'},
        {id: 'cellSize', role: 'parameter', label: 'cell size', binding: 'wgsl:params'}
      ]
    },
    sources: [
      'modules/experimental/src/gpu-raster/raster-sampling/gpu-raster-sampling.ts',
      'modules/experimental/src/gpu-raster/raster-sampling/raster-sampling-utils.ts'
    ]
  },
  GPURasterProfile: {
    family: 'sampling',
    descriptor: {scope: 'focal', reduction: 'scan-sort'},
    variables: {
      profileDem: 'elevation',
      profileLine: 'measure',
      profileLineOffsets: 'measure',
      profilePositions: 'measure',
      profileValues: 'elevation',
      profileDistances: 'measure',
      profileGains: 'measure',
      profileCount: 'measure',
      profileOverflow: 'category'
    },
    stages: ['representation', 'support', 'neighbourhood', 'operator'],
    datumKinds: [SAMPLE],
    neighbourhood: 'window',
    guards: [],
    choices: [
      {stage: 'operator', name: 'spacing', unit: 'm', sweepable: true},
      {
        stage: 'operator',
        name: 'method',
        alternatives: ['nearest', 'bilinear', 'bicubic'],
        sweepable: false
      },
      {stage: 'support', name: 'localOrigin', sweepable: false}
    ],
    formula: {
      id: 'gpu-raster.raster-profile.samples',
      derivation: 'curated',
      tex: String.raw`\term{distance}{d_k}=\min(k\,\term{spacing}{s},\ L),\quad \term{position}{p_k}=\operatorname{lerp}\bigl(\text{polyline},\ d_k\bigr),\quad \term{value}{z_k}=\operatorname{bilinear}\bigl(\term{raster}{r},\ p_k\bigr),\quad \term{gain}{G}=\sum_{k}\max(0,\ z_k-z_{k-1})`,
      ascii:
        'Per line of planar length L (sum of vertex-to-vertex distances): samples at d_k = k * spacing for k * spacing < L, plus a last sample at exactly L (a zero-length line gives one sample, an empty line none); p_k interpolates linearly along the segment holding d_k; z_k is the GPURasterSampling rule (bilinear here) at p_k, NaN outside the extent or at nodata; gain sums the positive and loss the negative differences between consecutive finite samples, minimum and maximum cover finite samples. Sums use a fixed reduce-then-scan tree, so float32 rounding differs from a serial sum.',
      symbols: [
        {
          id: 'distance',
          role: 'intermediate',
          label: 'distance along line',
          binding: 'wgsl:distancesOut'
        },
        {id: 'spacing', role: 'parameter', label: 'spacing', binding: 'wgsl:params'},
        {
          id: 'position',
          role: 'intermediate',
          label: 'sample position',
          binding: 'wgsl:positionsOut'
        },
        {id: 'value', role: 'output', label: 'sampled value', binding: 'wgsl:valuesOut'},
        {id: 'raster', role: 'input', label: 'raster', binding: 'wgsl:raster'},
        {id: 'gain', role: 'output', label: 'gain', binding: 'wgsl:gain'}
      ]
    },
    sources: [
      'modules/experimental/src/gpu-raster/raster-sampling/gpu-raster-profile.ts',
      'modules/experimental/src/gpu-raster/raster-sampling/gpu-raster-profile-scan.ts',
      'modules/experimental/src/gpu-raster/raster-sampling/raster-sampling-utils.ts'
    ]
  }
};
