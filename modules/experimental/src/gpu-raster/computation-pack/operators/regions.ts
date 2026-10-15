// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {RasterOperatorTable} from '../types';

const CELL = {kind: 'cell', label: 'grid cell', indexSpace: '2d'} as const;
const PATCH = {kind: 'patch', label: 'patch', indexSpace: '1d'} as const;

/** Raster pack entries of the `regions` family, keyed by exported class name. */
export const RASTER_REGIONS_OPERATORS: RasterOperatorTable = {
  GPURasterPatchMetrics: {
    family: 'regions',
    descriptor: {scope: 'zonal', reduction: 'count'},
    variables: {
      patchBurnMask: 'category',
      patchLabels: 'category',
      patchConverged: 'category',
      patchPixelCounts: 'measure',
      patchAreas: 'measure',
      patchPerimeters: 'measure',
      patchPerimeterFaces: 'measure'
    },
    stages: ['representation', 'neighbourhood', 'operator'],
    datumKinds: [PATCH, CELL],
    neighbourhood: 'stencil-3x3',
    guards: ['convergence'],
    choices: [
      {stage: 'neighbourhood', name: 'connectivity', alternatives: [4, 8], sweepable: false},
      {stage: 'operator', name: 'countRasterBorder', sweepable: false}
    ],
    formula: {
      id: 'gpu-raster.raster-patch-metrics.rule',
      derivation: 'curated',
      tex: String.raw`\term{area}{A_p}=\term{count}{n_p}\,\lvert\det G\rvert,\qquad \term{perimeter}{P_p}=\term{verticalFaces}{f_p^{\updownarrow}}\,\Delta x+\term{horizontalFaces}{f_p^{\leftrightarrow}}\,\Delta y`,
      ascii:
        'For dense patch label p (row p-1): n_p = pixels with that label; area = n_p * |a*e - b*d| of the grid affine (14400 m2 here); a pixel face is exposed when its 4-neighbour holds another label, background or an invalid pixel, or lies outside the raster (countRasterBorder true); perimeter = (exposed faces between vertical neighbours) * column step + (exposed faces between horizontal neighbours) * row step (120 m here); holes add their inner boundary. All counts are exact integer atomics; while the labelling has not converged every column is filled with the unresolved sentinel (NaN for floats).',
      symbols: [
        {id: 'area', role: 'output', label: 'area', binding: 'wgsl:areas'},
        {id: 'count', role: 'intermediate', label: 'pixel count', binding: 'wgsl:pixelCounts'},
        {id: 'perimeter', role: 'output', label: 'perimeter', binding: 'wgsl:perimeters'},
        {id: 'verticalFaces', role: 'intermediate', label: 'faces', binding: 'wgsl:acrossRows'},
        {id: 'horizontalFaces', role: 'intermediate', label: 'faces', binding: 'wgsl:acrossColumns'}
      ]
    },
    sources: [
      'modules/experimental/src/gpu-raster/raster-patches/gpu-raster-patch-metrics.ts',
      'modules/experimental/src/gpu-raster/raster-patches/patch-labels.ts'
    ]
  },
  GPURasterSieve: {
    family: 'regions',
    descriptor: {scope: 'zonal', reduction: 'select'},
    variables: {
      sieveBurnMask: 'category',
      sieveLabels: 'category',
      sieveConverged: 'category',
      sieveOutputLabels: 'category',
      sievePatchTargets: 'category',
      sieveSievedCount: 'measure'
    },
    stages: ['representation', 'neighbourhood', 'operator'],
    datumKinds: [PATCH, CELL],
    neighbourhood: 'stencil-3x3',
    guards: ['convergence'],
    choices: [
      {stage: 'operator', name: 'minimumPixels', sweepable: true},
      {stage: 'operator', name: 'mode', alternatives: ['remove', 'merge'], sweepable: false},
      {stage: 'neighbourhood', name: 'connectivity', alternatives: [4, 8], sweepable: false}
    ],
    formula: {
      id: 'gpu-raster.raster-sieve.rule',
      derivation: 'curated',
      tex: String.raw`\term{label}{\ell'(c)}=\begin{cases}\ell(c)&\term{count}{n_{\ell(c)}}\ge\term{minimum}{m}\\0&\text{otherwise}\end{cases}`,
      ascii:
        'remove mode: a patch with n pixels keeps its label when n >= minimumPixels and otherwise becomes 0 (background), so patchTargets[p-1] = p or 0 and every pixel takes patchTargets[label-1]; background stays 0; merge mode instead gives a small patch the label of its largest adjacent patch that is not small (lowest label on ties; none leaves it removed). sievedCount counts non-empty patches with n < minimumPixels; unconverged labels fill every output word with the unresolved sentinel.',
      symbols: [
        {id: 'label', role: 'output', label: 'sieved label', binding: 'wgsl:sievedLabels'},
        {id: 'count', role: 'intermediate', label: 'patch size', binding: 'wgsl:pixelCounts'},
        {id: 'minimum', role: 'parameter', label: 'minimum pixels', binding: 'wgsl:parameters'}
      ]
    },
    sources: [
      'modules/experimental/src/gpu-raster/raster-patches/gpu-raster-sieve.ts',
      'modules/experimental/src/gpu-raster/raster-patches/patch-labels.ts'
    ]
  },
  GPURasterExtremaPyramid: {
    family: 'regions',
    descriptor: {scope: 'block', reduction: 'select'},
    variables: {
      pyramidElevation: 'elevation',
      pyramidMaximumLevel1: 'elevation',
      pyramidMinimumLevel1: 'elevation'
    },
    stages: ['representation', 'support', 'operator'],
    datumKinds: [{kind: 'block', label: 'level-1 block', indexSpace: '2d'}, CELL],
    neighbourhood: 'window',
    guards: [],
    choices: [
      {stage: 'support', name: 'firstBlockSize', sweepable: false},
      {stage: 'support', name: 'footprint', alternatives: ['cell', 'bilinear'], sweepable: false},
      {stage: 'support', name: 'levelCount', sweepable: false}
    ],
    formula: {
      id: 'gpu-raster.raster-extrema-pyramid.rule',
      derivation: 'curated',
      tex: String.raw`\term{parent}{M_{l+1}(i,j)}=\max_{a,b\in\{0,1\}}\term{child}{M_l(2i+a,\,2j+b)},\qquad M_0(i,j)=\max_{\mathrm{valid}\ v\in\mathrm{block}(i,j)}v`,
      ascii:
        "Level 0 cell (i,j) is the maximum over the valid finite pixels of its S x S block (S = firstBlockSize = 4; footprint 'cell' adds no extra pixel). Level l+1 takes the maximum of the children (2i+a, 2j+b) of level l that exist, compared with strict > from the empty sentinel -FLT_MAX, in the order (0,0), (1,0), (0,1), (1,1); children past an odd edge are skipped, so an edge cell reduces fewer than four. The minimum is the same tree with < from +FLT_MAX; a cell with no valid pixel keeps the sentinel.",
      symbols: [
        {id: 'parent', role: 'output', label: 'parent maximum', binding: 'wgsl:maximumValue'},
        {id: 'child', role: 'input', label: 'child maximum', binding: 'wgsl:tileMaximum'}
      ]
    },
    sources: ['modules/experimental/src/gpu-raster/raster-pyramid/gpu-raster-extrema-pyramid.ts']
  }
};
