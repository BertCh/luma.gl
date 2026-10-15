// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {RasterOperatorTable} from '../types';

const CELL = {kind: 'cell', label: 'grid cell', indexSpace: '2d'} as const;

/** Raster operators of the algebra family, with formulas for the recorded settings. */
export const RASTER_ALGEBRA_OPERATORS: RasterOperatorTable = {
  GPURasterArithmetic: {
    family: 'algebra',
    descriptor: {scope: 'local', reduction: 'none'},
    variables: {
      landscapeElevation: 'elevation',
      nirBefore: 'rate',
      nirAfter: 'rate',
      nirDifference: 'score'
    },
    stages: ['representation', 'operator'],
    datumKinds: [CELL],
    neighbourhood: 'none',
    guards: [],
    choices: [
      {stage: 'operator', name: 'operation', sweepable: false},
      {stage: 'representation', name: 'scaleA', sweepable: false},
      {stage: 'representation', name: 'scaleB', sweepable: false},
      {stage: 'representation', name: 'noDataValue', sweepable: false}
    ],
    formula: {
      id: 'gpu-raster.raster-arithmetic.rule',
      derivation: 'curated',
      tex: String.raw`\term{result}{\Delta N}=10^{-4}\term{after}{N_{\mathrm{after,DN}}}-10^{-4}\term{before}{N_{\mathrm{before,DN}}}\quad(\text{both raw values valid; otherwise NaN})`,
      ascii:
        'Subtract mode in this run: a = nirAfterDN*0.0001+0; b = nirBeforeDN*0.0001+0; result = min(max(a-b,-Infinity),+Infinity); either raw operand NaN or equal to noDataValue (0 here), or a NaN arithmetic result, yields NaN. The sentinel check precedes scaling.',
      symbols: [
        {id: 'result', role: 'output', label: 'difference', binding: 'wgsl:valuesOut'},
        {id: 'after', role: 'input', label: 'after DN', binding: 'wgsl:aRaw'},
        {id: 'before', role: 'input', label: 'before DN', binding: 'wgsl:bRaw'}
      ]
    },
    sources: [
      'modules/experimental/src/gpu-raster/raster-algebra/gpu-raster-arithmetic.ts',
      'modules/experimental/src/gpu-raster/raster-algebra/raster-algebra-utils.ts'
    ]
  },
  GPURasterCellStatistics: {
    family: 'algebra',
    descriptor: {scope: 'local', reduction: 'weighted-mean'},
    variables: {
      landscapeElevation: 'elevation',
      nirMeanBefore: 'rate',
      nirMeanAfter: 'rate',
      nirMean: 'rate'
    },
    stages: ['representation', 'operator'],
    datumKinds: [CELL],
    neighbourhood: 'none',
    guards: [],
    choices: [
      {stage: 'operator', name: 'noDataPolicy', sweepable: false},
      {stage: 'operator', name: 'minimumValidCount', sweepable: true}
    ],
    formula: {
      id: 'gpu-raster.raster-cell-statistics.rule',
      derivation: 'curated',
      tex: String.raw`\term{mean}{\mu}=\frac{\term{sum}{\sum_{\ell\in V} x_\ell}}{\max(\term{count}{|V|},1)}\quad(|V|=2;\ \text{otherwise NaN})`,
      ascii:
        'Mean output: count valid layers (NaN and optional sentinel excluded), sum in layer order, mean = sum/max(count,1); write mean only when count >= max(1,minimumValidCount) and (noDataPolicy is ignore or count == layerCount), otherwise NaN. Here two reflectance layers, propagate, minimumValidCount=2; DN 0 is uploaded as NaN and valid DN is scaled by 0.0001.',
      symbols: [
        {id: 'mean', role: 'output', label: 'mean', binding: 'wgsl:meanOut'},
        {id: 'sum', role: 'intermediate', label: 'sum', binding: 'wgsl:sum'},
        {id: 'count', role: 'intermediate', label: 'count', binding: 'wgsl:count'}
      ]
    },
    sources: [
      'modules/experimental/src/gpu-raster/raster-algebra/gpu-raster-cell-statistics.ts',
      'modules/experimental/src/gpu-raster/raster-algebra/raster-algebra-utils.ts'
    ]
  },
  GPURasterConditional: {
    family: 'algebra',
    descriptor: {scope: 'local', reduction: 'none'},
    variables: {
      landscapeElevation: 'elevation',
      conditionalLandCover: 'category',
      conditionalFriction: 'rate'
    },
    stages: ['representation', 'operator'],
    datumKinds: [CELL],
    neighbourhood: 'none',
    guards: [],
    choices: [
      {stage: 'operator', name: 'comparison', sweepable: false},
      {stage: 'operator', name: 'threshold', sweepable: false},
      {stage: 'operator', name: 'constantA', sweepable: true},
      {stage: 'operator', name: 'constantB', sweepable: true}
    ],
    formula: {
      id: 'gpu-raster.raster-conditional.rule',
      derivation: 'curated',
      tex: String.raw`\term{result}{f}=\begin{cases}0.4&\term{condition}{L}=50\\1&L\ne50\end{cases}\quad(\text{valid condition and chosen branch; otherwise NaN})`,
      ascii:
        'condition = landCover == 50; chosen = condition ? 0.4 : 1; output = chosen if the condition value and chosen branch are neither NaN nor the optional noDataValue (0 here), otherwise NaN; the unchosen branch does not propagate nodata.',
      symbols: [
        {id: 'result', role: 'output', label: 'friction', binding: 'wgsl:valuesOut'},
        {id: 'condition', role: 'input', label: 'land cover', binding: 'wgsl:conditionValue'}
      ]
    },
    sources: [
      'modules/experimental/src/gpu-raster/raster-algebra/gpu-raster-conditional.ts',
      'modules/experimental/src/gpu-raster/raster-algebra/raster-algebra-utils.ts'
    ]
  },
  GPURasterReclassify: {
    family: 'algebra',
    descriptor: {scope: 'local', reduction: 'none'},
    variables: {
      landscapeElevation: 'elevation',
      suitabilityLandCover: 'category',
      suitability: 'rank'
    },
    stages: ['representation', 'operator'],
    datumKinds: [CELL],
    neighbourhood: 'none',
    guards: [],
    choices: [
      {stage: 'operator', name: 'breakCount', sweepable: false},
      {stage: 'operator', name: 'closed', sweepable: false}
    ],
    formula: {
      id: 'gpu-raster.raster-reclassify.rule',
      derivation: 'curated',
      tex: String.raw`\term{result}{y}=\term{classes}{v_k},\quad k=\#\{\term{breaks}{b_j}\le\term{value}{x}\}\quad(\text{valid }x;\ \text{otherwise NaN})`,
      ascii:
        'For valid x, k = number of active ascending breaks <= x (closed:left; closed:right counts breaks < x), reclassified = classValues[k]; invalid means NaN, optional sentinel, or validity==0 and writes NaN (classes output would be UINT_MAX). Here breaks=[20,35,45,65], values=[2,4,5,3,1], mapping observed tree/grass/crop/built-up/water to suitability 2/4/5/3/1.',
      symbols: [
        {id: 'value', role: 'input', label: 'land cover', binding: 'wgsl:source'},
        {id: 'breaks', role: 'parameter', label: 'breaks', binding: 'wgsl:breaks'},
        {id: 'classes', role: 'parameter', label: 'class values', binding: 'wgsl:classValues'},
        {id: 'result', role: 'output', label: 'suitability', binding: 'wgsl:reclassifiedOut'}
      ]
    },
    sources: [
      'modules/experimental/src/gpu-raster/raster-algebra/gpu-raster-reclassify.ts',
      'modules/experimental/src/gpu-raster/raster-algebra/raster-algebra-utils.ts'
    ]
  },
  GPUWeightedOverlay: {
    family: 'algebra',
    descriptor: {scope: 'local', reduction: 'sum'},
    variables: {
      landscapeElevation: 'elevation',
      overlayLandCover: 'category',
      overlayElevation: 'elevation',
      suitabilityScore: 'fraction'
    },
    stages: ['representation', 'operator'],
    datumKinds: [CELL],
    neighbourhood: 'none',
    guards: [],
    choices: [
      {stage: 'operator', name: 'landCoverWeight', sweepable: true},
      {stage: 'operator', name: 'elevationWeight', sweepable: true},
      {stage: 'operator', name: 'normalizeWeights', sweepable: false},
      {stage: 'operator', name: 'noDataPolicy', sweepable: false}
    ],
    formula: {
      id: 'gpu-raster.weighted-overlay.rule',
      derivation: 'curated',
      tex: String.raw`\term{score}{S}=\sum_{\ell=0}^{1}\term{weight}{w_\ell}\term{remap}{r_\ell}=0.7r_0+0.3r_1\quad(\text{both inputs and remaps valid; otherwise NaN})`,
      ascii:
        'Table mode here: r_l = remapValues[l,k_l], k_l counts active breaks <= x_l; score is fixed-order f32 sum 0.7*r_landCover + 0.3*r_elevation, no normalization. Land-cover breaks [20,35,45,65] map to [0.25,0.75,1,0.5,0]; elevation breaks [1100,1400,1700] m map to [1,0.75,0.5,0.25]. With propagate, any NaN/sentinel input yields NaN; a NaN remap is restricted under either policy; no contributing layers or a NaN sum also yields NaN. With ignore skip nodata layers; if normalizeWeights is enabled divide by sum(abs(contributing weights)), requiring that sum > 0.',
      symbols: [
        {id: 'score', role: 'output', label: 'score', binding: 'wgsl:scoreOut'},
        {id: 'weight', role: 'parameter', label: 'weight', binding: 'wgsl:weight'},
        {id: 'remap', role: 'intermediate', label: 'remapped', binding: 'wgsl:remapped'}
      ]
    },
    sources: [
      'modules/experimental/src/gpu-raster/raster-algebra/gpu-weighted-overlay.ts',
      'modules/experimental/src/gpu-raster/raster-algebra/raster-algebra-utils.ts'
    ]
  },
  GPURasterStretch: {
    family: 'algebra',
    descriptor: {scope: 'global', reduction: 'select'},
    variables: {landscapeElevation: 'elevation', stretchNirAfter: 'rate', stretchedNir: 'fraction'},
    stages: ['representation', 'operator', 'presentation'],
    datumKinds: [CELL],
    neighbourhood: 'none',
    guards: [],
    choices: [
      {stage: 'presentation', name: 'mode', sweepable: false},
      {stage: 'presentation', name: 'domain', sweepable: false},
      {stage: 'presentation', name: 'gamma', sweepable: true}
    ],
    formula: {
      id: 'gpu-raster.raster-stretch.rule',
      derivation: 'curated',
      tex: String.raw`\term{result}{t}=\operatorname{clamp}\!\left(\frac{\term{value}{x}-\term{low}{l}}{\term{high}{h}-l},0,1\right)\quad(h>l,\ \text{valid }x)`,
      ascii:
        'Linear auto-domain, gamma=1, sigmoid disabled: lo/hi are min/max of valid finite cells in the statistics window and region (0/0 if empty); invalid input (NaN, optional sentinel, validity==0) gives NaN; if hi>lo output clamp((x-lo)/(hi-lo),0,1); otherwise output 0 if x<lo, 1 if x>hi, else 0.5. Infinities are excluded from statistics but clamp in the apply step. Here DN 0 becomes NaN and valid DN is scaled by 0.0001.',
      symbols: [
        {id: 'result', role: 'output', label: 'stretched', binding: 'wgsl:stretchedOut'},
        {id: 'value', role: 'input', label: 'reflectance', binding: 'wgsl:value'},
        {id: 'low', role: 'intermediate', label: 'lo', binding: 'wgsl:lo'},
        {id: 'high', role: 'intermediate', label: 'hi', binding: 'wgsl:hi'}
      ]
    },
    sources: [
      'modules/experimental/src/gpu-raster/raster-stretch/gpu-raster-stretch.ts',
      'modules/experimental/src/gpu-raster/raster-algebra/raster-algebra-utils.ts'
    ]
  },
  GPUChangeDetection: {
    family: 'algebra',
    descriptor: {scope: 'local', reduction: 'none'},
    variables: {
      landscapeElevation: 'elevation',
      nirChangeBefore: 'rate',
      nirChangeAfter: 'rate',
      swirChangeBefore: 'rate',
      swirChangeAfter: 'rate',
      nirChangeDifference: 'score',
      swirChangeDifference: 'score'
    },
    stages: ['representation', 'operator'],
    datumKinds: [CELL],
    neighbourhood: 'none',
    guards: [],
    choices: [
      {stage: 'operator', name: 'beforeSlice', sweepable: false},
      {stage: 'operator', name: 'afterSlice', sweepable: false}
    ],
    formula: {
      id: 'gpu-raster.change-detection.rule',
      derivation: 'curated',
      tex: String.raw`\term{difference}{\Delta_b}=\term{after}{x_{1,b}}-\term{before}{x_{0,b}}\quad(\text{unmasked, in-range, non-NaN slices; otherwise NaN})`,
      ascii:
        'For each band separately, difference = after - before; output NaN when cell mask is zero, either selected slice index is outside [0,sliceCount), or either selected value is NaN. Here slices 0 and 1 are before/after NIR or SWIR reflectance (DN*0.0001, DN 0 uploaded as NaN); midpoint 0 denotes unchanged reflectance. This is a band difference, not dNBR.',
      symbols: [
        {id: 'difference', role: 'output', label: 'difference', binding: 'wgsl:differenceOut'},
        {id: 'after', role: 'input', label: 'after', binding: 'wgsl:after'},
        {id: 'before', role: 'input', label: 'before', binding: 'wgsl:before'}
      ]
    },
    sources: [
      'modules/experimental/src/gpu-raster/change-detection/gpu-change-detection.ts',
      'modules/experimental/src/gpu-raster/change-detection/change-detection-wgsl.ts'
    ]
  }
};
