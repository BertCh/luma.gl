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

/** Terrain pack entries of the `geomorphometry` family, keyed by exported class name. */
export const TERRAIN_GEOMORPHOMETRY_OPERATORS: RasterOperatorTable = {
  GPUTerrainCurvature: {
    family: 'geomorphometry',
    descriptor: {scope: 'focal', reduction: 'sum'},
    variables: {dem: 'elevation', curvatureProfile: 'score', curvaturePlan: 'score'},
    stages: ['support', 'neighbourhood', 'operator'],
    datumKinds: [CELL],
    neighbourhood: 'stencil-3x3',
    guards: ['metric-cell-size'],
    choices: [
      CELL_SIZE_MODE,
      {
        stage: 'neighbourhood',
        name: 'method',
        alternatives: ['evans-young', 'zevenbergen-thorne', 'florinsky'],
        sweepable: false
      },
      {stage: 'operator', name: 'zFactor', label: 'z factor', sweepable: true},
      {stage: 'operator', name: 'flatGradient', sweepable: false}
    ],
    // Evans-Young as implemented: window heights are differenced against the centre first, the five
    // partials come from 3x3 least squares divided by the ground cell sizes (north is up), and the
    // WhiteboxTools sign makes convex ground positive. Flat gradients give 0 for profile and plan.
    formula: {
      id: 'gpu-terrain.terrain-curvature.evans-young',
      tex: String.raw`\term{profile}{\kappa_{\mathrm{prof}}} = -\frac{\term{p}{p}^2\,\term{r}{r} + 2\,p\,\term{q}{q}\,\term{s}{s} + q^2\,\term{t}{t}}{\term{gradientSquared}{G}\,\term{weight}{W}^{3/2}},\quad \term{plan}{\kappa_{\mathrm{plan}}} = -\frac{q^2 r - 2\,p\,q\,s + p^2 t}{G^{3/2}},\quad G = p^2 + q^2,\ W = 1 + G`,
      ascii:
        'Evans-Young 3x3 least squares on heights minus the centre, z factor and ground cell sizes wx, wy: p = (z3+z6+z9-z1-z4-z7)/(6 wx); q = (z1+z2+z3-z7-z8-z9)/(6 wy); r = (z1+z3+z4+z6+z7+z9-2(z2+z8))/(3 wx^2); t = (z1+z2+z3+z7+z8+z9-2(z4+z6))/(3 wy^2); s = (z3+z7-z1-z9)/(4 wx wy) with z1 NW, z2 N, z3 NE, z4 W, z6 E, z7 SW, z8 S, z9 SE. profile = -(p^2 r + 2pqs + q^2 t)/(G W^1.5); plan = -(q^2 r - 2pqs + p^2 t)/G^1.5, so a convex hill top is positive. G <= flatGradient^2 gives 0 for both; an invalid or non-finite window sample (or a window leaving the raster with borderMode nodata) gives NaN.',
      derivation: 'curated',
      symbols: [
        {id: 'profile', role: 'output', label: 'profile curvature', binding: 'wgsl:profile'},
        {id: 'plan', role: 'output', label: 'plan curvature', binding: 'wgsl:plan'},
        {id: 'p', role: 'intermediate', label: 'p', binding: 'wgsl:p'},
        {id: 'q', role: 'intermediate', label: 'q', binding: 'wgsl:q'},
        {id: 'r', role: 'intermediate', label: 'r', binding: 'wgsl:r'},
        {id: 's', role: 'intermediate', label: 's', binding: 'wgsl:s'},
        {id: 't', role: 'intermediate', label: 't', binding: 'wgsl:t'},
        {
          id: 'gradientSquared',
          role: 'intermediate',
          label: 'G',
          binding: 'wgsl:gradientSquared'
        },
        {id: 'weight', role: 'intermediate', label: 'W', binding: 'wgsl:weight'}
      ]
    },
    sources: [
      'modules/experimental/src/gpu-terrain/terrain-curvature/gpu-terrain-curvature.ts',
      'modules/experimental/src/gpu-terrain/terrain-curvature/terrain-geomorphometry-utils.ts'
    ]
  },
  GPUGeomorphons: {
    family: 'geomorphometry',
    descriptor: {scope: 'focal', reduction: 'select'},
    variables: {dem: 'elevation', geomorphonForms: 'category'},
    stages: ['support', 'neighbourhood', 'operator'],
    datumKinds: [CELL],
    neighbourhood: 'ray',
    guards: ['metric-cell-size'],
    choices: [
      CELL_SIZE_MODE,
      {stage: 'neighbourhood', name: 'searchRadius', unit: 'cell', sweepable: true},
      {stage: 'neighbourhood', name: 'skipRadius', unit: 'cell', sweepable: false},
      {stage: 'operator', name: 'flatThresholdDegrees', unit: 'deg', sweepable: true},
      {
        stage: 'operator',
        name: 'comparison',
        alternatives: ['anglev1', 'anglev2', 'anglev2-distance'],
        sweepable: false
      }
    ],
    // Jasiewicz-Stepinski as implemented (GRASS anglev1): per direction the zenith is the sample with
    // the largest height per step and the nadir the smallest; the digit is decided by which of the
    // two exceeds the flatness threshold more steeply, then the counts of minus and plus digits index
    // the landform table.
    formula: {
      id: 'gpu-terrain.geomorphons.ternary-pattern',
      tex: String.raw`\term{digit}{d_i} = \begin{cases} 0 & \max\!\left(\left|\term{zenithHeight}{\Delta_Z}\right| - \term{tangent}{\tau}\,\term{zenithCount}{k_Z}\,\term{step}{s},\ \left|\term{nadirHeight}{\Delta_N}\right| - \tau\,\term{nadirCount}{k_N}\,s\right) \le 0 \\ \operatorname{sgn}\!\left(\frac{|\Delta_Z|}{k_Z} - \frac{|\Delta_N|}{k_N}\right) & \text{otherwise} \end{cases},\quad \term{form}{F} = \term{formTable}{T}\left[\,9\,\term{minusCount}{m} + \term{plusCount}{p}\,\right]`,
      ascii:
        'For each of 8 directions i (NE, N, NW, W, SW, S, SE, E) walk k = skipRadius+1.. while k*step < 0.99999*searchRadius*cellSizeY and stay inside the raster, ignoring invalid samples; step is the ground size of one step. Height h_k = zFactor*(z_k - z_centre). Zenith = sample maximising h_k/k (height Delta_Z at count k_Z); nadir = sample minimising h_k/k (Delta_N at k_N). With tau = tan(flatThresholdDegrees) a side is over the threshold when |Delta| > tau * k * step (the flat distance, when set and smaller, replaces k*step). anglev1: digit 0 unless one side is over; then digit +1 if |Delta_Z|/k_Z > |Delta_N|/k_N, -1 if smaller, 0 on an exact tie (anglev2 variants resolve ties to +1 or by distance). A direction with an invalid first neighbour or no valid sample has digit 0. m = number of -1 digits, p = number of +1 digits; form = FORM_TABLE[9m + p]: 1 flat, 2 peak, 3 ridge, 4 shoulder, 5 spur, 6 slope, 7 hollow, 8 footslope, 9 valley, 10 pit. Invalid cells and cells within skipRadius+1 of the border get 0.',
      derivation: 'curated',
      symbols: [
        {id: 'digit', role: 'intermediate', label: 'direction digit', binding: 'wgsl:digit'},
        {
          id: 'zenithHeight',
          role: 'intermediate',
          label: 'zenith height',
          binding: 'wgsl:zenithHeight'
        },
        {
          id: 'nadirHeight',
          role: 'intermediate',
          label: 'nadir height',
          binding: 'wgsl:nadirHeight'
        },
        {
          id: 'zenithCount',
          role: 'intermediate',
          label: 'zenith steps',
          binding: 'wgsl:zenithCount'
        },
        {id: 'nadirCount', role: 'intermediate', label: 'nadir steps', binding: 'wgsl:nadirCount'},
        {id: 'tangent', role: 'parameter', label: 'tan(flat threshold)', binding: 'wgsl:tangent'},
        {id: 'step', role: 'intermediate', label: 'ground step', binding: 'wgsl:step'},
        {id: 'minusCount', role: 'intermediate', label: 'minus digits', binding: 'wgsl:minusCount'},
        {id: 'plusCount', role: 'intermediate', label: 'plus digits', binding: 'wgsl:plusCount'},
        {id: 'formTable', role: 'constant', label: 'landform table', binding: 'wgsl:FORM_TABLE'},
        {id: 'form', role: 'output', label: 'landform code', binding: 'wgsl:outputForm'}
      ]
    },
    sources: ['modules/experimental/src/gpu-terrain/geomorphons/gpu-geomorphons.ts']
  },
  GPUTerrainRuggedness: {
    family: 'geomorphometry',
    descriptor: {scope: 'focal', reduction: 'sum'},
    variables: {dem: 'elevation', terrainRuggednessIndex: 'rate'},
    stages: ['neighbourhood', 'operator'],
    datumKinds: [CELL],
    neighbourhood: 'stencil-3x3',
    guards: [],
    choices: [
      {
        stage: 'operator',
        name: 'terrainRuggednessAlgorithm',
        alternatives: ['riley', 'wilson'],
        sweepable: false
      },
      {
        stage: 'neighbourhood',
        name: 'edgeMode',
        alternatives: ['nodata', 'extrapolate'],
        sweepable: false
      }
    ],
    // Riley (default, gdaldem 3.3+) as implemented: differences are taken against the centre first;
    // the centre itself adds 0 to the sum of squares. No cell size or z factor is involved.
    formula: {
      id: 'gpu-terrain.terrain-ruggedness.riley',
      tex: String.raw`\term{ruggedness}{\mathrm{TRI}} = \sqrt{\sum_{k=1}^{9}\left(\term{windowValue}{z_k} - \term{centerValue}{z_c}\right)^2}`,
      ascii:
        'Riley: TRI = sqrt(sum over the 3x3 window of (z_k - z_centre)^2), in elevation units (the centre term is 0). terrainRuggednessAlgorithm wilson uses sum(|z_k - z_centre|)/8 instead. With edgeMode nodata the outer ring and any window with an invalid sample are NaN; extrapolate (gdaldem -compute_edges) extends outside samples linearly (2a - b) and replaces invalid neighbours by the centre.',
      derivation: 'curated',
      symbols: [
        {id: 'ruggedness', role: 'output', label: 'TRI', binding: 'wgsl:ruggedness'},
        {id: 'windowValue', role: 'input', label: 'window height', binding: 'wgsl:windowValues'},
        {id: 'centerValue', role: 'input', label: 'centre height', binding: 'wgsl:centerValue'}
      ]
    },
    sources: ['modules/experimental/src/gpu-terrain/topographic-position/gpu-terrain-ruggedness.ts']
  },
  GPUTerrainVectorRuggedness: {
    family: 'geomorphometry',
    descriptor: {scope: 'focal', reduction: 'sum'},
    variables: {dem: 'elevation', vectorRuggednessMeasure: 'fraction'},
    stages: ['support', 'neighbourhood', 'operator'],
    datumKinds: [CELL],
    neighbourhood: 'window',
    guards: ['metric-cell-size'],
    choices: [
      CELL_SIZE_MODE,
      {stage: 'neighbourhood', name: 'radius', unit: 'cell', sweepable: true},
      {stage: 'operator', name: 'zFactor', label: 'z factor', sweepable: true},
      {
        stage: 'neighbourhood',
        name: 'borderMode',
        alternatives: ['clamp', 'nodata'],
        sweepable: false
      }
    ],
    // Sappington VRM as implemented: a first kernel turns Horn gradients (1-2-1 weights over 8 ground
    // cell sizes, row-direction sign) into unit normals; the window kernel sums the normals of valid
    // cells inside the clipped square window.
    formula: {
      id: 'gpu-terrain.terrain-vector-ruggedness.sappington',
      tex: String.raw`\term{vrm}{\mathrm{VRM}} = \operatorname{clamp}\!\left(1 - \frac{\left\lVert \term{vectorSum}{\textstyle\sum_{j} \hat{n}_j} \right\rVert}{\term{count}{N}},\ 0,\ 1\right),\quad \hat{n} = \frac{(-\term{eastGradient}{p},\ -\term{northGradient}{q},\ 1)}{\lVert(-p,\ -q,\ 1)\rVert}`,
      ascii:
        'Unit normal per cell from Horn gradients p = zFactor*(east - west 1-2-1 sums)/(8*wx), q = northSign*zFactor*(south - north 1-2-1 sums)/(8*wy): n = (-p, -q, 1)/|(-p, -q, 1)|. VRM = clamp(1 - |sum of n over the (2r+1)^2 window, clipped to the raster and to cells with a valid normal| / N, 0, 1), N = number of those cells. radius is a per-frame setting in [1, maximumRadius]. An invalid centre normal gives NaN. borderMode clamp repeats edge samples for the normals; nodata invalidates border normals.',
      derivation: 'curated',
      symbols: [
        {id: 'vrm', role: 'output', label: 'VRM', binding: 'wgsl:ruggedness'},
        {id: 'vectorSum', role: 'intermediate', label: 'normal sum', binding: 'wgsl:vectorSum'},
        {id: 'count', role: 'intermediate', label: 'valid normals', binding: 'wgsl:count'},
        {id: 'eastGradient', role: 'intermediate', label: 'p', binding: 'wgsl:eastGradient'},
        {id: 'northGradient', role: 'intermediate', label: 'q', binding: 'wgsl:northGradient'}
      ]
    },
    sources: [
      'modules/experimental/src/gpu-terrain/topographic-position/gpu-terrain-vector-ruggedness.ts'
    ]
  },
  GPUTerrainTopographicPosition: {
    family: 'geomorphometry',
    descriptor: {scope: 'focal', reduction: 'weighted-mean'},
    variables: {dem: 'elevation', topographicPositionIndex: 'score'},
    stages: ['neighbourhood', 'operator'],
    datumKinds: [CELL],
    neighbourhood: 'window',
    guards: [],
    choices: [
      {stage: 'neighbourhood', name: 'radius', unit: 'cell', sweepable: true},
      {stage: 'neighbourhood', name: 'innerRadius', unit: 'cell', sweepable: false},
      {stage: 'representation', name: 'quantum', unit: 'm', sweepable: false}
    ],
    // As implemented: elevations are quantized to integers (quantum 1/256), summed in an exact 64-bit
    // summed-area table, and the annulus is the clipped outer box minus the clipped inner box. The
    // difference to the centre is formed in integer arithmetic before the single conversion to float.
    formula: {
      id: 'gpu-terrain.terrain-topographic-position.annulus-mean',
      tex: String.raw`\term{tpi}{\mathrm{TPI}} = -\,\frac{\term{quantum}{Q}\;\term{centredSum}{\textstyle\sum_{a}\left(q_a - q_c\right)}}{\term{annulusCount}{N}}`,
      ascii:
        'q = round(z/Q) with Q = quantum (default 1/256); the annulus A holds cells with Chebyshev distance in (innerRadius, radius] from the centre, clipped to the raster and to valid cells (N of them); innerRadius 0 excludes only the centre. TPI = z_c - mean(z over A) = -Q * sum_A (q_a - q_c) / N, with the sum taken exactly modulo 2^64 from the summed-area table. NaN where the centre is invalid or N = 0. One output plane per scale.',
      derivation: 'curated',
      symbols: [
        {id: 'tpi', role: 'output', label: 'TPI', binding: 'wgsl:tpi'},
        {id: 'quantum', role: 'constant', label: 'Q', binding: 'wgsl:QUANTUM'},
        {
          id: 'centredSum',
          role: 'intermediate',
          label: 'centred sum',
          binding: 'wgsl:getCentredSum'
        },
        {id: 'annulusCount', role: 'intermediate', label: 'N', binding: 'wgsl:annulus'}
      ]
    },
    sources: [
      'modules/experimental/src/gpu-terrain/topographic-position/gpu-terrain-topographic-position.ts',
      'modules/experimental/src/gpu-terrain/topographic-position/terrain-summed-area-table.ts'
    ]
  },
  GPUTerrainWeissLandforms: {
    family: 'geomorphometry',
    descriptor: {scope: 'focal', reduction: 'select', scopeSecondary: 'global'},
    variables: {
      dem: 'elevation',
      weissLandforms: 'category',
      weissSmallPosition: 'score',
      weissLargePosition: 'score'
    },
    stages: ['support', 'neighbourhood', 'operator'],
    datumKinds: [CELL],
    neighbourhood: 'window',
    guards: ['metric-cell-size'],
    choices: [
      CELL_SIZE_MODE,
      {stage: 'neighbourhood', name: 'smallRadius', unit: 'cell', sweepable: true},
      {stage: 'neighbourhood', name: 'largeRadius', unit: 'cell', sweepable: true},
      {
        stage: 'operator',
        name: 'standardization',
        alternatives: ['global', 'local'],
        sweepable: false
      },
      {stage: 'operator', name: 'standardThreshold', sweepable: true},
      {stage: 'operator', name: 'slopeThresholdDegrees', unit: 'deg', sweepable: true}
    ],
    // Weiss (2001) as implemented with global standardization: TPI of two scales is z-scored over every
    // valid cell, classed at +-threshold, and the 3x3 class pair indexes ten landforms; only the
    // mid/mid pair is split by Horn slope.
    formula: {
      id: 'gpu-terrain.terrain-weiss-landforms.classes',
      tex: String.raw`\term{small}{s} = \frac{\term{position}{\mathrm{TPI}_3} - \term{mean}{\mu}}{\term{deviation}{\sigma}},\quad c(x) = \begin{cases} 0 & x \le -\term{threshold}{t} \\ 2 & x \ge t \\ 1 & \text{otherwise} \end{cases},\quad \term{landform}{L} = \begin{cases} 1 + c_l & c_s = 0 \\ \begin{cases} 4 & c_l = 0 \\ 5\text{ if }\term{slope}{\theta} \le \term{slopeThreshold}{\beta}\text{ else }6 & c_l = 1 \\ 7 & c_l = 2 \end{cases} & c_s = 1 \\ 8 + c_l & c_s = 2 \end{cases}`,
      ascii:
        'Per scale (small and large, radii in cells) the TPI plane comes from the summed-area table; global standardization gives z = (TPI - mean)/standardDeviation over all valid cells of that scale (population statistics; z = 0 when the deviation is 0; local uses the window DEV instead). Class c(z): 0 if z <= -standardThreshold, 2 if z >= standardThreshold, else 1. With cs from the small scale and cl from the large one: cs 0 -> 1 + cl (canyon, midslope drainage, upland drainage); cs 1 -> cl 0: U-shaped valley 4, cl 1: plain 5 if Horn slope <= slopeThresholdDegrees else open slope 6, cl 2: upper slope 7; cs 2 -> 8 + cl (local ridge, midslope ridge, mountain top). Cells with an invalid elevation, position or slope get 0.',
      derivation: 'curated',
      symbols: [
        {id: 'small', role: 'intermediate', label: 'standardized small TPI', binding: 'wgsl:small'},
        {id: 'position', role: 'input', label: 'TPI planes', binding: 'wgsl:positionValues'},
        {id: 'mean', role: 'intermediate', label: 'scale mean', binding: 'wgsl:mean'},
        {
          id: 'deviation',
          role: 'intermediate',
          label: 'scale deviation',
          binding: 'wgsl:deviation'
        },
        {
          id: 'threshold',
          role: 'parameter',
          label: 'standard threshold',
          binding: 'wgsl:threshold'
        },
        {id: 'slope', role: 'intermediate', label: 'slope', binding: 'wgsl:slope'},
        {
          id: 'slopeThreshold',
          role: 'parameter',
          label: 'slope threshold',
          binding: 'wgsl:slopeThreshold'
        },
        {id: 'landform', role: 'output', label: 'landform code', binding: 'wgsl:landform'}
      ]
    },
    sources: [
      'modules/experimental/src/gpu-terrain/topographic-position/gpu-terrain-weiss-landforms.ts'
    ]
  }
};
