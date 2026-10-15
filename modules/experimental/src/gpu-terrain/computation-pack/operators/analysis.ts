// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {RasterOperatorTable} from '../../../gpu-raster/computation-pack/types';

const CELL = {kind: 'cell', label: 'grid cell', indexSpace: '2d'} as const;

/** Terrain pack entries of the `analysis` family, keyed by exported class name. */
export const TERRAIN_ANALYSIS_OPERATORS: RasterOperatorTable = {
  GPUTerrainDerivatives: {
    family: 'analysis',
    descriptor: {scope: 'focal', reduction: 'sum'},
    variables: {dem: 'elevation', slope: 'rate', aspect: 'direction', hillshade: 'fraction'},
    stages: ['support', 'neighbourhood', 'operator', 'presentation'],
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
      {stage: 'operator', name: 'zFactor', label: 'z factor', sweepable: true},
      {stage: 'presentation', name: 'azimuthDegrees', unit: 'deg', sweepable: true},
      {stage: 'presentation', name: 'altitudeDegrees', unit: 'deg', sweepable: true}
    ],
    // Horn (1981) as implemented: two Sobel passes (GPURasterGradient, response 8 per unit ramp),
    // then the fused shade kernel divides by 8 × ground cell size and applies the z factor; rows
    // advancing south flip the sign of the north gradient (ROW_NORTH_SIGN).
    formula: {
      id: 'gpu-terrain.terrain-derivatives.horn',
      tex: String.raw`\term{slope}{\theta} = \arctan\sqrt{p^2 + q^2},\quad p = \frac{\term{zFactor}{\zeta}\,\term{gradientX}{G_x}}{8\,\term{groundCell}{\Delta x}},\quad q = \term{northSign}{\sigma_n}\,\frac{\term{zFactor}{\zeta}\,\term{gradientY}{G_y}}{8\,\term{groundCell}{\Delta y}},\quad \term{hillshade}{H} = \max\!\left(0,\ \frac{(-p,\,-q,\,1)}{\lVert(-p,\,-q,\,1)\rVert}\cdot\hat{\ell}\right)`,
      ascii:
        'slope = atan(sqrt(p^2 + q^2)) in degrees; p = zFactor * Gx / (8 * dx); q = northSign * zFactor * Gy / (8 * dy); Gx, Gy = 3x3 Sobel sums; aspect = atan2(-p, -q) in [0, 360) (-1 when flat); hillshade = max(0, normalize(-p, -q, 1) . light)',
      derivation: 'curated',
      symbols: [
        {id: 'slope', role: 'output', label: 'θ', binding: 'wgsl:slopeValues'},
        {id: 'hillshade', role: 'output', label: 'H', binding: 'wgsl:hillshadeValues'},
        {id: 'gradientX', role: 'intermediate', label: 'G_x', binding: 'wgsl:gradientX'},
        {id: 'gradientY', role: 'intermediate', label: 'G_y', binding: 'wgsl:gradientY'},
        {id: 'zFactor', role: 'parameter', label: 'ζ', binding: 'wgsl:zFactor'},
        {id: 'groundCell', role: 'parameter', label: 'Δx, Δy', binding: 'wgsl:groundCell'},
        {id: 'northSign', role: 'constant', label: 'σ_n', binding: 'wgsl:ROW_NORTH_SIGN'}
      ]
    },
    sources: ['modules/experimental/src/gpu-terrain/terrain-analysis/gpu-terrain-derivatives.ts']
  },
  GPUTerrainContours: {
    family: 'analysis',
    descriptor: {
      scope: 'focal',
      reduction: 'none'
    },
    variables: {
      dem: 'elevation',
      contourOverflow: 'category',
      contourVertices1100: 'measure',
      contourCount1100: 'measure',
      contourVertices1200: 'measure',
      contourCount1200: 'measure',
      contourVertices1300: 'measure',
      contourCount1300: 'measure',
      contourVertices1400: 'measure',
      contourCount1400: 'measure',
      contourVertices1500: 'measure',
      contourCount1500: 'measure',
      contourVertices1600: 'measure',
      contourCount1600: 'measure',
      contourVertices1700: 'measure',
      contourCount1700: 'measure'
    },
    stages: ['support', 'neighbourhood', 'operator'],
    datumKinds: [
      {
        kind: 'cell',
        label: 'grid cell',
        indexSpace: '2d'
      }
    ],
    neighbourhood: 'window',
    guards: [],
    choices: [
      {
        stage: 'operator',
        name: 'levelInterval',
        sweepable: false,
        unit: 'm'
      }
    ],
    formula: {
      id: 'gpu-terrain.terrain-contours.interpolation',
      tex: String.raw`\term{fraction}{u}=\operatorname{clamp}\!\left(\frac{\term{level}{\ell}-\term{first}{h_a}}{\term{second}{h_b}-\term{first}{h_a}},0,1\right),\quad \term{vertices}{p}=p_a+u(p_b-p_a)`,
      ascii:
        'Each crossed edge uses u=clamp((level-first)/(second-first),0,1); if differences overflow, divide all heights by max(1,abs(level),abs(first),abs(second)); zero scaled denominator gives u=0.5. Positions are local pixel centers (+0.5 for area pixels). Equal heights count as high; flat or invalid-corner cells emit nothing; saddle pairing uses the asymptotic determinant with high-diagonal tie; counts clamp to capacity and overflow reports truncation.',
      derivation: 'curated',
      symbols: [
        {
          id: 'fraction',
          role: 'intermediate',
          label: 'u',
          binding: 'wgsl:directNumerator'
        },
        {
          id: 'level',
          role: 'parameter',
          label: 'level',
          binding: 'wgsl:contourLevel'
        },
        {
          id: 'first',
          role: 'input',
          label: 'first height',
          binding: 'wgsl:first'
        },
        {
          id: 'second',
          role: 'input',
          label: 'second height',
          binding: 'wgsl:second'
        },
        {
          id: 'vertices',
          role: 'output',
          label: 'position',
          binding: 'wgsl:outputVertices'
        }
      ]
    },
    sources: [
      'modules/experimental/src/gpu-terrain/terrain-analysis/gpu-terrain-contours.ts',
      'modules/experimental/src/gpu-raster/gpu-raster-contours.ts'
    ]
  },
  GPUTerrainViewshed: {
    family: 'analysis',
    descriptor: {
      scope: 'pairwise',
      reduction: 'select'
    },
    variables: {
      dem: 'elevation',
      viewshedVisibility: 'category'
    },
    stages: ['support', 'neighbourhood', 'operator'],
    datumKinds: [
      {
        kind: 'cell',
        label: 'grid cell',
        indexSpace: '2d'
      }
    ],
    neighbourhood: 'ray',
    guards: ['metric-cell-size'],
    choices: [
      {
        stage: 'support',
        name: 'cellSizeMode',
        sweepable: false,
        alternatives: ['uniform', 'web-mercator', 'geographic']
      },
      {
        stage: 'operator',
        name: 'observerHeight',
        sweepable: false,
        unit: 'm'
      },
      {
        stage: 'operator',
        name: 'curvatureCoefficient',
        sweepable: false,
        unit: '1/m'
      }
    ],
    formula: {
      id: 'gpu-terrain.terrain-viewshed.tangent',
      tex: String.raw`\term{maximum}{S}=\max_k\frac{\term{sample}{h_k}-\term{curvature}{c}\,\term{sampleDistance}{s_k}^{2}-\term{eye}{z_o}}{\term{sampleDistance}{s_k}},\quad \term{target}{T}=\frac{\term{targetTop}{z_t}-c\,\term{distance}{D}^{2}-z_o}{D},\quad \term{code}{v}=\begin{cases}0&S>T+\term{band}{\beta}\\1&S\le T-\beta\\4&\text{otherwise}\end{cases}`,
      ascii:
        'D=length((target-observer)*cellSize); S=max((bilinearHeight-c*s*s-eye)/s) at k=1..ceil(max(abs(delta)))-1, skipping invalid samples and stopping beyond D-max(ignoreDistance,ignoreFraction*D); T=(targetTop-c*D*D-eye)/D; beta=(toleranceMeters+tolerancePerKilometer*D/1000)/D; hidden=0 if S>T+beta, visible=1 if S<=T-beta, otherwise marginal=4; invalid endpoints give noData=3, D>maximumDistance>0 gives outOfRange=2, D=0 gives visible; no tested samples give visible.',
      derivation: 'curated',
      symbols: [
        {
          id: 'maximum',
          role: 'intermediate',
          label: 'maximum sample tangent',
          binding: 'wgsl:maxSlope'
        },
        {
          id: 'sample',
          role: 'input',
          label: 'bilinear sample',
          binding: 'wgsl:elevationSample'
        },
        {
          id: 'curvature',
          role: 'parameter',
          label: 'curvature coefficient',
          binding: 'wgsl:curvature'
        },
        {
          id: 'sampleDistance',
          role: 'intermediate',
          label: 'sample distance',
          binding: 'wgsl:sampleDistance'
        },
        {
          id: 'eye',
          role: 'input',
          label: 'eye elevation',
          binding: 'wgsl:eye'
        },
        {
          id: 'target',
          role: 'intermediate',
          label: 'target tangent',
          binding: 'wgsl:targetSlope'
        },
        {
          id: 'targetTop',
          role: 'input',
          label: 'target elevation',
          binding: 'wgsl:targetTop'
        },
        {
          id: 'distance',
          role: 'intermediate',
          label: 'target distance',
          binding: 'wgsl:targetDistance'
        },
        {
          id: 'code',
          role: 'output',
          label: 'visibility code',
          binding: 'wgsl:code'
        },
        {
          id: 'band',
          role: 'parameter',
          label: 'tolerance band',
          binding: 'wgsl:band'
        }
      ]
    },
    sources: [
      'modules/experimental/src/gpu-terrain/terrain-analysis/gpu-terrain-viewshed.ts',
      'modules/experimental/src/gpu-terrain/terrain-analysis/terrain-sight-line.ts'
    ]
  },
  GPUTerrainLineOfSight: {
    family: 'analysis',
    descriptor: {
      scope: 'pairwise',
      reduction: 'select'
    },
    variables: {
      dem: 'elevation',
      sightLinePairs: 'measure',
      sightLineOffsets: 'measure',
      sightLineCoordinates: 'measure',
      sightLineVisibility: 'category',
      sightLineClearance: 'score',
      sightLineVisibilityGeometry: 'category',
      sightLineClearanceGeometry: 'score'
    },
    stages: ['support', 'neighbourhood', 'operator'],
    datumKinds: [
      {
        kind: 'pair',
        label: 'observer-target pair',
        indexSpace: '1d'
      }
    ],
    neighbourhood: 'ray',
    guards: ['metric-cell-size'],
    choices: [
      {
        stage: 'support',
        name: 'cellSizeMode',
        sweepable: false,
        alternatives: ['uniform', 'web-mercator', 'geographic']
      },
      {
        stage: 'operator',
        name: 'observerHeight',
        sweepable: false,
        unit: 'm'
      },
      {
        stage: 'operator',
        name: 'curvatureCoefficient',
        sweepable: false,
        unit: '1/m'
      }
    ],
    formula: {
      id: 'gpu-terrain.terrain-line-of-sight.tangent',
      tex: String.raw`\term{maximum}{S}=\max_k\frac{\term{sample}{h_k}-\term{curvature}{c}\,\term{sampleDistance}{s_k}^{2}-\term{eye}{z_o}}{\term{sampleDistance}{s_k}},\quad \term{target}{T}=\frac{\term{targetTop}{z_t}-c\,\term{distance}{D}^{2}-z_o}{D},\quad \term{code}{v}=\begin{cases}0&S>T+\term{band}{\beta}\\1&S\le T-\beta\\4&\text{otherwise}\end{cases},\quad \term{clearance}{C}=(T-S)D`,
      ascii:
        'D=length((target-observer)*cellSize); S=max((bilinearHeight-c*s*s-eye)/s) at k=1..ceil(max(abs(delta)))-1, skipping invalid samples and stopping beyond D-max(ignoreDistance,ignoreFraction*D); T=(targetTop-c*D*D-eye)/D; beta=(toleranceMeters+tolerancePerKilometer*D/1000)/D; hidden=0 if S>T+beta, visible=1 if S<=T-beta, otherwise marginal=4; invalid endpoints give noData=3, D>maximumDistance>0 gives outOfRange=2, D=0 gives visible; no tested samples give visible. Clearance=(T-S)*D, 3.4028234663852886e38 for coincident points or no tested sample, NaN for noData or outOfRange; requesting clearance disables hidden-ray early exit.',
      derivation: 'curated',
      symbols: [
        {
          id: 'maximum',
          role: 'intermediate',
          label: 'maximum sample tangent',
          binding: 'wgsl:maxSlope'
        },
        {
          id: 'sample',
          role: 'input',
          label: 'bilinear sample',
          binding: 'wgsl:elevationSample'
        },
        {
          id: 'curvature',
          role: 'parameter',
          label: 'curvature coefficient',
          binding: 'wgsl:curvature'
        },
        {
          id: 'sampleDistance',
          role: 'intermediate',
          label: 'sample distance',
          binding: 'wgsl:sampleDistance'
        },
        {
          id: 'eye',
          role: 'input',
          label: 'eye elevation',
          binding: 'wgsl:eye'
        },
        {
          id: 'target',
          role: 'intermediate',
          label: 'target tangent',
          binding: 'wgsl:targetSlope'
        },
        {
          id: 'targetTop',
          role: 'input',
          label: 'target elevation',
          binding: 'wgsl:targetTop'
        },
        {
          id: 'distance',
          role: 'intermediate',
          label: 'target distance',
          binding: 'wgsl:targetDistance'
        },
        {
          id: 'code',
          role: 'output',
          label: 'visibility code',
          binding: 'wgsl:code'
        },
        {
          id: 'band',
          role: 'parameter',
          label: 'tolerance band',
          binding: 'wgsl:band'
        },
        {
          id: 'clearance',
          role: 'output',
          label: 'clearance (m)',
          binding: 'wgsl:clearance'
        }
      ]
    },
    sources: [
      'modules/experimental/src/gpu-terrain/terrain-analysis/gpu-terrain-line-of-sight.ts',
      'modules/experimental/src/gpu-terrain/terrain-analysis/terrain-sight-line.ts'
    ]
  },
  GPUTerrainCumulativeViewshed: {
    family: 'analysis',
    descriptor: {
      scope: 'pairwise',
      reduction: 'count'
    },
    variables: {
      dem: 'elevation',
      cumulativeObservers: 'measure',
      visibleCount: 'measure'
    },
    stages: ['support', 'neighbourhood', 'operator'],
    datumKinds: [
      {
        kind: 'cell',
        label: 'grid cell',
        indexSpace: '2d'
      }
    ],
    neighbourhood: 'ray',
    guards: ['metric-cell-size'],
    choices: [
      {
        stage: 'support',
        name: 'cellSizeMode',
        sweepable: false,
        alternatives: ['uniform', 'web-mercator', 'geographic']
      },
      {
        stage: 'operator',
        name: 'observerHeight',
        sweepable: false,
        unit: 'm'
      },
      {
        stage: 'operator',
        name: 'curvatureCoefficient',
        sweepable: false,
        unit: '1/m'
      }
    ],
    formula: {
      id: 'gpu-terrain.terrain-cumulative-viewshed.count',
      tex: String.raw`\term{count}{N(p)}=\sum_{o\in\term{observers}{O}}\mathbf{1}\!\left[\term{result}{v(o,p)}=\term{visible}{1}\right]`,
      ascii:
        'Start counts at zero; atomically add 1 per (observer,cell) whose shared curvature-aware sight-line code is VISIBLE (1). Marginal, hidden, invalid observers/targets and out-of-range pairs add zero; coincident valid observer cells count as visible. Target ground is the raw pixel elevation.',
      derivation: 'curated',
      symbols: [
        {
          id: 'count',
          role: 'output',
          label: 'visible observers',
          binding: 'wgsl:visibleCount'
        },
        {
          id: 'observers',
          role: 'input',
          label: 'observers',
          binding: 'wgsl:observers'
        },
        {
          id: 'result',
          role: 'intermediate',
          label: 'sight-line result',
          binding: 'wgsl:result'
        },
        {
          id: 'visible',
          role: 'constant',
          label: 'VISIBLE',
          binding: 'wgsl:VISIBLE'
        }
      ]
    },
    sources: [
      'modules/experimental/src/gpu-terrain/terrain-analysis/gpu-terrain-cumulative-viewshed.ts',
      'modules/experimental/src/gpu-terrain/terrain-analysis/terrain-sight-line.ts'
    ]
  }
};
