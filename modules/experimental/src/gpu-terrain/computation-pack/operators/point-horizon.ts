// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {RasterOperatorTable} from '../../../gpu-raster/computation-pack/types';

/** Terrain pack entries of the `point-horizon` family, keyed by exported class name. */
export const TERRAIN_POINT_HORIZON_OPERATORS: RasterOperatorTable = {
  GPUPointHorizonProfile: {
    family: 'point-horizon',
    descriptor: {
      scope: 'pairwise',
      reduction: 'select'
    },
    variables: {
      dem: 'elevation',
      horizonObservers: 'measure',
      pointHorizonSkylineAngle: 'score',
      pointHorizonTangent: 'score'
    },
    stages: ['support', 'neighbourhood', 'operator'],
    datumKinds: [
      {
        kind: 'ray',
        label: 'observer and azimuth bin',
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
        stage: 'support',
        name: 'projection',
        sweepable: false,
        alternatives: ['planar', 'web-mercator']
      },
      {
        stage: 'neighbourhood',
        name: 'maximumDistance',
        sweepable: false,
        unit: 'm'
      },
      {
        stage: 'operator',
        name: 'traversal',
        sweepable: false,
        alternatives: ['march', 'pyramid']
      },
      {
        stage: 'operator',
        name: 'curvatureCoefficient',
        sweepable: false,
        unit: '1/m'
      },
      {
        stage: 'neighbourhood',
        name: 'azimuthCount',
        sweepable: false
      }
    ],
    formula: {
      id: 'gpu-terrain.point-horizon-profile.maximum',
      tex: String.raw`\term{tangent}{t_*}=\max_{\term{distance}{d}\in L}\left(\frac{\term{relative}{h-h_0}}{d}-\term{curvature}{c}d\right),\quad\term{angle}{\alpha}=\frac{180}{\pi}\arctan(t_*)`,
      ascii:
        'March the exact power-of-two distance lattice; keep tBest=max(hr/d-c*d) over valid bilinear relative heights, ending at the first out-of-grid footprint or distance cap. Skyline angle=atanAccurate(tBest)*180/pi. No valid samples give tangent=-3.4028234663852886e38 and angle=-90 degrees; invalid observer or projection-tag mismatch gives NaN. Output index=observer*azimuthSpan+bin.',
      derivation: 'curated',
      symbols: [
        {
          id: 'tangent',
          role: 'output',
          label: 'maximum tangent',
          binding: 'wgsl:tangentOutput'
        },
        {
          id: 'distance',
          role: 'intermediate',
          label: 'sample distance',
          binding: 'wgsl:d'
        },
        {
          id: 'relative',
          role: 'input',
          label: 'height relative to eye',
          binding: 'wgsl:hr'
        },
        {
          id: 'curvature',
          role: 'parameter',
          label: 'curvature',
          binding: 'wgsl:curvature'
        },
        {
          id: 'angle',
          role: 'output',
          label: 'skyline angle',
          binding: 'wgsl:skylineAngleOutput'
        }
      ]
    },
    sources: [
      'modules/experimental/src/gpu-terrain/point-horizon/gpu-point-horizon-profile.ts',
      'modules/experimental/src/gpu-terrain/point-horizon/point-horizon-march.ts'
    ]
  },
  GPUPointHorizonVisibility: {
    family: 'point-horizon',
    descriptor: {
      scope: 'pairwise',
      reduction: 'select'
    },
    variables: {
      dem: 'elevation',
      horizonObservers: 'measure',
      horizonTargets: 'measure',
      pointHorizonVisibility: 'category',
      pointHorizonDetails: 'score'
    },
    stages: ['support', 'neighbourhood', 'operator'],
    datumKinds: [
      {
        kind: 'target',
        label: 'target and observer',
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
        stage: 'support',
        name: 'projection',
        sweepable: false,
        alternatives: ['planar', 'web-mercator']
      },
      {
        stage: 'neighbourhood',
        name: 'maximumDistance',
        sweepable: false,
        unit: 'm'
      },
      {
        stage: 'operator',
        name: 'traversal',
        sweepable: false,
        alternatives: ['march', 'pyramid']
      },
      {
        stage: 'operator',
        name: 'curvatureCoefficient',
        sweepable: false,
        unit: '1/m'
      },
      {
        stage: 'operator',
        name: 'toleranceDegrees',
        sweepable: false,
        unit: 'deg'
      },
      {
        stage: 'operator',
        name: 'sigmaZ',
        sweepable: false,
        unit: 'm'
      },
      {
        stage: 'operator',
        name: 'skylineToleranceDegrees',
        sweepable: false,
        unit: 'deg'
      },
      {
        stage: 'neighbourhood',
        name: 'targetIgnoreDistance',
        sweepable: false,
        unit: 'm'
      }
    ],
    formula: {
      id: 'gpu-terrain.point-horizon-visibility.classification',
      tex: String.raw`\term{code}{v}=\begin{cases}0&\term{target}{\alpha_P}<\term{occluder}{\alpha_Q}-\term{tolerance}{\tau}\\4&|\alpha_P-\alpha_Q|<\tau\\1&\text{otherwise}\end{cases},\quad\term{skylineFlag}{s}=\mathbf{1}[v\ne0\land\alpha_P\ge\term{skyline}{\alpha_S}-\term{settings}{\tau_S}]`,
      ascii:
        'tP=(targetGround+targetHeight-eye)/D-c*D; alphaP=atanAccurate(tP) in degrees. March full bearing for skyline and retain the maximum before q=D-max(ignoreDistance,ignoreFraction*D) for occluder; default ignoreDistance is one ground pixel. tau=toleranceDegrees+sigmaZ/D*180/pi; hidden if alphaP<alphaQ-tau, marginal if abs(alphaP-alphaQ)<tau, otherwise visible. onSkyline=(code!=hidden && alphaP>=alphaSky-skylineToleranceDegrees). Empty occluder/skyline angles=-90; invalid endpoints/index/tag give noData; beyond cap gives outOfRange; coincident points give visible; all three exceptional cases have NaN details.',
      derivation: 'curated',
      symbols: [
        {
          id: 'code',
          role: 'output',
          label: 'visibility',
          binding: 'wgsl:code'
        },
        {
          id: 'target',
          role: 'intermediate',
          label: 'target angle',
          binding: 'wgsl:targetAngle'
        },
        {
          id: 'occluder',
          role: 'intermediate',
          label: 'occluder angle',
          binding: 'wgsl:occluderAngle'
        },
        {
          id: 'tolerance',
          role: 'parameter',
          label: 'angular tolerance',
          binding: 'wgsl:tolerance'
        },
        {
          id: 'skylineFlag',
          role: 'output',
          label: 'on skyline',
          binding: 'wgsl:onSkyline'
        },
        {
          id: 'skyline',
          role: 'intermediate',
          label: 'skyline angle',
          binding: 'wgsl:skylineAngle'
        },
        {
          id: 'settings',
          role: 'parameter',
          label: 'skyline tolerance',
          binding: 'wgsl:settings'
        }
      ]
    },
    sources: [
      'modules/experimental/src/gpu-terrain/point-horizon/gpu-point-horizon-visibility.ts',
      'modules/experimental/src/gpu-terrain/point-horizon/point-horizon-march.ts'
    ]
  }
};
