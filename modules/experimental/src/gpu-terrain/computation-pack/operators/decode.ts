// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {RasterOperatorTable} from '../../../gpu-raster/computation-pack/types';

/** Terrain pack entries of the `decode` family, keyed by exported class name. */
export const TERRAIN_DECODE_OPERATORS: RasterOperatorTable = {
  GPUTerrainRGBDecode: {
    family: 'decode',
    descriptor: {
      scope: 'local',
      reduction: 'none'
    },
    variables: {
      terrariumRGBA: 'category',
      decodedElevation: 'elevation',
      decodedValidity: 'category'
    },
    stages: ['support', 'neighbourhood', 'operator'],
    datumKinds: [
      {
        kind: 'cell',
        label: 'grid cell',
        indexSpace: '2d'
      }
    ],
    neighbourhood: 'none',
    guards: [],
    choices: [
      {
        stage: 'representation',
        name: 'encoding',
        sweepable: false,
        alternatives: ['terrarium', 'mapbox']
      },
      {
        stage: 'operator',
        name: 'alphaNoData',
        sweepable: false
      },
      {
        stage: 'operator',
        name: 'clampBathymetry',
        sweepable: false
      }
    ],
    formula: {
      id: 'gpu-terrain.terrain-rgb-decode.terrarium',
      tex: String.raw`\term{code}{N}=65536\term{red}{R}+256\term{green}{G}+\term{blue}{B},\quad\term{height}{h}=\operatorname{f32}(N)\,2^{-8}-32768`,
      ascii:
        'Unpack R=word&255, G=(word>>8)&255, B=(word>>16)&255, A=word>>24; Terrarium height=f32((R<<16)|(G<<8)|B)*0.00390625-32768 exactly. Alpha zero (by default), encoded nodata, invalid source or height outside closed validRange [-11000,9000] gives NaN and validity 0; otherwise validity 1. Optional sea clamp maps -12000<h<0 to 0 after validity; disabled here. Mapbox uses f32(i32(N)-100000)*f32(0.1).',
      derivation: 'curated',
      symbols: [
        {
          id: 'code',
          role: 'intermediate',
          label: 'RGB code',
          binding: 'wgsl:code'
        },
        {
          id: 'red',
          role: 'input',
          label: 'red byte',
          binding: 'wgsl:red'
        },
        {
          id: 'green',
          role: 'input',
          label: 'green byte',
          binding: 'wgsl:green'
        },
        {
          id: 'blue',
          role: 'input',
          label: 'blue byte',
          binding: 'wgsl:blue'
        },
        {
          id: 'height',
          role: 'output',
          label: 'decoded height',
          binding: 'wgsl:height'
        }
      ]
    },
    sources: ['modules/experimental/src/gpu-terrain/terrain-decode/gpu-terrain-rgb-decode.ts']
  },
  GPUTerrainSpikeRepair: {
    family: 'decode',
    descriptor: {
      scope: 'frontier',
      reduction: 'select'
    },
    variables: {
      spikedElevation: 'elevation',
      repairedElevation: 'elevation',
      repairValidity: 'category',
      repairLabels: 'category',
      repairStatistics: 'measure',
      repairConverged: 'category'
    },
    stages: ['support', 'neighbourhood', 'operator', 'iteration'],
    datumKinds: [
      {
        kind: 'cell',
        label: 'grid cell',
        indexSpace: '2d'
      }
    ],
    neighbourhood: 'window',
    guards: ['convergence'],
    choices: [
      {
        stage: 'operator',
        name: 'step',
        sweepable: false,
        unit: 'm'
      },
      {
        stage: 'neighbourhood',
        name: 'jump',
        sweepable: false,
        unit: 'm'
      },
      {
        stage: 'operator',
        name: 'tolerance',
        sweepable: false,
        unit: 'm'
      },
      {
        stage: 'operator',
        name: 'agreement',
        sweepable: false
      },
      {
        stage: 'operator',
        name: 'maximumComponentFraction',
        sweepable: false
      },
      {
        stage: 'iteration',
        name: 'maximumComponentIterations',
        sweepable: false
      }
    ],
    formula: {
      id: 'gpu-terrain.terrain-spike-repair.component-shift',
      tex: String.raw`\term{multiple}{k}=\operatorname{round}(\term{difference}{\Delta h}/\term{step}{s}),\quad V(k)=\mathbf{1}[0<|k|\le127\land|\Delta h-sk|<\term{tolerance}{\epsilon}],\quad\term{output}{h_i\prime}=\term{input}{h_i}-s\,k_C\,\mathbf{1}[\term{shift}{\operatorname{shift}(C)}]`,
      ascii:
        'Join 4-neighbours with abs(dh)<=jump, label components and choose the largest (ties lowest label) as main. For each ordered seam, k=round(dh*f32(1/step)); vote iff 0<abs(k)<=127 and abs(dh-step*k)<tolerance. Reconstruct candidate k by strict majority bits, recount agreeing seams. Shift a converged nonempty non-main component only if size*100<=validCount*maximumComponentPercent, seams>0, candidate exists and agreeCount*100>=seams*agreementPercent; replace h by h-step*k, otherwise keep h. Invalid cells stay NaN/validity 0. Nonconvergence gives valid cells +Infinity, validity and labels 0xffffffff, convergence 0.',
      derivation: 'curated',
      symbols: [
        {
          id: 'multiple',
          role: 'intermediate',
          label: 'step multiple',
          binding: 'wgsl:multiple'
        },
        {
          id: 'difference',
          role: 'input',
          label: 'border height jump',
          binding: 'wgsl:difference'
        },
        {
          id: 'step',
          role: 'parameter',
          label: 'encoding step',
          binding: 'wgsl:STEP'
        },
        {
          id: 'tolerance',
          role: 'parameter',
          label: 'seam tolerance',
          binding: 'wgsl:TOLERANCE'
        },
        {
          id: 'output',
          role: 'output',
          label: 'repaired height',
          binding: 'wgsl:valuesOut'
        },
        {
          id: 'input',
          role: 'input',
          label: 'height',
          binding: 'wgsl:values'
        },
        {
          id: 'shift',
          role: 'intermediate',
          label: 'component shift decision',
          binding: 'wgsl:shift'
        }
      ]
    },
    sources: [
      'modules/experimental/src/gpu-terrain/terrain-decode/gpu-terrain-spike-repair.ts',
      'modules/gpgpu/src/gpu-graph/gpu-graph-connected-components-internals.ts'
    ]
  }
};
