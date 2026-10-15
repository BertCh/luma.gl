// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {RasterOperatorTable} from '../types';

const PARTICLE = {kind: 'particle', label: 'particle', indexSpace: '1d'} as const;
const CELL = {kind: 'cell', label: 'grid cell', indexSpace: '2d'} as const;
const LINE = {kind: 'line', label: 'streamline', indexSpace: '1d'} as const;

/** Raster pack entries of the `vector-field` family, keyed by exported class name. */
export const RASTER_VECTOR_FIELD_OPERATORS: RasterOperatorTable = {
  GPUParticleAdvection: {
    family: 'vector-field',
    descriptor: {scope: 'focal', reduction: 'none'},
    variables: {
      particleWindSpeed: 'rate',
      particleSeeds: 'measure',
      particleTrails: 'measure',
      particleTrailOffsets: 'measure',
      particlePositions: 'measure',
      particleSpeeds: 'rate'
    },
    stages: ['representation', 'support', 'operator', 'iteration'],
    datumKinds: [PARTICLE],
    neighbourhood: 'window',
    guards: [],
    choices: [
      {stage: 'operator', name: 'timeStep', unit: 'h', sweepable: true},
      {stage: 'operator', name: 'speedScale', sweepable: false},
      {stage: 'operator', name: 'dropRate', sweepable: false},
      {stage: 'iteration', name: 'stepCount', sweepable: true}
    ],
    formula: {
      id: 'gpu-raster.particle-advection.rk2',
      derivation: 'curated',
      tex: String.raw`\term{next}{p'}=\term{position}{p}+\term{step}{h}\,\mathbf{v}\!\left(p+\tfrac{h}{2}\,\mathbf{v}(p)\right),\quad h=\term{timeStep}{\Delta t}\,\term{speedScale}{\sigma},\quad \mathbf{v}=\operatorname{bilinear}\bigl(\term{field}{V}\bigr)`,
      ascii:
        'Each step is one midpoint (RK2) step p1 = p + h * v(p + h/2 * v(p)) with h = timeStep * speedScale, where v bilinearly interpolates the (u, v) cell centres of the field (row 0 at the smallest y, clamped to the outermost centres). The field and h share one unit system: here the field is in degrees per hour and timeStep is 2 hours, so p is lon/lat degrees. A particle respawns instead of stepping when any of the three samples (p, midpoint, landing) is outside the field or touches a NaN cell, when its age reaches maximumAge (0 = never), when its speed is below minimumSpeed, or with probability dropRate; this run uses none of those and no particle left the field. Trail slot frame % L receives the new position.',
      symbols: [
        {id: 'next', role: 'output', label: 'next position', binding: 'wgsl:positions'},
        {id: 'position', role: 'input', label: 'position', binding: 'wgsl:position'},
        {id: 'step', role: 'intermediate', label: 'step', binding: 'wgsl:step'},
        {id: 'timeStep', role: 'parameter', label: 'time step', binding: 'wgsl:timeStep'},
        {id: 'speedScale', role: 'parameter', label: 'speed scale', binding: 'wgsl:speedScale'},
        {id: 'field', role: 'input', label: 'velocity field', binding: 'wgsl:velocities'}
      ]
    },
    sources: [
      'modules/experimental/src/gpu-raster/particle-advection/gpu-particle-advection.ts',
      'modules/experimental/src/gpu-raster/vector-field-common/vector-field-sampling.ts'
    ]
  },
  GPULineIntegralConvolution: {
    family: 'vector-field',
    descriptor: {scope: 'focal', reduction: 'weighted-mean'},
    variables: {
      licWindSpeed: 'rate',
      licValues: 'fraction',
      licSpeeds: 'rate'
    },
    stages: ['representation', 'support', 'neighbourhood', 'operator'],
    datumKinds: [CELL],
    neighbourhood: 'path',
    guards: [],
    choices: [
      {stage: 'neighbourhood', name: 'stepCount', sweepable: true},
      {stage: 'neighbourhood', name: 'stepLength', unit: 'pixel', sweepable: true},
      {stage: 'operator', name: 'seed', sweepable: false},
      {stage: 'operator', name: 'period', sweepable: false}
    ],
    formula: {
      id: 'gpu-raster.line-integral-convolution.hann',
      derivation: 'curated',
      tex: String.raw`\term{value}{I}=\frac{\sum_{s=-L}^{L}\term{window}{w_s}\,\term{noise}{N}(x_s)}{\sum_{s=-L}^{L}w_s},\quad w_s=\tfrac12\Bigl(1+\cos\frac{\pi s}{\term{stepCount}{L}+1}\Bigr)`,
      ascii:
        'x_0 is the output pixel centre; x_s for s = 1..L follows the normalised field forward and for s = -1..-L backward, each step an RK2 midpoint step of stepLength output pixels along sign * v / |v|, with v bilinearly sampled from the field. A direction stops early at the field or output edge, NaN data, or speed zero or below minimumSpeed; only visited pixels join both sums. The noise N is white noise per output pixel, Philox keyed (seed, column, row). With period > 0 the window is multiplied by 0.5 * (1 + cos(2 pi (s / period - phase))); this run is static (period 0). A pixel whose centre has no field data is NaN. Direction is taken in the field space, so a lon/lat field with a lon/lat output step follows streamlines in degrees.',
      symbols: [
        {id: 'value', role: 'output', label: 'LIC value', binding: 'wgsl:values'},
        {id: 'window', role: 'intermediate', label: 'weight', binding: 'wgsl:weight'},
        {id: 'noise', role: 'input', label: 'white noise', binding: 'wgsl:noise'},
        {
          id: 'stepCount',
          role: 'parameter',
          label: 'steps per direction',
          binding: 'wgsl:STEP_COUNT'
        }
      ]
    },
    sources: [
      'modules/experimental/src/gpu-raster/flow-texture/gpu-line-integral-convolution.ts',
      'modules/experimental/src/gpu-raster/vector-field-common/vector-field-sampling.ts'
    ]
  },
  GPUStreamlines: {
    family: 'vector-field',
    descriptor: {scope: 'frontier', reduction: 'select'},
    variables: {
      streamlinesWindSpeed: 'rate',
      streamlinesPositions: 'measure',
      streamlinesLineOffsets: 'measure',
      streamlinesLineIds: 'category',
      streamlinesLineCount: 'measure',
      streamlinesRequiredCount: 'measure',
      streamlinesOverflow: 'category'
    },
    stages: ['representation', 'support', 'operator', 'iteration'],
    datumKinds: [LINE],
    neighbourhood: 'path',
    guards: [],
    choices: [
      {stage: 'support', name: 'separation', unit: 'deg', sweepable: true},
      {stage: 'operator', name: 'stepLength', unit: 'deg', sweepable: true},
      {stage: 'operator', name: 'stepsPerDirection', sweepable: false},
      {stage: 'operator', name: 'minimumPoints', sweepable: false},
      {stage: 'iteration', name: 'maximumIterations', sweepable: false}
    ],
    formula: {
      formula: 'none',
      reason:
        'Each seed is traced by RK2 and the traces are then pruned by priority rounds that claim and decide occupancy-grid cells; the accepted lines have no closed form.'
    },
    sources: [
      'modules/experimental/src/gpu-raster/flow-texture/gpu-streamlines.ts',
      'modules/experimental/src/gpu-raster/vector-field-common/vector-field-sampling.ts'
    ]
  }
};
