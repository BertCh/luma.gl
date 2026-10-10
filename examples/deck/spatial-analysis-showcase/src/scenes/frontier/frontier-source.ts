// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GPUCommandNode, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {createFrontierKernelNode} from './frontier-kernel';

export const FRONTIER_SOURCE_PARAMETER_LENGTH = 12;

/** Live Gaussian-plume assumptions for the inverse-source prototype. */
export type FrontierSourceSettings = {
  bounds: readonly [number, number, number, number];
  wind: readonly [number, number];
  emissionRate: number;
  dispersionBase: number;
  dispersionGrowth: number;
  decayLength: number;
  noiseSigma: number;
  background?: number;
};

/** Packs one inverse-source parameter buffer. */
export function getFrontierSourceParameterValues(
  settings: FrontierSourceSettings,
  target = new Float32Array(FRONTIER_SOURCE_PARAMETER_LENGTH)
): Float32Array {
  target.set(settings.bounds, 0);
  target.set(settings.wind, 4);
  target.set(
    [
      settings.emissionRate,
      settings.dispersionBase,
      settings.dispersionGrowth,
      settings.decayLength,
      settings.noiseSigma,
      settings.background ?? 0
    ],
    6
  );
  return target;
}

/** Inputs and GPU-resident outputs of {@link FrontierSourceInference}. */
export type FrontierSourceProps = {
  id?: string;
  width: number;
  height: number;
  sensorCount: number;
  sensorPositions: GraphDataView<'float32x2'>;
  sensorValues: GraphDataView<'float32'>;
  parameters: GraphDataView<'float32'>;
  scores: GraphDataView<'float32'>;
  /** `[maximumScoreBits, bestCell]`. */
  summary: GraphDataView<'uint32'>;
  /** One `float32x2` row receiving the winning candidate position. */
  bestPosition: GraphDataView<'float32x2'>;
};

/**
 * Scores a candidate raster against sparse sensors and publishes the best position on the GPU.
 *
 * The relative-fit surface and marker position feed deck.gl storage-buffer layers directly. The
 * two-word atomic summary supports asynchronous scalar diagnostics without reading back the field.
 */
export class FrontierSourceInference {
  readonly id: string;
  readonly props: FrontierSourceProps;

  constructor(props: FrontierSourceProps) {
    this.id = props.id ?? 'frontier-source';
    this.props = props;
    if (props.sensorCount < 1 || props.sensorCount > 64) {
      throw new Error('FrontierSourceInference supports 1 to 64 sensors');
    }
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const props = this.props;
    const cellCount = props.width * props.height;
    return [
      createFrontierKernelNode(graph, {
        id: `${this.id}-clear`,
        operation: 'FrontierSourceInference',
        variant: 'clear',
        bindings: [
          {name: 'summary', view: props.summary, type: 'atomic<u32>', access: 'read_write'}
        ],
        invocationCount: 1,
        body: `atomicStore(&summary[summaryOffset], 0u);
  atomicStore(&summary[summaryOffset + 1u], 0xffffffffu);`
      }),
      createFrontierKernelNode(graph, {
        id: `${this.id}-score`,
        operation: 'FrontierSourceInference',
        variant: 'gaussian-plume',
        bindings: [
          {name: 'sensors', view: props.sensorPositions, type: 'f32', access: 'read'},
          {name: 'observations', view: props.sensorValues, type: 'f32', access: 'read'},
          {name: 'parameters', view: props.parameters, type: 'f32', access: 'read'},
          {name: 'scores', view: props.scores, type: 'f32', access: 'read_write'},
          {name: 'summary', view: props.summary, type: 'atomic<u32>', access: 'read_write'}
        ],
        invocationCount: cellCount,
        declarations: `const WIDTH: u32 = ${props.width}u;
const HEIGHT: u32 = ${props.height}u;
const SENSOR_COUNT: u32 = ${props.sensorCount}u;`,
        body: `let minimum = vec2<f32>(parameters[parametersOffset], parameters[parametersOffset + 1u]);
  let maximum = vec2<f32>(parameters[parametersOffset + 2u], parameters[parametersOffset + 3u]);
  let cellSize = (maximum - minimum) / vec2<f32>(f32(WIDTH), f32(HEIGHT));
  let column = index % WIDTH;
  let row = index / WIDTH;
  let source = minimum + (vec2<f32>(f32(column), f32(row)) + 0.5) * cellSize;
  let rawWind = vec2<f32>(parameters[parametersOffset + 4u], parameters[parametersOffset + 5u]);
  let wind = rawWind / max(length(rawWind), 1.0e-6);
  let crosswindAxis = vec2<f32>(-wind.y, wind.x);
  let emissionRate = max(parameters[parametersOffset + 6u], 0.0);
  let dispersionBase = max(parameters[parametersOffset + 7u], 1.0e-6);
  let dispersionGrowth = max(parameters[parametersOffset + 8u], 0.0);
  let decayLength = max(parameters[parametersOffset + 9u], 1.0e-6);
  let noiseSigma = max(parameters[parametersOffset + 10u], 1.0e-6);
  let background = parameters[parametersOffset + 11u];
  var squaredError = 0.0;
  for (var sensor = 0u; sensor < SENSOR_COUNT; sensor++) {
    let positionIndex = sensorsOffset + 2u * sensor;
    let sensorPosition = vec2<f32>(sensors[positionIndex], sensors[positionIndex + 1u]);
    let delta = sensorPosition - source;
    let downwind = dot(delta, wind);
    var predicted = background;
    if (downwind > 0.0) {
      let sigma = dispersionBase + dispersionGrowth * sqrt(downwind);
      let crosswind = dot(delta, crosswindAxis);
      let centerline = emissionRate / (1.0 + downwind / decayLength);
      predicted += centerline * exp(-0.5 * crosswind * crosswind / (sigma * sigma));
    }
    let residual = (observations[observationsOffset + sensor] - predicted) / noiseSigma;
    squaredError += residual * residual;
  }
  let score = exp(-0.5 * squaredError / f32(SENSOR_COUNT));
  scores[scoresOffset + index] = score;
  atomicMax(&summary[summaryOffset], bitcast<u32>(score));`
      }),
      createFrontierKernelNode(graph, {
        id: `${this.id}-select`,
        operation: 'FrontierSourceInference',
        variant: 'select',
        bindings: [
          {name: 'scores', view: props.scores, type: 'f32', access: 'read'},
          {name: 'summary', view: props.summary, type: 'atomic<u32>', access: 'read_write'}
        ],
        invocationCount: cellCount,
        body: `let maximumBits = atomicLoad(&summary[summaryOffset]);
  if (bitcast<u32>(scores[scoresOffset + index]) == maximumBits) {
    atomicMin(&summary[summaryOffset + 1u], index);
  }`
      }),
      createFrontierKernelNode(graph, {
        id: `${this.id}-publish`,
        operation: 'FrontierSourceInference',
        variant: 'publish-position',
        bindings: [
          {name: 'parameters', view: props.parameters, type: 'f32', access: 'read'},
          {name: 'summary', view: props.summary, type: 'u32', access: 'read'},
          {name: 'bestPosition', view: props.bestPosition, type: 'f32', access: 'read_write'}
        ],
        invocationCount: 1,
        declarations: `const WIDTH: u32 = ${props.width}u;
const HEIGHT: u32 = ${props.height}u;`,
        body: `let winner = summary[summaryOffset + 1u];
  let minimum = vec2<f32>(parameters[parametersOffset], parameters[parametersOffset + 1u]);
  let maximum = vec2<f32>(parameters[parametersOffset + 2u], parameters[parametersOffset + 3u]);
  let cellSize = (maximum - minimum) / vec2<f32>(f32(WIDTH), f32(HEIGHT));
  let position = minimum + (vec2<f32>(f32(winner % WIDTH), f32(winner / WIDTH)) + 0.5) * cellSize;
  bestPosition[bestPositionOffset] = position.x;
  bestPosition[bestPositionOffset + 1u] = position.y;`
      })
    ];
  }
}
