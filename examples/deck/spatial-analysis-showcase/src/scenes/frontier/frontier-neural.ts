// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GPUCommandNode, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {createFrontierKernelNode} from './frontier-kernel';

export const FRONTIER_NEURAL_CHANNEL_COUNT = 4;
export const FRONTIER_NEURAL_FEATURE_COUNT = 8;
export const FRONTIER_NEURAL_HIDDEN_COUNT = 16;
export const FRONTIER_NEURAL_WEIGHT_COUNT =
  FRONTIER_NEURAL_FEATURE_COUNT * FRONTIER_NEURAL_HIDDEN_COUNT +
  FRONTIER_NEURAL_HIDDEN_COUNT +
  FRONTIER_NEURAL_CHANNEL_COUNT * FRONTIER_NEURAL_HIDDEN_COUNT +
  FRONTIER_NEURAL_CHANNEL_COUNT;
export const FRONTIER_NEURAL_PARAMETER_LENGTH = 8;

/** Tiny 848-byte residual-diffusion preset; replaceable without graph recompilation. */
export function getFrontierNeuralWeights(): Float32Array {
  const weights = new Float32Array(FRONTIER_NEURAL_WEIGHT_COUNT);
  const outputWeightOffset =
    FRONTIER_NEURAL_FEATURE_COUNT * FRONTIER_NEURAL_HIDDEN_COUNT + FRONTIER_NEURAL_HIDDEN_COUNT;
  const setInput = (hidden: number, feature: number, value: number) => {
    weights[hidden * FRONTIER_NEURAL_FEATURE_COUNT + feature] = value;
  };
  const setOutput = (channel: number, hidden: number, value: number) => {
    weights[outputWeightOffset + channel * FRONTIER_NEURAL_HIDDEN_COUNT + hidden] = value;
  };
  for (let channel = 0; channel < FRONTIER_NEURAL_CHANNEL_COUNT; channel++) {
    setInput(4 + channel * 2, 4 + channel, 1);
    setInput(5 + channel * 2, 4 + channel, -1);
    setOutput(channel, 4 + channel * 2, 0.24);
    setOutput(channel, 5 + channel * 2, -0.24);
  }
  return weights;
}

/** Live controls of one neural cellular update. */
export type FrontierNeuralSettings = {
  stepScale?: number;
  brushPosition?: readonly [number, number];
  brushRadius?: number;
  brushStrength?: number;
  mutation?: number;
  generation?: number;
  damping?: number;
};

/** Packs the live neural controls. */
export function getFrontierNeuralParameterValues(
  settings: FrontierNeuralSettings = {},
  target = new Float32Array(FRONTIER_NEURAL_PARAMETER_LENGTH)
): Float32Array {
  const brush = settings.brushPosition ?? [-1e6, -1e6];
  target.set([
    settings.stepScale ?? 0.5,
    brush[0],
    brush[1],
    settings.brushRadius ?? 0,
    settings.brushStrength ?? 0,
    settings.mutation ?? 0,
    Math.max(0, Math.floor(settings.generation ?? 0)),
    settings.damping ?? 0.01
  ]);
  return target;
}

/** Inputs and outputs of one Frontier Lab NCA step. */
export type FrontierNeuralProps = {
  id?: string;
  width: number;
  height: number;
  state: GraphDataView<'float32'>;
  weights: GraphDataView<'float32'>;
  parameters: GraphDataView<'float32'>;
  /** Per-cell upper bound. Image inpainting uses one for every cell. */
  habitat: GraphDataView<'float32'>;
  /** Three-channel reference image used as the temporal structural prior. */
  prior: GraphDataView<'float32'>;
  nextState: GraphDataView<'float32'>;
  /** Packed RGBA8 output, with red in the low byte. */
  display: GraphDataView<'uint32'>;
  boundary?: 'clamp' | 'wrap';
};

/** One compact neural-style MLP local-rule update over a four-channel cellular field. */
export class FrontierNeuralAutomaton {
  readonly id: string;
  readonly props: FrontierNeuralProps;

  constructor(props: FrontierNeuralProps) {
    this.id = props.id ?? 'frontier-neural';
    this.props = props;
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const props = this.props;
    const edge =
      (props.boundary ?? 'clamp') === 'wrap'
        ? 'return (value % size + size) % size;'
        : 'return clamp(value, 0, size - 1);';
    const hiddenBiasOffset = FRONTIER_NEURAL_FEATURE_COUNT * FRONTIER_NEURAL_HIDDEN_COUNT;
    const outputWeightOffset = hiddenBiasOffset + FRONTIER_NEURAL_HIDDEN_COUNT;
    const outputBiasOffset =
      outputWeightOffset + FRONTIER_NEURAL_CHANNEL_COUNT * FRONTIER_NEURAL_HIDDEN_COUNT;
    return [
      createFrontierKernelNode(graph, {
        id: `${this.id}-step`,
        operation: 'FrontierNeuralAutomaton',
        variant: props.boundary ?? 'clamp',
        bindings: [
          {name: 'state', view: props.state, type: 'f32', access: 'read'},
          {name: 'weights', view: props.weights, type: 'f32', access: 'read'},
          {name: 'parameters', view: props.parameters, type: 'f32', access: 'read'},
          {name: 'habitat', view: props.habitat, type: 'f32', access: 'read'},
          {name: 'prior', view: props.prior, type: 'f32', access: 'read'},
          {name: 'nextState', view: props.nextState, type: 'f32', access: 'read_write'},
          {name: 'display', view: props.display, type: 'u32', access: 'read_write'}
        ],
        invocationCount: props.width * props.height,
        declarations: `const WIDTH: i32 = ${props.width};
const HEIGHT: i32 = ${props.height};
const CHANNEL_COUNT: u32 = ${FRONTIER_NEURAL_CHANNEL_COUNT}u;
const FEATURE_COUNT: u32 = ${FRONTIER_NEURAL_FEATURE_COUNT}u;
const HIDDEN_COUNT: u32 = ${FRONTIER_NEURAL_HIDDEN_COUNT}u;
const HIDDEN_BIAS_OFFSET: u32 = ${hiddenBiasOffset}u;
const OUTPUT_WEIGHT_OFFSET: u32 = ${outputWeightOffset}u;
const OUTPUT_BIAS_OFFSET: u32 = ${outputBiasOffset}u;

fn edge(value: i32, size: i32) -> i32 { ${edge} }
fn loadState(column: i32, row: i32, channel: u32) -> f32 {
  let x = edge(column, WIDTH);
  let y = edge(row, HEIGHT);
  return state[stateOffset + u32(y * WIDTH + x) * CHANNEL_COUNT + channel];
}
fn hash(value: u32) -> f32 {
  var word = value + 0x9e3779b9u;
  word = (word ^ (word >> 16u)) * 0x21f0aaadu;
  word = (word ^ (word >> 15u)) * 0x735a2d97u;
  word = word ^ (word >> 15u);
  return f32(word) / 4294967295.0;
}`,
        body: `let column = i32(index % u32(WIDTH));
  let row = i32(index / u32(WIDTH));
  var feature: array<f32, ${FRONTIER_NEURAL_FEATURE_COUNT}>;
  for (var channel = 0u; channel < CHANNEL_COUNT; channel++) {
    let center = loadState(column, row, channel);
    feature[channel] = center;
    feature[CHANNEL_COUNT + channel] =
      loadState(column - 1, row, channel) + loadState(column + 1, row, channel) +
      loadState(column, row - 1, channel) + loadState(column, row + 1, channel) - 4.0 * center;
  }
  var hidden: array<f32, ${FRONTIER_NEURAL_HIDDEN_COUNT}>;
  for (var hiddenIndex = 0u; hiddenIndex < HIDDEN_COUNT; hiddenIndex++) {
    var value = weights[weightsOffset + HIDDEN_BIAS_OFFSET + hiddenIndex];
    for (var featureIndex = 0u; featureIndex < FEATURE_COUNT; featureIndex++) {
      value += weights[weightsOffset + hiddenIndex * FEATURE_COUNT + featureIndex] * feature[featureIndex];
    }
    hidden[hiddenIndex] = max(value, 0.0);
  }
  let viability = habitat[habitatOffset + index];
  var next: array<f32, ${FRONTIER_NEURAL_CHANNEL_COUNT}>;
  for (var outputIndex = 0u; outputIndex < CHANNEL_COUNT; outputIndex++) {
    var delta = weights[weightsOffset + OUTPUT_BIAS_OFFSET + outputIndex];
    for (var hiddenIndex = 0u; hiddenIndex < HIDDEN_COUNT; hiddenIndex++) {
      delta += weights[weightsOffset + OUTPUT_WEIGHT_OFFSET + outputIndex * HIDDEN_COUNT + hiddenIndex] * hidden[hiddenIndex];
    }
    if (feature[3] >= 0.999) {
      next[outputIndex] = feature[outputIndex];
    } else {
      let noise = (hash(index * 17u + outputIndex * 131u + u32(parameters[parametersOffset + 6u]) * 8191u) - 0.5) * parameters[parametersOffset + 5u] * feature[3];
      let lower = select(-1.0, 0.0, outputIndex == 3u);
      next[outputIndex] = clamp(
        feature[outputIndex] + parameters[parametersOffset] * delta + noise - parameters[parametersOffset + 7u] * feature[outputIndex],
        lower,
        viability
      );
    }
  }
  let brushDelta = vec2<f32>(f32(column), f32(row)) - vec2<f32>(parameters[parametersOffset + 1u], parameters[parametersOffset + 2u]);
  let brush = max(1.0 - length(brushDelta) / max(parameters[parametersOffset + 3u], 1.0e-6), 0.0);
  let brushStrength = parameters[parametersOffset + 4u];
  if (brushStrength > 0.0 && brush > 0.0) {
    let amount = clamp(brush * brushStrength * viability, 0.0, viability);
    next[0] = max(next[0], amount);
    next[1] = max(next[1], amount * 0.45);
    next[2] = max(next[2], amount * 0.15);
    next[3] = max(next[3], amount);
  } else if (brushStrength < 0.0 && brush > 0.0) {
    for (var channel = 0u; channel < CHANNEL_COUNT; channel++) { next[channel] = 0.0; }
  }
  let base = nextStateOffset + index * CHANNEL_COUNT;
  for (var channel = 0u; channel < CHANNEL_COUNT; channel++) { nextState[base + channel] = next[channel]; }
  let priorBase = priorOffset + index * 3u;
  let red = u32(round(clamp(prior[priorBase] + next[0], 0.0, 1.0) * 255.0));
  let green = u32(round(clamp(prior[priorBase + 1u] + next[1], 0.0, 1.0) * 255.0));
  let blue = u32(round(clamp(prior[priorBase + 2u] + next[2], 0.0, 1.0) * 255.0));
  display[displayOffset + index] = red | (green << 8u) | (blue << 16u) | (255u << 24u);`
      })
    ];
  }
}
