// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  validatePackedUint32View,
  validatePackedView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView,
  type GraphTextureView
} from '@luma.gl/gpgpu/gpu-core';
import {GPURasterBufferToTexture} from '../../gpu-raster/index';
import {createMapGraphKernelNode, type MapGraphKernelBinding} from '../map-graph-kernels';
import type {GPUMapGraphRecipe} from '../map-graph-types';
import {
  captureGraphCommandNodes,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../map-graph-utils';
import {getFieldSamplingWGSL} from './flow-texture-field';
import {LINE_INTEGRAL_CONVOLUTION_NOISE_PURPOSE} from './line-integral-convolution-cpu';
import {
  GPU_LINE_INTEGRAL_CONVOLUTION_PARAMETER_LENGTH,
  GPU_LINE_INTEGRAL_CONVOLUTION_WORD_PARAMETER_LENGTH
} from './line-integral-convolution-parameters';
import {PHILOX_WGSL} from './flow-texture-random';

const OPERATION = 'GPULineIntegralConvolution';
const MAXIMUM_STEP_COUNT = 256;

/** Caller-owned outputs of {@link GPULineIntegralConvolution}. */
export type GPULineIntegralConvolutionOutput = {
  /**
   * Row-major LIC values in `[0, 1]`, `width * height` rows, row 0 with the smallest y. NaN where
   * the pixel centre has no field data.
   */
  values: GraphDataView<'float32'>;
  /** Optional field speed at each pixel centre (NaN without data), for colour modulation. */
  speeds?: GraphDataView<'float32'>;
  /** Optional `r32float` storage texture of `width x height` receiving `values`. */
  texture?: GraphTextureView<'r32float'>;
};

/**
 * Properties for {@link GPULineIntegralConvolution}.
 *
 * Per-frame (no recompile): the contents of `parameters` (extents, step length, minimum speed,
 * phase, period), `wordParameters` (seed) and `velocities`. Topology: field and output sizes,
 * `stepCount`, which optional outputs are present.
 */
export type GPULineIntegralConvolutionProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'line-integral-convolution'`. */
  id?: string;
  /** Packed row-major `(u, v)` field, `fieldWidth * fieldHeight` rows; NaN marks no data. */
  velocities: GraphDataView<'float32x2'>;
  /** Field width in cells. Compile-time. */
  fieldWidth: number;
  /** Field height in cells. Compile-time. */
  fieldHeight: number;
  /** Output width in pixels. Compile-time. */
  width: number;
  /** Output height in pixels. Compile-time. */
  height: number;
  /** Streamline steps in each direction (`L`), at most 256. Compile-time. Defaults to 20. */
  stepCount?: number;
  /** Per-frame float32 parameters from `getGPULineIntegralConvolutionParameterValues`. */
  parameters: GraphDataView<'float32'>;
  /** Per-frame uint32 parameters from `getGPULineIntegralConvolutionWordParameterValues`. */
  wordParameters: GraphDataView<'uint32'>;
  /** Caller-owned outputs. */
  output: GPULineIntegralConvolutionOutput;
};

/**
 * Line integral convolution (Cabral and Leedom 1993): a texture that smears white noise along the
 * streamlines of a vector field, so the flow direction is visible everywhere at once.
 *
 * The first node fills a `width x height` white-noise raster with
 * `philox((column, row, 0, 0), (seed, 1)).x`. The second traces, from each pixel centre, `L` RK2
 * midpoint steps forward and `L` backward along the normalised field (one step is `stepLength`
 * output pixels), stopping at the field or output edge, at NaN data, or where the speed is zero or
 * below `minimumSpeed`. The value is the weighted mean of the noise at the visited pixels with a
 * Hann window `0.5 * (1 + cos(pi * s / (L + 1)))` over the signed step `s`. When `period > 0` the
 * window is multiplied by the ripple `0.5 * (1 + cos(2 * pi * (s / period - phase)))`; advancing
 * `phase` each frame makes the texture flow along the field without recomputing anything else
 * (animated LIC, Forssell 1994). The output extent is independent of the field extent, so a
 * viewport can be rendered at screen resolution from a coarse field.
 *
 * Determinism: no atomics and counter-based noise, so encodings are bitwise reproducible on one
 * adapter. Across adapters, `cos` and `normalize` rounding can move a sample into a neighbouring
 * pixel near a pixel edge.
 */
export class GPULineIntegralConvolution implements GPUMapGraphRecipe {
  /** Prefix for graph node and transient IDs. */
  readonly id: string;
  /** Stable recipe name. */
  readonly recipe = 'line-integral-convolution';
  /** Streamline steps in each direction. */
  readonly stepCount: number;
  /** Validated properties. */
  readonly props: Readonly<GPULineIntegralConvolutionProps>;

  constructor(props: GPULineIntegralConvolutionProps) {
    const id = props.id ?? 'line-integral-convolution';
    for (const [name, value] of [
      ['fieldWidth', props.fieldWidth],
      ['fieldHeight', props.fieldHeight],
      ['width', props.width],
      ['height', props.height]
    ] as const) {
      if (!Number.isInteger(value) || value < 1 || value > 65535) {
        throw new Error(`${id} ${name} must be an integer in [1, 65535]`);
      }
    }
    const stepCount = props.stepCount ?? 20;
    if (!Number.isInteger(stepCount) || stepCount < 0 || stepCount > MAXIMUM_STEP_COUNT) {
      throw new Error(`${id} stepCount must be an integer in [0, ${MAXIMUM_STEP_COUNT}]`);
    }
    validatePackedView(props.velocities, ['float32x2'], `${id} velocities`);
    if (props.velocities.length !== props.fieldWidth * props.fieldHeight) {
      throw new Error(`${id} velocities must have fieldWidth * fieldHeight rows`);
    }
    validatePackedView(props.parameters, ['float32'], `${id} parameters`);
    if (props.parameters.length < GPU_LINE_INTEGRAL_CONVOLUTION_PARAMETER_LENGTH) {
      throw new Error(
        `${id} parameters must contain ${GPU_LINE_INTEGRAL_CONVOLUTION_PARAMETER_LENGTH} float32 rows`
      );
    }
    validatePackedUint32View(props.wordParameters, `${id} wordParameters`);
    if (props.wordParameters.length < GPU_LINE_INTEGRAL_CONVOLUTION_WORD_PARAMETER_LENGTH) {
      throw new Error(
        `${id} wordParameters must contain ${GPU_LINE_INTEGRAL_CONVOLUTION_WORD_PARAMETER_LENGTH} uint32 rows`
      );
    }
    const pixelCount = props.width * props.height;
    const {output} = props;
    validatePackedView(output.values, ['float32'], `${id} output.values`);
    if (output.values.length !== pixelCount) {
      throw new Error(`${id} output.values must have width * height rows`);
    }
    if (output.speeds) {
      validatePackedView(output.speeds, ['float32'], `${id} output.speeds`);
      if (output.speeds.length !== pixelCount) {
        throw new Error(`${id} output.speeds must have width * height rows`);
      }
    }
    if (
      output.texture &&
      (output.texture.format !== 'r32float' ||
        output.texture.width !== props.width ||
        output.texture.height !== props.height)
    ) {
      throw new Error(`${id} output.texture must be r32float with width x height`);
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      [output.values, output.speeds],
      [props.velocities, props.parameters, props.wordParameters]
    );
    this.id = id;
    this.stepCount = stepCount;
    this.props = props;
  }

  /** Returns the noise, convolution and optional texture nodes. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, stepCount} = this;
    const {output, width, height} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      props.velocities,
      props.parameters,
      props.wordParameters,
      output.values,
      output.speeds
    ]);
    if (output.texture && output.texture.texture.graph !== graph) {
      throw new Error(`${id} views must belong to the target graph`);
    }
    const pixelCount = width * height;
    const noise = createTransientView(graph, `${id}-noise`, 'float32', pixelCount);
    const nodes: GPUCommandNode<Parameters>[] = [
      createMapGraphKernelNode<Parameters>(graph, {
        id: `${id}-noise`,
        operation: OPERATION,
        variant: 'noise',
        bindings: [
          {
            name: 'words',
            view: props.wordParameters,
            type: 'u32',
            access: 'read'
          },
          {name: 'noise', view: noise, type: 'f32', access: 'read_write'}
        ],
        invocationCount: pixelCount,
        declarations: `${PHILOX_WGSL}
const WIDTH: u32 = ${width}u;`,
        body: `let seed = words[wordsOffset];
  let counter = vec4<u32>(index % WIDTH, index / WIDTH, 0u, 0u);
  noise[noiseOffset + index] = philoxUnitFloat(
    philox4x32(counter, vec2<u32>(seed, ${LINE_INTEGRAL_CONVOLUTION_NOISE_PURPOSE}u)).x
  );`
      })
    ];
    const bindings: MapGraphKernelBinding[] = [
      {
        name: 'velocities',
        view: props.velocities,
        type: 'f32',
        access: 'read'
      },
      {
        name: 'parameters',
        view: props.parameters,
        type: 'f32',
        access: 'read'
      },
      {name: 'noise', view: noise, type: 'f32', access: 'read'},
      {
        name: 'values',
        view: output.values,
        type: 'f32',
        access: 'read_write'
      }
    ];
    if (output.speeds) {
      bindings.push({
        name: 'speeds',
        view: output.speeds,
        type: 'f32',
        access: 'read_write'
      });
    }
    nodes.push(
      createMapGraphKernelNode<Parameters>(graph, {
        id: `${id}-convolve`,
        operation: OPERATION,
        variant: 'convolve',
        bindings,
        invocationCount: pixelCount,
        declarations: `${getFieldSamplingWGSL('velocities', props.fieldWidth, props.fieldHeight)}
const WIDTH: u32 = ${width}u;
const HEIGHT: u32 = ${height}u;
const STEP_COUNT: u32 = ${stepCount}u;
const PI: f32 = 3.14159265358979;
fn readParameter(slot: u32) -> f32 { return parameters[parametersOffset + slot]; }
fn getWeight(signedStep: f32, phase: f32, period: f32) -> f32 {
  let window = 0.5 * (1.0 + cos(PI * signedStep / f32(STEP_COUNT + 1u)));
  var ripple = 1.0;
  if (period > 0.0) {
    ripple = 0.5 * (1.0 + cos(2.0 * PI * (signedStep / period - phase)));
  }
  return window * ripple;
}
// Unit flow direction times sign, or (0, 0, 0) where the trace must stop.
fn getDirection(position: vec2<f32>, fieldExtent: vec4<f32>, directionSign: f32, minimumSpeed: f32) -> vec3<f32> {
  let sample = sampleField(position, fieldExtent);
  if (sample.z == 0.0) {
    return vec3<f32>(0.0);
  }
  let speed = length(sample.xy);
  if (!(speed > 0.0) || speed < minimumSpeed) {
    return vec3<f32>(0.0);
  }
  return vec3<f32>(directionSign * sample.xy / speed, 1.0);
}`,
        body: /* wgsl */ `
  let fieldExtent = vec4<f32>(readParameter(0u), readParameter(1u), readParameter(2u), readParameter(3u));
  let origin = vec2<f32>(readParameter(4u), readParameter(5u));
  let cell = vec2<f32>(readParameter(6u), readParameter(7u));
  let stepVector = readParameter(8u) * cell;
  let minimumSpeed = readParameter(9u);
  let phase = readParameter(10u);
  let period = readParameter(11u);
  let column = index % WIDTH;
  let row = index / WIDTH;
  let centre = origin + (vec2<f32>(f32(column), f32(row)) + vec2<f32>(0.5)) * cell;
  let centreSample = sampleField(centre, fieldExtent);
  if (centreSample.z == 0.0) {
    var nanBits = 0x7fc00000u;
    values[valuesOffset + index] = bitcast<f32>(nanBits);
    ${output.speeds ? 'speeds[speedsOffset + index] = bitcast<f32>(nanBits);' : ''}
    return;
  }
  ${output.speeds ? 'speeds[speedsOffset + index] = length(centreSample.xy);' : ''}
  var weightSum = getWeight(0.0, phase, period);
  var sum = weightSum * noise[noiseOffset + index];
  for (var direction = 0u; direction < 2u; direction = direction + 1u) {
    let directionSign = select(1.0, -1.0, direction == 1u);
    var position = centre;
    for (var stepIndex = 1u; stepIndex <= STEP_COUNT; stepIndex = stepIndex + 1u) {
      let first = getDirection(position, fieldExtent, directionSign, minimumSpeed);
      if (first.z == 0.0) {
        break;
      }
      let middle = getDirection(position + 0.5 * stepVector * first.xy, fieldExtent, directionSign, minimumSpeed);
      if (middle.z == 0.0) {
        break;
      }
      position = position + stepVector * middle.xy;
      let pixel = floor((position - origin) / cell);
      if (pixel.x < 0.0 || pixel.y < 0.0 || pixel.x >= f32(WIDTH) || pixel.y >= f32(HEIGHT)) {
        break;
      }
      let weight = getWeight(directionSign * f32(stepIndex), phase, period);
      sum = sum + weight * noise[noiseOffset + u32(pixel.y) * WIDTH + u32(pixel.x)];
      weightSum = weightSum + weight;
    }
  }
  values[valuesOffset + index] = select(noise[noiseOffset + index], sum / weightSum, weightSum > 0.0);`
      })
    );
    if (output.texture) {
      const texture = output.texture;
      nodes.push(
        ...captureGraphCommandNodes(graph, () =>
          new GPURasterBufferToTexture({
            id: `${id}-texture`,
            input: {
              id: `${id}-texture-band`,
              format: 'float32',
              storage: {kind: 'buffer', values: output.values}
            },
            output: texture
          }).addToGraph(graph)
        )
      );
    }
    return nodes;
  }
}
