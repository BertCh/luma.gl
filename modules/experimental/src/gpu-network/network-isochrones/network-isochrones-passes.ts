// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GPUCommandNode, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {createWGSLKernelNode, getWGSLFloatLiteral} from '../../utils/wgsl-kernel-nodes';

const OPERATION = 'GPUNetworkIsochrones';

/** Raster accumulation rule of the splat: smallest or largest cost per pixel. */
export type GPUNetworkIsochronesMode = 'min' | 'max';

/** Encoded value of an untouched pixel for each mode. @internal */
export function getIsochronesEncodedInitialValue(mode: GPUNetworkIsochronesMode): number {
  return mode === 'min' ? 0xffffffff : 0;
}

/**
 * Writes the {@link GPUIsobands} parameter view from the isochrone parameters (cell sizes from the
 * extent and the raster size). @internal
 */
export function createIsochronesBandParametersNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    width: number;
    height: number;
    parameters: GraphDataView<'float32'>;
    bandParameters: GraphDataView<'float32'>;
  }
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'band-parameters',
    bindings: [
      {name: 'params', view: props.parameters, type: 'f32', access: 'read'},
      {name: 'bandParams', view: props.bandParameters, type: 'f32', access: 'read_write'}
    ],
    invocationCount: 1,
    declarations: `const WIDTH: f32 = ${props.width}.0;
const HEIGHT: f32 = ${props.height}.0;`,
    body: `bandParams[bandParamsOffset] = params[paramsOffset];
  bandParams[bandParamsOffset + 1u] = params[paramsOffset + 7u];
  bandParams[bandParamsOffset + 2u] = params[paramsOffset + 8u];
  bandParams[bandParamsOffset + 3u] = 0.0;
  bandParams[bandParamsOffset + 4u] = params[paramsOffset + 1u];
  bandParams[bandParamsOffset + 5u] = params[paramsOffset + 2u];
  bandParams[bandParamsOffset + 6u] = params[paramsOffset + 3u];
  bandParams[bandParamsOffset + 7u] = params[paramsOffset + 4u];
  bandParams[bandParamsOffset + 8u] = (params[paramsOffset + 3u] - params[paramsOffset + 1u]) / WIDTH;
  bandParams[bandParamsOffset + 9u] = (params[paramsOffset + 4u] - params[paramsOffset + 2u]) / HEIGHT;
  bandParams[bandParamsOffset + 10u] = 0.0;
  bandParams[bandParamsOffset + 11u] = 0.0;`
  });
}

/**
 * One thread per edge: samples the edge at a bounded number of points, interpolates the cost
 * linearly from the source node cost along the edge weight, and applies `atomicMin` or `atomicMax`
 * to every pixel within the buffer of each sample. Costs are non-negative f32, so their bit
 * patterns order like integers; max mode stores `bits + 1` so zero means untouched. @internal
 */
export function createIsochronesSplatNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    nodeCount: number;
    width: number;
    height: number;
    mode: GPUNetworkIsochronesMode;
    maximumBufferPixels: number;
    maximumSamplesPerEdge: number;
    offsets: GraphDataView<'uint32'>;
    neighbors: GraphDataView<'uint32'>;
    weights: GraphDataView<'float32'>;
    costs: GraphDataView<'float32'>;
    nodePositions: GraphDataView<'float32x2'>;
    parameters: GraphDataView<'float32'>;
    encoded: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  const isMin = props.mode === 'min';
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: `splat-${props.mode}`,
    bindings: [
      {name: 'offsets', view: props.offsets, type: 'u32', access: 'read'},
      {name: 'neighbors', view: props.neighbors, type: 'u32', access: 'read'},
      {name: 'weights', view: props.weights, type: 'f32', access: 'read'},
      {name: 'costs', view: props.costs, type: 'f32', access: 'read'},
      {name: 'positions', view: props.nodePositions, type: 'f32', access: 'read'},
      {name: 'params', view: props.parameters, type: 'f32', access: 'read'},
      {name: 'encoded', view: props.encoded, type: 'atomic<u32>', access: 'read_write'}
    ],
    invocationCount: props.neighbors.length,
    declarations: `const NODE_COUNT: u32 = ${props.nodeCount}u;
const WIDTH: i32 = ${props.width};
const HEIGHT: i32 = ${props.height};
const MAXIMUM_BUFFER_PIXELS: u32 = ${props.maximumBufferPixels}u;
const MAXIMUM_SAMPLES: u32 = ${props.maximumSamplesPerEdge}u;
const COST_CEILING: f32 = ${getWGSLFloatLiteral(3.0e38)};

fn isochronesIsFinite(value: f32) -> bool {
  return value == value && abs(value) < COST_CEILING;
}

fn isochronesFindSource(edge: u32) -> u32 {
  var low = 0u;
  var high = NODE_COUNT;
  while (low + 1u < high) {
    let middle = (low + high) >> 1u;
    if (offsets[offsetsOffset + middle] <= edge) {
      low = middle;
    } else {
      high = middle;
    }
  }
  return low;
}

fn isochronesGetPosition(node: u32) -> vec2f {
  return vec2f(positions[positionsOffset + 2u * node], positions[positionsOffset + 2u * node + 1u]);
}`,
    body: `if (index >= offsets[offsetsOffset + NODE_COUNT]) { return; }
  let targetNode = neighbors[neighborsOffset + index];
  if (targetNode >= NODE_COUNT) { return; }
  let weight = weights[weightsOffset + index];
  let source = isochronesFindSource(index);
  let sourceCost = costs[costsOffset + source];
  if (!(weight >= 0.0) || !isochronesIsFinite(weight) || !isochronesIsFinite(sourceCost) || sourceCost < 0.0) { return; }
  let startPoint = isochronesGetPosition(source);
  let endPoint = isochronesGetPosition(targetNode);
  if (!isochronesIsFinite(startPoint.x) || !isochronesIsFinite(startPoint.y) || !isochronesIsFinite(endPoint.x) || !isochronesIsFinite(endPoint.y)) { return; }
  let minimum = vec2f(params[paramsOffset + 1u], params[paramsOffset + 2u]);
  let cellSize = vec2f(
    (params[paramsOffset + 3u] - minimum.x) / f32(WIDTH),
    (params[paramsOffset + 4u] - minimum.y) / f32(HEIGHT)
  );
  let walk = max(params[paramsOffset + 6u], 0.0);
  let radius = max(max(params[paramsOffset + 5u], 0.0), 0.5 * length(cellSize));
  let reachX = i32(min(u32(ceil(radius / cellSize.x)), MAXIMUM_BUFFER_PIXELS));
  let reachY = i32(min(u32(ceil(radius / cellSize.y)), MAXIMUM_BUFFER_PIXELS));
  let spacing = 0.5 * radius;
  let sampleCount = clamp(u32(ceil(length(endPoint - startPoint) / spacing)) + 1u, 2u, MAXIMUM_SAMPLES);
  for (var sampleIndex = 0u; sampleIndex < sampleCount; sampleIndex++) {
    let t = f32(sampleIndex) / f32(sampleCount - 1u);
    let point = mix(startPoint, endPoint, vec2f(t));
    let edgeCost = sourceCost + t * weight;
    let centerPixel = vec2i(floor((point - minimum) / cellSize));
    for (var dy = -reachY; dy <= reachY; dy++) {
      let row = centerPixel.y + dy;
      if (row < 0 || row >= HEIGHT) { continue; }
      for (var dx = -reachX; dx <= reachX; dx++) {
        let column = centerPixel.x + dx;
        if (column < 0 || column >= WIDTH) { continue; }
        let pixelCenter = minimum + (vec2f(f32(column), f32(row)) + 0.5) * cellSize;
        let pixelDistance = length(pixelCenter - point);
        if (pixelDistance > radius) { continue; }
        let cost = abs(edgeCost + walk * pixelDistance);
        if (!isochronesIsFinite(cost)) { continue; }
        let slot = encodedOffset + u32(row * WIDTH + column);
        ${isMin ? 'atomicMin(&encoded[slot], bitcast<u32>(cost));' : 'atomicMax(&encoded[slot], bitcast<u32>(cost) + 1u);'}
      }
    }
  }`
  });
}

/** Decodes the accumulated pixels to float32 costs, `unreachedCost` where nothing landed. @internal */
export function createIsochronesDecodeNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    mode: GPUNetworkIsochronesMode;
    unreachedCost: number;
    encoded: GraphDataView<'uint32'>;
    values: GraphDataView<'float32'>;
    pixelCount: number;
  }
): GPUCommandNode<Parameters> {
  const isMin = props.mode === 'min';
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: `decode-${props.mode}`,
    bindings: [
      {name: 'encoded', view: props.encoded, type: 'u32', access: 'read'},
      {name: 'values', view: props.values, type: 'f32', access: 'read_write'}
    ],
    invocationCount: props.pixelCount,
    declarations: `const UNREACHED_COST: f32 = ${getWGSLFloatLiteral(props.unreachedCost)};`,
    body: `let word = encoded[encodedOffset + index];
  ${
    isMin
      ? 'values[valuesOffset + index] = select(bitcast<f32>(word), UNREACHED_COST, word == 0xffffffffu);'
      : 'values[valuesOffset + index] = select(bitcast<f32>(word - 1u), UNREACHED_COST, word == 0u);'
  }`
  });
}

/** One thread per node: mask of nodes with a finite cost at or below the cell cost limit. @internal */
export function createIsochronesNodeMaskNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    costs: GraphDataView<'float32'>;
    parameters: GraphDataView<'float32'>;
    mask: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'node-mask',
    bindings: [
      {name: 'costs', view: props.costs, type: 'f32', access: 'read'},
      {name: 'params', view: props.parameters, type: 'f32', access: 'read'},
      {name: 'mask', view: props.mask, type: 'u32', access: 'read_write'}
    ],
    invocationCount: props.costs.length,
    body: `let cost = costs[costsOffset + index];
  let isReached = cost == cost && abs(cost) < ${getWGSLFloatLiteral(3.0e38)} && cost <= params[paramsOffset + 9u];
  mask[maskOffset + index] = select(0u, 1u, isReached);`
  });
}
