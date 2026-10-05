// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GraphVectorView,
  validatePackedUint32View,
  validatePackedView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {createMapGraphKernelNode, type MapGraphKernelBinding} from '../map-graph-kernels';
import type {GPUMapGraphPositions2D, GPUMapGraphRecipe} from '../map-graph-types';
import {
  getGraphViewChunks,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../map-graph-utils';
import type {GPURegionShape} from './region-statistics-types';

/** Number of float32 elements in a region screen transform. */
export const GPU_REGION_SCREEN_TRANSFORM_LENGTH = 20;

/** WGSL helper that rejects NaN and infinities. @internal */
export const REGION_STATISTICS_WGSL_HELPERS = /* wgsl */ `
fn isFiniteValue(value: f32) -> bool { return value == value && abs(value) <= 3.402823466e+38; }`;

/** Properties for {@link GPURegionMask}. */
export type GPURegionMaskProps = {
  /** ID prefix. Defaults to `'region-mask'`. */
  id?: string;
  /** Packed 2D world positions. The topology is compile-time. */
  positions: GPUMapGraphPositions2D;
  /** Rectangle or lasso polygon, optionally in screen space. Shape contents are per-frame. */
  region: GPURegionShape;
  /** Packed mask with `length == positions.length`. Every row is written with 0 or 1. */
  outputMask: GraphDataView<'uint32'>;
  /** One-row flag: 1 when the polygon vertex count exceeds capacity, else 0. Always written. */
  overflow: GraphDataView<'uint32'>;
};

/**
 * Writes a 0/1 source-aligned mask of points inside a per-frame rectangle or even-odd lasso
 * polygon, in world space or in screen pixels through a view-projection transform.
 */
export class GPURegionMask implements GPUMapGraphRecipe {
  /** Prefix for every node ID. */
  readonly id: string;
  /** Stable recipe name. */
  readonly recipe = 'region-mask';
  /** Validated properties. */
  readonly props: GPURegionMaskProps;

  constructor(props: GPURegionMaskProps) {
    this.id = props.id ?? this.recipe;
    this.props = props;
    const {id} = this;
    const {positions, region, outputMask, overflow} = props;
    for (const chunk of getGraphViewChunks(positions)) {
      validatePackedView(chunk, ['float32x2'], `${id} positions`);
    }
    validatePackedUint32View(outputMask, `${id} outputMask`);
    if (outputMask.length !== positions.length) {
      throw new Error(`${id} outputMask length must equal positions length`);
    }
    validatePackedUint32View(overflow, `${id} overflow`);
    if (overflow.length < 1) {
      throw new Error(`${id} overflow must contain one uint32 row`);
    }
    if (region.kind === 'rectangle') {
      validatePackedView(region.bounds, ['float32'], `${id} bounds`);
      if (region.bounds.length !== 4) {
        throw new Error(`${id} bounds must contain four float32 values`);
      }
    } else {
      validatePackedView(region.vertices, ['float32x2'], `${id} vertices`);
      if (region.vertices.length < 3) {
        throw new Error(`${id} vertices must hold at least three vertices`);
      }
      validatePackedUint32View(region.vertexCount, `${id} vertexCount`);
      if (region.vertexCount.length < 1) {
        throw new Error(`${id} vertexCount must contain one uint32 row`);
      }
    }
    if (region.screenTransform) {
      validatePackedView(region.screenTransform, ['float32'], `${id} screenTransform`);
      if (region.screenTransform.length !== GPU_REGION_SCREEN_TRANSFORM_LENGTH) {
        throw new Error(`${id} screenTransform must contain 20 float32 values`);
      }
    }
    validateGraphOutputsDisjointFromInputs(id, [outputMask, overflow], getRegionInputs(props));
    if (outputMask.buffer === overflow.buffer) {
      throw new Error(`${id} outputMask and overflow must use separate buffers`);
    }
  }

  /** Returns the overflow node followed by one mask node per nonempty position chunk. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {positions, region, outputMask, overflow} = props;
    validateGraphViewsBelongToGraph(id, graph, [outputMask, overflow, ...getRegionInputs(props)]);
    const isPolygon = region.kind === 'polygon';
    const shape = region.kind === 'rectangle' ? region.bounds : region.vertices;
    const vertexCapacity = region.kind === 'polygon' ? region.vertices.length : 0;
    const nodes: GPUCommandNode<Parameters>[] = [];

    const overflowBindings: MapGraphKernelBinding[] = [
      {name: 'overflow', view: overflow, type: 'u32', access: 'read_write'}
    ];
    if (region.kind === 'polygon') {
      overflowBindings.push({
        name: 'vertexCount',
        view: region.vertexCount,
        type: 'u32',
        access: 'read'
      });
    }
    nodes.push(
      createMapGraphKernelNode<Parameters>(graph, {
        id: `${id}-overflow`,
        operation: 'GPURegionMask',
        variant: 'overflow',
        bindings: overflowBindings,
        invocationCount: 1,
        body: isPolygon
          ? `overflow[overflowOffset] = select(0u, 1u, vertexCount[vertexCountOffset] > ${vertexCapacity}u);`
          : 'overflow[overflowOffset] = 0u;'
      })
    );

    const toRegionSpace = region.screenTransform
      ? /* wgsl */ `
struct RegionPoint { position: vec2f, valid: bool }
fn toRegionSpace(world: vec2f) -> RegionPoint {
  let t = screenTransformOffset;
  let column0 = vec4f(screenTransform[t], screenTransform[t + 1u], screenTransform[t + 2u], screenTransform[t + 3u]);
  let column1 = vec4f(screenTransform[t + 4u], screenTransform[t + 5u], screenTransform[t + 6u], screenTransform[t + 7u]);
  let column3 = vec4f(screenTransform[t + 12u], screenTransform[t + 13u], screenTransform[t + 14u], screenTransform[t + 15u]);
  let clip = column0 * world.x + column1 * world.y + column3;
  if (!(clip.w > 0.0)) { return RegionPoint(vec2f(0.0), false); }
  let ndc = clip.xy / clip.w;
  let pixel = vec2f(
    (ndc.x * 0.5 + 0.5) * screenTransform[t + 16u],
    (0.5 - ndc.y * 0.5) * screenTransform[t + 17u]
  );
  return RegionPoint(pixel, isFiniteValue(pixel.x) && isFiniteValue(pixel.y));
}`
      : /* wgsl */ `
struct RegionPoint { position: vec2f, valid: bool }
fn toRegionSpace(world: vec2f) -> RegionPoint { return RegionPoint(world, true); }`;
    const isInsideRegion = isPolygon
      ? /* wgsl */ `
fn readVertex(vertexIndex: u32) -> vec2f {
  return vec2f(shape[shapeOffset + vertexIndex * 2u], shape[shapeOffset + vertexIndex * 2u + 1u]);
}
// Even-odd crossing test (PNPOLY, half-open in y).
fn isInsideRegion(point: vec2f) -> bool {
  let activeCount = min(vertexCount[vertexCountOffset], ${vertexCapacity}u);
  if (activeCount < 3u) { return false; }
  var inside = false;
  var previous = readVertex(activeCount - 1u);
  for (var vertexIndex = 0u; vertexIndex < activeCount; vertexIndex++) {
    let current = readVertex(vertexIndex);
    if ((current.y > point.y) != (previous.y > point.y)) {
      let crossingX = (previous.x - current.x) * (point.y - current.y) / (previous.y - current.y) + current.x;
      if (point.x < crossingX) { inside = !inside; }
    }
    previous = current;
  }
  return inside;
}`
      : /* wgsl */ `
// Inclusive bounds; invalid bounds (non-finite or min > max) select nothing.
fn isInsideRegion(point: vec2f) -> bool {
  let minimum = vec2f(shape[shapeOffset], shape[shapeOffset + 1u]);
  let maximum = vec2f(shape[shapeOffset + 2u], shape[shapeOffset + 3u]);
  let validBounds = isFiniteValue(minimum.x) && isFiniteValue(minimum.y) &&
    isFiniteValue(maximum.x) && isFiniteValue(maximum.y) && all(minimum <= maximum);
  return validBounds && all(point >= minimum) && all(point <= maximum);
}`;

    const isVector = positions instanceof GraphVectorView;
    let rowStart = 0;
    for (const [chunkIndex, chunk] of getGraphViewChunks(positions).entries()) {
      if (chunk.length > 0) {
        const bindings: MapGraphKernelBinding[] = [
          {name: 'positions', view: chunk, type: 'f32', access: 'read'},
          {name: 'shape', view: shape, type: 'f32', access: 'read'},
          {name: 'outputMask', view: outputMask, type: 'u32', access: 'read_write'}
        ];
        if (region.kind === 'polygon') {
          bindings.push({
            name: 'vertexCount',
            view: region.vertexCount,
            type: 'u32',
            access: 'read'
          });
        }
        if (region.screenTransform) {
          bindings.push({
            name: 'screenTransform',
            view: region.screenTransform,
            type: 'f32',
            access: 'read'
          });
        }
        nodes.push(
          createMapGraphKernelNode<Parameters>(graph, {
            id: isVector ? `${id}-chunk-${chunkIndex}` : id,
            operation: 'GPURegionMask',
            variant: `${region.kind}${region.screenTransform ? '-screen' : ''}`,
            bindings,
            invocationCount: chunk.length,
            declarations: `const ROW_START: u32 = ${rowStart}u;
${REGION_STATISTICS_WGSL_HELPERS}
${toRegionSpace}
${isInsideRegion}`,
            body: `let world = vec2f(positions[positionsOffset + index * 2u], positions[positionsOffset + index * 2u + 1u]);
  var selected = false;
  if (isFiniteValue(world.x) && isFiniteValue(world.y)) {
    let regionPoint = toRegionSpace(world);
    selected = regionPoint.valid && isInsideRegion(regionPoint.position);
  }
  outputMask[outputMaskOffset + ROW_START + index] = select(0u, 1u, selected);`
          })
        );
      }
      rowStart += chunk.length;
    }
    return nodes;
  }
}

/** Returns every read-only view of a region mask. */
function getRegionInputs(props: GPURegionMaskProps): (GraphDataView | GraphVectorView)[] {
  const {region} = props;
  return [
    props.positions,
    ...(region.kind === 'rectangle' ? [region.bounds] : [region.vertices, region.vertexCount]),
    ...(region.screenTransform ? [region.screenTransform] : [])
  ];
}
