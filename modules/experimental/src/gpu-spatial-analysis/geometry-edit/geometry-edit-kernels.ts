// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GPUCommandNode, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {createWGSLKernelNode, type WGSLKernelBinding} from '../../utils/wgsl-kernel-nodes';

/**
 * WGSL source of `readPoint(row)`, reading `positions` (a `float32x2` view bound as
 * `array<f32>`). @internal
 */
const READ_POINT_SOURCE = /* wgsl */ `
fn readPoint(row: u32) -> vec2<f32> {
  return vec2<f32>(positions[positionsOffset + 2u * row], positions[positionsOffset + 2u * row + 1u]);
}
`;

/**
 * Returns WGSL for `${functionName}(row)`: the largest `i` in `[0, ${countName})` with
 * `${offsetsName}[i] <= row`. Empty groups share an offset with their successor, so the search
 * skips them. @internal
 */
function getUpperBoundSource(functionName: string, offsetsName: string, countName: string): string {
  return /* wgsl */ `
fn ${functionName}(row: u32) -> u32 {
  var low = 0u;
  var high = ${countName};
  while (high - low > 1u) {
    let middle = (low + high) / 2u;
    if (${offsetsName}[${offsetsName}Offset + middle] <= row) {
      low = middle;
    } else {
      high = middle;
    }
  }
  return low;
}
`;
}

/** Properties for {@link createRingOrientationNode}. @internal */
export type RingOrientationNodeProps = {
  id: string;
  operation: string;
  /** `'reverse'` flags every ring; `'orient-polygons'` flags rings that wind the wrong way. */
  mode: 'reverse' | 'orient-polygons';
  positions: GraphDataView<'float32x2'>;
  ringOffsets: GraphDataView<'uint32'>;
  polygonOffsets?: GraphDataView<'uint32'>;
  parameters?: GraphDataView<'float32'>;
  /** One `uint32` row per ring, 1 when the ring must be reversed. */
  ringFlags: GraphDataView<'uint32'>;
};

/**
 * Builds the per-ring node that decides which rings to reverse. One invocation per ring sums the
 * shoelace area in coordinates relative to the ring's first vertex.
 *
 * @internal
 */
export function createRingOrientationNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: RingOrientationNodeProps
): GPUCommandNode<Parameters> {
  const isOrient = props.mode === 'orient-polygons';
  const bindings: WGSLKernelBinding[] = [
    {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
    {name: 'ringOffsets', view: props.ringOffsets, type: 'u32', access: 'read'}
  ];
  if (isOrient) {
    bindings.push(
      {name: 'polygonOffsets', view: props.polygonOffsets!, type: 'u32', access: 'read'},
      {name: 'parameters', view: props.parameters!, type: 'f32', access: 'read'}
    );
  }
  bindings.push({name: 'ringFlags', view: props.ringFlags, type: 'u32', access: 'read_write'});
  const ringCount = props.ringOffsets.length - 1;
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: props.operation,
    variant: 'ring-orientation',
    bindings,
    invocationCount: ringCount,
    declarations: `const VERTEX_COUNT: u32 = ${props.positions.length}u;
${
  isOrient
    ? `const POLYGON_COUNT: u32 = ${props.polygonOffsets!.length - 1}u;
${READ_POINT_SOURCE}
${getUpperBoundSource('findPolygon', 'polygonOffsets', 'POLYGON_COUNT')}`
    : ''
}`,
    body: isOrient
      ? /* wgsl */ `
  let start = ringOffsets[ringOffsetsOffset + index];
  let end = min(ringOffsets[ringOffsetsOffset + index + 1u], VERTEX_COUNT);
  var shouldReverse = false;
  if (end > start && end - start >= 3u) {
    let origin = readPoint(start);
    var previous = vec2<f32>(0.0, 0.0);
    var twiceArea = 0.0;
    for (var row = start + 1u; row < end; row++) {
      let current = readPoint(row) - origin;
      twiceArea += previous.x * current.y - current.x * previous.y;
      previous = current;
    }
    var isExterior = true;
    if (index >= polygonOffsets[polygonOffsetsOffset] &&
        index < polygonOffsets[polygonOffsetsOffset + POLYGON_COUNT]) {
      isExterior = polygonOffsets[polygonOffsetsOffset + findPolygon(index)] == index;
    }
    let exteriorClockwise = parameters[parametersOffset] != 0.0;
    let wantClockwise = select(!exteriorClockwise, exteriorClockwise, isExterior);
    shouldReverse = (twiceArea > 0.0 && wantClockwise) || (twiceArea < 0.0 && !wantClockwise);
  }
  ringFlags[ringFlagsOffset + index] = select(0u, 1u, shouldReverse);`
      : `ringFlags[ringFlagsOffset + index] = 1u;`
  });
}

/** Properties for {@link createRingReverseNode}. @internal */
export type RingReverseNodeProps = {
  id: string;
  operation: string;
  positions: GraphDataView<'float32x2'>;
  ringOffsets: GraphDataView<'uint32'>;
  ringFlags: GraphDataView<'uint32'>;
  outputPositions: GraphDataView<'float32x2'>;
};

/**
 * Builds the per-vertex node that copies positions, reversing the vertex order inside every
 * flagged ring. Vertices outside every ring are copied unchanged.
 *
 * @internal
 */
export function createRingReverseNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: RingReverseNodeProps
): GPUCommandNode<Parameters> {
  const ringCount = props.ringOffsets.length - 1;
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: props.operation,
    variant: 'ring-reverse',
    bindings: [
      {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
      {name: 'ringOffsets', view: props.ringOffsets, type: 'u32', access: 'read'},
      {name: 'ringFlags', view: props.ringFlags, type: 'u32', access: 'read'},
      {name: 'outputPositions', view: props.outputPositions, type: 'f32', access: 'read_write'}
    ],
    invocationCount: props.positions.length,
    declarations: `const VERTEX_COUNT: u32 = ${props.positions.length}u;
const RING_COUNT: u32 = ${ringCount}u;
${getUpperBoundSource('findRing', 'ringOffsets', 'RING_COUNT')}`,
    body: /* wgsl */ `
  var source = index;
  let first = ringOffsets[ringOffsetsOffset];
  let last = min(ringOffsets[ringOffsetsOffset + RING_COUNT], VERTEX_COUNT);
  if (index >= first && index < last) {
    let ring = findRing(index);
    if (ringFlags[ringFlagsOffset + ring] != 0u) {
      let start = ringOffsets[ringOffsetsOffset + ring];
      let end = min(ringOffsets[ringOffsetsOffset + ring + 1u], VERTEX_COUNT);
      source = start + end - 1u - index;
    }
  }
  outputPositions[outputPositionsOffset + 2u * index] = positions[positionsOffset + 2u * source];
  outputPositions[outputPositionsOffset + 2u * index + 1u] =
    positions[positionsOffset + 2u * source + 1u];`
  });
}

/** Properties for {@link createAffineTransformNode}. @internal */
export type AffineTransformNodeProps = {
  id: string;
  operation: string;
  positions: GraphDataView<'float32x2'>;
  ringOffsets?: GraphDataView<'uint32'>;
  featureRingOffsets?: GraphDataView<'uint32'>;
  /** Global per-frame parameters; mutually exclusive with `featureTransforms`. */
  parameters?: GraphDataView<'float32'>;
  /** `6 * featureCount` rows of `[a, b, d, e, xoff, yoff]`. */
  featureTransforms?: GraphDataView<'float32'>;
  /** Per-feature bounds; present when the `'center'` origin is compiled. */
  bounds?: GraphDataView<'float32x4'>;
  /** Per-feature centroids; present when the `'centroid'` origin is compiled. */
  centroids?: GraphDataView<'float32x2'>;
  featureCount: number;
  outputPositions: GraphDataView<'float32x2'>;
};

/**
 * Builds the per-vertex affine transform node.
 *
 * @internal
 */
export function createAffineTransformNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: AffineTransformNodeProps
): GPUCommandNode<Parameters> {
  const hasRings = Boolean(props.ringOffsets);
  const bindings: WGSLKernelBinding[] = [
    {name: 'positions', view: props.positions, type: 'f32', access: 'read'}
  ];
  if (props.ringOffsets) {
    bindings.push({name: 'ringOffsets', view: props.ringOffsets, type: 'u32', access: 'read'});
  }
  if (props.featureRingOffsets) {
    bindings.push({
      name: 'featureRingOffsets',
      view: props.featureRingOffsets,
      type: 'u32',
      access: 'read'
    });
  }
  if (props.parameters) {
    bindings.push({name: 'parameters', view: props.parameters, type: 'f32', access: 'read'});
  }
  if (props.featureTransforms) {
    bindings.push({
      name: 'featureTransforms',
      view: props.featureTransforms,
      type: 'f32',
      access: 'read'
    });
  }
  if (props.bounds) {
    bindings.push({name: 'bounds', view: props.bounds, type: 'f32', access: 'read'});
  }
  if (props.centroids) {
    bindings.push({name: 'centroids', view: props.centroids, type: 'f32', access: 'read'});
  }
  bindings.push({
    name: 'outputPositions',
    view: props.outputPositions,
    type: 'f32',
    access: 'read_write'
  });
  const featureLookup = hasRings
    ? `let ring = findRing(index);
  ${props.featureRingOffsets ? 'featureIndex = min(findFeature(ring), FEATURE_COUNT - 1u);' : 'featureIndex = min(ring, FEATURE_COUNT - 1u);'}`
    : '';
  const transform = props.featureTransforms
    ? `let base = 6u * featureIndex + featureTransformsOffset;
  let result = vec2<f32>(
    featureTransforms[base] * point.x + featureTransforms[base + 1u] * point.y +
      featureTransforms[base + 4u],
    featureTransforms[base + 2u] * point.x + featureTransforms[base + 3u] * point.y +
      featureTransforms[base + 5u]
  );`
    : `let a = parameters[parametersOffset];
  let b = parameters[parametersOffset + 1u];
  let d = parameters[parametersOffset + 2u];
  let e = parameters[parametersOffset + 3u];
  let translation = vec2<f32>(parameters[parametersOffset + 4u], parameters[parametersOffset + 5u]);
  var origin = vec2<f32>(parameters[parametersOffset + 6u], parameters[parametersOffset + 7u]);
  let originMode = u32(parameters[parametersOffset + 8u] + 0.5);
  ${
    props.bounds
      ? `if (originMode == 1u) {
    let box = vec4<f32>(
      bounds[boundsOffset + 4u * featureIndex],
      bounds[boundsOffset + 4u * featureIndex + 1u],
      bounds[boundsOffset + 4u * featureIndex + 2u],
      bounds[boundsOffset + 4u * featureIndex + 3u]
    );
    origin = 0.5 * (box.xy + box.zw);
  }`
      : ''
  }
  ${
    props.centroids
      ? `if (originMode == 2u) {
    origin = vec2<f32>(
      centroids[centroidsOffset + 2u * featureIndex],
      centroids[centroidsOffset + 2u * featureIndex + 1u]
    );
  }`
      : ''
  }
  let relative = point - origin;
  let result = vec2<f32>(a * relative.x + b * relative.y, d * relative.x + e * relative.y) +
    origin + translation;`;
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: props.operation,
    variant: 'affine',
    bindings,
    invocationCount: props.positions.length,
    declarations: `const FEATURE_COUNT: u32 = ${Math.max(props.featureCount, 1)}u;
${
  props.ringOffsets
    ? `const RING_COUNT: u32 = ${props.ringOffsets.length - 1}u;
${getUpperBoundSource('findRing', 'ringOffsets', 'RING_COUNT')}`
    : ''
}
${
  props.featureRingOffsets
    ? `const FEATURE_RING_COUNT: u32 = ${props.featureRingOffsets.length - 1}u;
${getUpperBoundSource('findFeature', 'featureRingOffsets', 'FEATURE_RING_COUNT')}`
    : ''
}
${READ_POINT_SOURCE}`,
    body: /* wgsl */ `
  let point = readPoint(index);
  var featureIndex = 0u;
  ${featureLookup}
  ${transform}
  outputPositions[outputPositionsOffset + 2u * index] = result.x;
  outputPositions[outputPositionsOffset + 2u * index + 1u] = result.y;`
  });
}

/** Properties for {@link createCleanupFlagNode}. @internal */
export type CleanupFlagNodeProps = {
  id: string;
  operation: string;
  positions: GraphDataView<'float32x2'>;
  ringOffsets: GraphDataView<'uint32'>;
  parameters: GraphDataView<'float32'>;
  isPolygon: boolean;
  /** One `uint32` row per vertex receiving 1 for kept vertices; cleared beforehand. */
  keepFlags: GraphDataView<'uint32'>;
  /** One-row counter of collapsed rings; cleared beforehand. */
  collapsedCount: GraphDataView<'uint32'>;
};

/** WGSL `snapPoint` helper shared by the cleanup kernels. @internal */
const SNAP_SOURCE = /* wgsl */ `
fn snapValue(value: f32, gridSize: f32) -> f32 {
  if (gridSize > 0.0) {
    return floor(value / gridSize + 0.5) * gridSize;
  }
  return value;
}

fn readSnappedPoint(row: u32, gridSize: f32) -> vec2<f32> {
  return vec2<f32>(
    snapValue(positions[positionsOffset + 2u * row], gridSize),
    snapValue(positions[positionsOffset + 2u * row + 1u], gridSize)
  );
}
`;

/** Properties for {@link createCleanupPointsNode}. @internal */
export type CleanupPointsNodeProps = {
  id: string;
  operation: string;
  positions: GraphDataView<'float32x2'>;
  parameters: GraphDataView<'float32'>;
  outputPositions: GraphDataView<'float32x2'>;
  /** One-row scalar receiving the clamped point count. */
  count: GraphDataView<'uint32'>;
  /** One-row scalar receiving `1` when the point capacity overflowed, otherwise `0`. */
  overflow: GraphDataView<'uint32'>;
  /** Optional one-row scalar receiving the unclamped point count. */
  requiredCount?: GraphDataView<'uint32'>;
};

/**
 * Builds the point-input cleanup node. Each row is an independent Point feature, so the kernel
 * snaps coordinates without applying repeated-point removal across rows and publishes the output
 * counts from invocation zero.
 *
 * @internal
 */
export function createCleanupPointsNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: CleanupPointsNodeProps
): GPUCommandNode<Parameters> {
  const bindings: WGSLKernelBinding[] = [
    {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
    {name: 'parameters', view: props.parameters, type: 'f32', access: 'read'},
    {name: 'outputPositions', view: props.outputPositions, type: 'f32', access: 'read_write'},
    {name: 'countOut', view: props.count, type: 'u32', access: 'read_write'},
    {name: 'overflowOut', view: props.overflow, type: 'u32', access: 'read_write'}
  ];
  if (props.requiredCount) {
    bindings.push({name: 'totalOut', view: props.requiredCount, type: 'u32', access: 'read_write'});
  }
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: props.operation,
    variant: 'cleanup-points',
    bindings,
    invocationCount: props.positions.length,
    declarations: `const POINT_COUNT: u32 = ${props.positions.length}u;
const CAPACITY: u32 = ${props.outputPositions.length}u;
${SNAP_SOURCE}`,
    body: /* wgsl */ `
  if (index == 0u) {
    countOut[countOutOffset] = min(POINT_COUNT, CAPACITY);
    overflowOut[overflowOutOffset] = select(0u, 1u, POINT_COUNT > CAPACITY);
    ${props.requiredCount ? 'totalOut[totalOutOffset] = POINT_COUNT;' : ''}
  }
  if (index >= CAPACITY) {
    return;
  }
  let point = readSnappedPoint(index, parameters[parametersOffset]);
  outputPositions[outputPositionsOffset + 2u * index] = point.x;
  outputPositions[outputPositionsOffset + 2u * index + 1u] = point.y;`
  });
}

/**
 * Builds the per-ring node that decides which vertices survive snapping and repeated-point
 * removal. One invocation per ring walks its vertices in order, so the result matches GEOS
 * `remove_repeated_points` exactly (first and last vertex kept, others dropped when within the
 * tolerance of the previously kept vertex, a kept vertex within tolerance of the last vertex
 * replaced by it).
 *
 * @internal
 */
export function createCleanupFlagNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: CleanupFlagNodeProps
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: props.operation,
    variant: 'cleanup-flags',
    bindings: [
      {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
      {name: 'ringOffsets', view: props.ringOffsets, type: 'u32', access: 'read'},
      {name: 'parameters', view: props.parameters, type: 'f32', access: 'read'},
      {name: 'keepFlags', view: props.keepFlags, type: 'u32', access: 'read_write'},
      {
        name: 'collapsedCount',
        view: props.collapsedCount,
        type: 'atomic<u32>',
        access: 'read_write'
      }
    ],
    invocationCount: props.ringOffsets.length - 1,
    declarations: `const VERTEX_COUNT: u32 = ${props.positions.length}u;
const IS_POLYGON: bool = ${props.isPolygon ? 'true' : 'false'};
${SNAP_SOURCE}
fn isWithinTolerance(delta: vec2<f32>, tolerance: f32) -> bool {
  if (tolerance > 0.0) {
    return dot(delta, delta) <= tolerance * tolerance;
  }
  return delta.x == 0.0 && delta.y == 0.0;
}`,
    body: /* wgsl */ `
  let gridSize = parameters[parametersOffset];
  let tolerance = max(parameters[parametersOffset + 1u], 0.0);
  let removeRepeated = parameters[parametersOffset + 2u] != 0.0;
  let start = ringOffsets[ringOffsetsOffset + index];
  let end = min(ringOffsets[ringOffsetsOffset + index + 1u], VERTEX_COUNT);
  if (end <= start) {
    return;
  }
  if (!removeRepeated || end - start < 2u) {
    for (var row = start; row < end; row++) {
      keepFlags[keepFlagsOffset + row] = 1u;
    }
    return;
  }
  var lastKept = start;
  var keptCount = 1u;
  keepFlags[keepFlagsOffset + start] = 1u;
  var lastKeptPoint = readSnappedPoint(start, gridSize);
  for (var row = start + 1u; row + 1u < end; row++) {
    let point = readSnappedPoint(row, gridSize);
    if (isWithinTolerance(point - lastKeptPoint, tolerance)) {
      keepFlags[keepFlagsOffset + row] = 0u;
    } else {
      keepFlags[keepFlagsOffset + row] = 1u;
      lastKept = row;
      lastKeptPoint = point;
      keptCount++;
    }
  }
  let finalRow = end - 1u;
  keepFlags[keepFlagsOffset + finalRow] = 1u;
  keptCount++;
  if (lastKept != start &&
      isWithinTolerance(readSnappedPoint(finalRow, gridSize) - lastKeptPoint, tolerance)) {
    keepFlags[keepFlagsOffset + lastKept] = 0u;
    keptCount--;
  }
  if (IS_POLYGON && keptCount < 3u) {
    for (var row = start; row < end; row++) {
      keepFlags[keepFlagsOffset + row] = 0u;
    }
    atomicAdd(&collapsedCount[collapsedCountOffset], 1u);
  }`
  });
}

/** Properties for {@link createCleanupEmitNode}. @internal */
export type CleanupEmitNodeProps = {
  id: string;
  operation: string;
  positions: GraphDataView<'float32x2'>;
  parameters: GraphDataView<'float32'>;
  keepFlags: GraphDataView<'uint32'>;
  /** Exclusive scan of `keepFlags`. */
  ranks: GraphDataView<'uint32'>;
  outputPositions: GraphDataView<'float32x2'>;
};

/**
 * Builds the per-vertex node that writes the snapped surviving vertices to their compacted rows.
 *
 * @internal
 */
export function createCleanupEmitNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: CleanupEmitNodeProps
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: props.operation,
    variant: 'cleanup-emit',
    bindings: [
      {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
      {name: 'parameters', view: props.parameters, type: 'f32', access: 'read'},
      {name: 'keepFlags', view: props.keepFlags, type: 'u32', access: 'read'},
      {name: 'ranks', view: props.ranks, type: 'u32', access: 'read'},
      {name: 'outputPositions', view: props.outputPositions, type: 'f32', access: 'read_write'}
    ],
    invocationCount: props.positions.length,
    declarations: `const CAPACITY: u32 = ${props.outputPositions.length}u;
${SNAP_SOURCE}`,
    body: /* wgsl */ `
  if (keepFlags[keepFlagsOffset + index] == 0u) {
    return;
  }
  let outputRow = ranks[ranksOffset + index];
  if (outputRow >= CAPACITY) {
    return;
  }
  let point = readSnappedPoint(index, parameters[parametersOffset]);
  outputPositions[outputPositionsOffset + 2u * outputRow] = point.x;
  outputPositions[outputPositionsOffset + 2u * outputRow + 1u] = point.y;`
  });
}

/** Properties for {@link createCleanupPublishNode}. @internal */
export type CleanupPublishNodeProps = {
  id: string;
  operation: string;
  ringOffsets: GraphDataView<'uint32'>;
  keepFlags: GraphDataView<'uint32'>;
  ranks: GraphDataView<'uint32'>;
  outputRingOffsets: GraphDataView<'uint32'>;
  count: GraphDataView<'uint32'>;
  overflow: GraphDataView<'uint32'>;
  requiredCount?: GraphDataView<'uint32'>;
  /** Vertex capacity of the output positions. */
  capacity: number;
};

/**
 * Builds the per-ring-offset node that republishes ring offsets in the compacted layout, clamped
 * to the vertex capacity, and writes the count, overflow and total count words.
 *
 * @internal
 */
export function createCleanupPublishNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: CleanupPublishNodeProps
): GPUCommandNode<Parameters> {
  const vertexCount = props.keepFlags.length;
  const bindings: WGSLKernelBinding[] = [
    {name: 'ringOffsets', view: props.ringOffsets, type: 'u32', access: 'read'},
    {name: 'keepFlags', view: props.keepFlags, type: 'u32', access: 'read'},
    {name: 'ranks', view: props.ranks, type: 'u32', access: 'read'},
    {name: 'outputRingOffsets', view: props.outputRingOffsets, type: 'u32', access: 'read_write'},
    {name: 'countOut', view: props.count, type: 'u32', access: 'read_write'},
    {name: 'overflowOut', view: props.overflow, type: 'u32', access: 'read_write'}
  ];
  if (props.requiredCount) {
    bindings.push({name: 'totalOut', view: props.requiredCount, type: 'u32', access: 'read_write'});
  }
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: props.operation,
    variant: 'cleanup-publish',
    bindings,
    invocationCount: props.ringOffsets.length,
    declarations: `const VERTEX_COUNT: u32 = ${vertexCount}u;
const CAPACITY: u32 = ${props.capacity}u;
const RING_OFFSET_COUNT: u32 = ${props.ringOffsets.length}u;`,
    body: /* wgsl */ `
  let total = select(
    ranks[ranksOffset + VERTEX_COUNT - 1u] + keepFlags[keepFlagsOffset + VERTEX_COUNT - 1u],
    0u,
    VERTEX_COUNT == 0u
  );
  let vertex = min(ringOffsets[ringOffsetsOffset + index], VERTEX_COUNT);
  let rank = select(ranks[ranksOffset + vertex], total, vertex >= VERTEX_COUNT);
  outputRingOffsets[outputRingOffsetsOffset + index] = min(rank, CAPACITY);
  if (index == 0u) {
    countOut[countOutOffset] = min(total, CAPACITY);
    overflowOut[overflowOutOffset] = select(0u, 1u, total > CAPACITY);
    ${props.requiredCount ? 'totalOut[totalOutOffset] = total;' : ''}
  }`
  });
}
