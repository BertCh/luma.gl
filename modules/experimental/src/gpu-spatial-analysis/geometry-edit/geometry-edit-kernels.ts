// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {
  GPUCommandGraph,
  GPUCommandGraphNodeCondition,
  GPUCommandNode,
  GraphBufferUse,
  GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {
  createWGSLDispatchGate,
  createWGSLKernelNode,
  type WGSLDispatchGate,
  type WGSLKernelBinding
} from '../../utils/wgsl-kernel-nodes';
import {IS_FINITE_BITS_WGSL} from '../../utils/wgsl-common';
import {EXACT_ORIENTATION_WGSL} from '../../utils/exact-orientation-wgsl';

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
 * winding at the ring's lexicographically smallest vertex with exact orientation.
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
${EXACT_ORIENTATION_WGSL}
${getUpperBoundSource('findPolygon', 'polygonOffsets', 'POLYGON_COUNT')}`
    : ''
}`,
    body: isOrient
      ? /* wgsl */ `
  let start = ringOffsets[ringOffsetsOffset + index];
  let end = min(ringOffsets[ringOffsetsOffset + index + 1u], VERTEX_COUNT);
  var shouldReverse = false;
  if (end > start && end - start >= 3u) {
    // Winding at the lexicographically smallest vertex (first on ties), decided by exact
    // orientation so sliver rings are not misjudged by f32 shoelace cancellation.
    var extreme = start;
    var center = readPoint(start);
    for (var row = start + 1u; row < end; row++) {
      let candidate = readPoint(row);
      if (candidate.x < center.x || (candidate.x == center.x && candidate.y < center.y)) {
        extreme = row;
        center = candidate;
      }
    }
    let count = end - start;
    var previous = center;
    var next = center;
    var foundPrevious = false;
    var foundNext = false;
    for (var step = 1u; step < count; step++) {
      if (!foundPrevious) {
        previous = readPoint(start + (extreme - start + count - step) % count);
        foundPrevious = any(previous != center);
      }
      if (!foundNext) {
        next = readPoint(start + (extreme - start + step) % count);
        foundNext = any(next != center);
      }
      if (foundPrevious && foundNext) { break; }
    }
    var winding = 0;
    if (foundPrevious && foundNext) { winding = orientSign(previous, center, next); }
    // Non-finite or uncertain winding leaves the ring as stored.
    if (winding == 2) { winding = 0; }
    var isExterior = true;
    if (index >= polygonOffsets[polygonOffsetsOffset] &&
        index < polygonOffsets[polygonOffsetsOffset + POLYGON_COUNT]) {
      isExterior = polygonOffsets[polygonOffsetsOffset + findPolygon(index)] == index;
    }
    let exteriorClockwise = parameters[parametersOffset] != 0.0;
    let wantClockwise = select(!exteriorClockwise, exteriorClockwise, isExterior);
    shouldReverse = (winding > 0 && wantClockwise) || (winding < 0 && !wantClockwise);
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
  /** Vertex offsets per ring (line, or multi-point feature). */
  ringOffsets?: GraphDataView<'uint32'>;
  /** Feature-to-ring offsets (or, with `polygonOffsets`, feature-to-polygon offsets). */
  featureOffsets?: GraphDataView<'uint32'>;
  /** Polygon-to-ring offsets. */
  polygonOffsets?: GraphDataView<'uint32'>;
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
  if (props.featureOffsets) {
    bindings.push({
      name: 'featureOffsets',
      view: props.featureOffsets,
      type: 'u32',
      access: 'read'
    });
  }
  if (props.polygonOffsets) {
    bindings.push({
      name: 'polygonOffsets',
      view: props.polygonOffsets,
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
  // Vertex -> ring -> (polygon ->) feature.
  const featureLookup = hasRings
    ? `let ring = findRing(index);
  ${
    props.polygonOffsets
      ? 'featureIndex = min(findFeature(findPolygon(ring)), FEATURE_COUNT - 1u);'
      : props.featureOffsets
        ? 'featureIndex = min(findFeature(ring), FEATURE_COUNT - 1u);'
        : 'featureIndex = min(ring, FEATURE_COUNT - 1u);'
  }`
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
  props.polygonOffsets
    ? `const POLYGON_COUNT: u32 = ${props.polygonOffsets.length - 1}u;
${getUpperBoundSource('findPolygon', 'polygonOffsets', 'POLYGON_COUNT')}`
    : ''
}
${
  props.featureOffsets
    ? `const FEATURE_OFFSET_COUNT: u32 = ${props.featureOffsets.length - 1}u;
${getUpperBoundSource('findFeature', 'featureOffsets', 'FEATURE_OFFSET_COUNT')}`
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
export type CleanupFlagNodeProps<Parameters = unknown> = {
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
  /** Optional gate, for the sequential pass of a {@link createCleanupRepeatGate} slot. */
  condition?: GPUCommandGraphNodeCondition<Parameters>;
  /** Declared uses of the gate buffer. */
  extraResources?: readonly GraphBufferUse[];
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

/**
 * WGSL `isWithinTolerance`: inclusive Euclidean test for `tolerance > 0`, otherwise exact
 * equality of the finite operands (so a NaN or infinite coordinate is never a repeat, and
 * `-0` equals `+0`). The tolerance-0 vertex-parallel pass calls it with `0.0` so both passes share
 * one definition. @internal
 */
const WITHIN_TOLERANCE_SOURCE = /* wgsl */ `
${IS_FINITE_BITS_WGSL}
fn isWithinTolerance(first: vec2<f32>, second: vec2<f32>, tolerance: f32) -> bool {
  // Test operands before subtraction: Metal can fold (a - b) == 0 into a == b,
  // which would incorrectly remove repeated infinities in the constant-tolerance path.
  if (!isFiniteBits(first.x) || !isFiniteBits(first.y) ||
      !isFiniteBits(second.x) || !isFiniteBits(second.y)) { return false; }
  let delta = first - second;
  if (tolerance > 0.0) {
    return dot(delta, delta) <= tolerance * tolerance;
  }
  return delta.x == 0.0 && delta.y == 0.0;
}`;

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
  props: CleanupFlagNodeProps<Parameters>
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
    condition: props.condition,
    extraResources: props.extraResources,
    declarations: `const VERTEX_COUNT: u32 = ${props.positions.length}u;
const IS_POLYGON: bool = ${props.isPolygon ? 'true' : 'false'};
${SNAP_SOURCE}
${WITHIN_TOLERANCE_SOURCE}`,
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
    if (isWithinTolerance(point, lastKeptPoint, tolerance)) {
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
      isWithinTolerance(readSnappedPoint(finalRow, gridSize), lastKeptPoint, tolerance)) {
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

/** Indirect dispatch slots of {@link createCleanupRepeatGate}. @internal */
const REPEAT_GATE_SLOT_SEQUENTIAL = 0;
const REPEAT_GATE_SLOT_FLAG = 1;
const REPEAT_GATE_SLOT_RESOLVE = 2;
const REPEAT_GATE_SLOT_APPLY = 3;

/**
 * Vertices per invocation of the tolerance-0 flag pass. Each invocation walks a short chunk and
 * issues one atomic per ring segment instead of one per vertex, so a very long ring costs one
 * atomic per `CLEANUP_REPEAT_CHUNK` vertices. @internal
 */
export const CLEANUP_REPEAT_CHUNK = 32;

/** `uint32` words per ring of the tolerance-0 ring state: kept interior count, last, resolution. */
export const CLEANUP_RING_STATE_WORDS = 3;

/** Ring resolution word value of a ring emptied because too few vertices survived. @internal */
const RING_COLLAPSED = '0xffffffffu';

/**
 * Creates the indirect dispatch gate that selects, on the GPU and per frame, between the
 * sequential per-ring flag pass (`tolerance > 0`) and the tolerance-0 vertex-parallel passes
 * (slots for flag, resolve and apply). The gate is written by {@link createCleanupRepeatGateNode}.
 *
 * @internal
 */
export function createCleanupRepeatGate<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {id: string; operation: string; vertexCount: number; ringCount: number}
): WGSLDispatchGate<Parameters> {
  return createWGSLDispatchGate(graph, {
    id: props.id,
    operation: props.operation,
    shapes: [
      [props.ringCount],
      [Math.ceil(props.vertexCount / CLEANUP_REPEAT_CHUNK)],
      [props.ringCount],
      [props.vertexCount]
    ]
  });
}

/** Returns the gate condition and resources of one pass of {@link createCleanupRepeatGate}. @internal */
export function getCleanupRepeatGate<Parameters>(
  gate: WGSLDispatchGate<Parameters>,
  pass: 'sequential' | 'flag' | 'resolve' | 'apply',
  nodeId: string
): {condition: GPUCommandGraphNodeCondition<Parameters>; extraResources: GraphBufferUse[]} {
  const slot = {
    sequential: REPEAT_GATE_SLOT_SEQUENTIAL,
    flag: REPEAT_GATE_SLOT_FLAG,
    resolve: REPEAT_GATE_SLOT_RESOLVE,
    apply: REPEAT_GATE_SLOT_APPLY
  }[pass];
  const {condition, extraResources} = gate.getGate(slot);
  return {condition: {...condition, id: `${nodeId}-gate`}, extraResources};
}

/**
 * Builds the one-thread node that arms either the sequential pass or the three tolerance-0
 * passes. The choice mirrors the sequential kernel: `max(tolerance, 0) > 0` selects sequential,
 * so a zero (or NaN) tolerance takes the exact-repeat path.
 *
 * @internal
 */
export function createCleanupRepeatGateNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    parameters: GraphDataView<'float32'>;
    gate: WGSLDispatchGate<Parameters>;
  }
): GPUCommandNode<Parameters> {
  const disarm = (slot: number) => `dispatch[dispatchOffset + ${3 * slot}u] = 0u;`;
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: props.operation,
    variant: 'cleanup-repeat-gate',
    bindings: [
      {name: 'parameters', view: props.parameters, type: 'f32', access: 'read'},
      props.gate.binding
    ],
    invocationCount: 1,
    body: /* wgsl */ `
  ${props.gate.armWGSL}
  if (max(parameters[parametersOffset + 1u], 0.0) > 0.0) {
    ${disarm(REPEAT_GATE_SLOT_FLAG)}
    ${disarm(REPEAT_GATE_SLOT_RESOLVE)}
    ${disarm(REPEAT_GATE_SLOT_APPLY)}
  } else {
    ${disarm(REPEAT_GATE_SLOT_SEQUENTIAL)}
  }`
  });
}

/** Properties shared by the tolerance-0 passes. @internal */
type CleanupRepeatNodeProps<Parameters> = {
  id: string;
  operation: string;
  positions: GraphDataView<'float32x2'>;
  ringOffsets: GraphDataView<'uint32'>;
  parameters: GraphDataView<'float32'>;
  /** Cleared per-ring state, {@link CLEANUP_RING_STATE_WORDS} `uint32` words per ring. */
  ringState: GraphDataView<'uint32'>;
  condition: GPUCommandGraphNodeCondition<Parameters>;
  extraResources: readonly GraphBufferUse[];
};

/**
 * Builds the tolerance-0 flag pass. Vertex `i` of a ring of at least two vertices is kept when it
 * is the first or last vertex, or when its snapped position differs from the snapped position of
 * vertex `i - 1` (the exact-repeat test of the sequential pass, which compares against the last
 * kept vertex: a dropped vertex equals that vertex, so comparing with the previous vertex is
 * equivalent). Each invocation handles {@link CLEANUP_REPEAT_CHUNK} consecutive vertices, so work
 * per invocation is constant, and folds the kept interior vertices of each ring segment into the
 * ring's count and last index in `ringState` with one atomic pair per segment, for
 * {@link createCleanupRepeatResolveNode}. Rings below two vertices, and every ring when
 * repeated-point removal is off, keep all vertices; vertices outside every ring stay unflagged.
 *
 * @internal
 */
export function createCleanupRepeatFlagNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: CleanupRepeatNodeProps<Parameters> & {keepFlags: GraphDataView<'uint32'>}
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: props.operation,
    variant: 'cleanup-repeat-flags',
    bindings: [
      {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
      {name: 'ringOffsets', view: props.ringOffsets, type: 'u32', access: 'read'},
      {name: 'parameters', view: props.parameters, type: 'f32', access: 'read'},
      {name: 'keepFlags', view: props.keepFlags, type: 'u32', access: 'read_write'},
      {name: 'ringState', view: props.ringState, type: 'atomic<u32>', access: 'read_write'}
    ],
    invocationCount: Math.ceil(props.positions.length / CLEANUP_REPEAT_CHUNK),
    condition: props.condition,
    extraResources: props.extraResources,
    declarations: `const VERTEX_COUNT: u32 = ${props.positions.length}u;
const RING_COUNT: u32 = ${props.ringOffsets.length - 1}u;
const CHUNK: u32 = ${CLEANUP_REPEAT_CHUNK}u;
${SNAP_SOURCE}
${WITHIN_TOLERANCE_SOURCE}
${getUpperBoundSource('findRing', 'ringOffsets', 'RING_COUNT')}`,
    body: /* wgsl */ `
  let chunkEnd = min((index + 1u) * CHUNK, min(ringOffsets[ringOffsetsOffset + RING_COUNT], VERTEX_COUNT));
  var row = max(index * CHUNK, ringOffsets[ringOffsetsOffset]);
  let removeRepeated = parameters[parametersOffset + 2u] != 0.0;
  let gridSize = parameters[parametersOffset];
  while (row < chunkEnd) {
    let ring = findRing(row);
    let start = ringOffsets[ringOffsetsOffset + ring];
    let end = min(ringOffsets[ringOffsetsOffset + ring + 1u], VERTEX_COUNT);
    if (row < start || row >= end) {
      row++;
      continue;
    }
    let segmentEnd = min(end, chunkEnd);
    let plain = !removeRepeated || end - start < 2u;
    var count = 0u;
    var last = 0u;
    for (; row < segmentEnd; row++) {
      if (plain || row == start || row + 1u == end) {
        keepFlags[keepFlagsOffset + row] = 1u;
        continue;
      }
      if (isWithinTolerance(readSnappedPoint(row, gridSize), readSnappedPoint(row - 1u, gridSize), 0.0)) {
        keepFlags[keepFlagsOffset + row] = 0u;
      } else {
        keepFlags[keepFlagsOffset + row] = 1u;
        count++;
        last = row - start;
      }
    }
    if (count > 0u) {
      let stateBase = ringStateOffset + ${CLEANUP_RING_STATE_WORDS}u * ring;
      atomicAdd(&ringState[stateBase], count);
      atomicMax(&ringState[stateBase + 1u], last);
    }
  }`
  });
}

/**
 * Builds the tolerance-0 per-ring resolve pass (constant work per ring). It applies the two ring
 * rules of the sequential pass: the last kept interior vertex is replaced by the final vertex
 * when the final vertex repeats it (never when that vertex is the first one), and a polygon ring
 * left with fewer than 3 vertices is emptied and counted. The outcome is written to the ring's
 * resolution word: 0 for no change, the ring-relative index of the replaced vertex, or
 * `0xffffffff` for a collapsed ring.
 *
 * @internal
 */
export function createCleanupRepeatResolveNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: CleanupRepeatNodeProps<Parameters> & {
    isPolygon: boolean;
    collapsedCount: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: props.operation,
    variant: 'cleanup-repeat-resolve',
    bindings: [
      {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
      {name: 'ringOffsets', view: props.ringOffsets, type: 'u32', access: 'read'},
      {name: 'parameters', view: props.parameters, type: 'f32', access: 'read'},
      {name: 'ringState', view: props.ringState, type: 'u32', access: 'read_write'},
      {
        name: 'collapsedCount',
        view: props.collapsedCount,
        type: 'atomic<u32>',
        access: 'read_write'
      }
    ],
    invocationCount: props.ringOffsets.length - 1,
    condition: props.condition,
    extraResources: props.extraResources,
    declarations: `const VERTEX_COUNT: u32 = ${props.positions.length}u;
const IS_POLYGON: bool = ${props.isPolygon ? 'true' : 'false'};
${SNAP_SOURCE}
${WITHIN_TOLERANCE_SOURCE}`,
    body: /* wgsl */ `
  let start = ringOffsets[ringOffsetsOffset + index];
  let end = min(ringOffsets[ringOffsetsOffset + index + 1u], VERTEX_COUNT);
  if (end <= start || end - start < 2u || parameters[parametersOffset + 2u] == 0.0) {
    return;
  }
  let stateBase = ringStateOffset + ${CLEANUP_RING_STATE_WORDS}u * index;
  // First and final vertex plus the kept interior vertices.
  var keptCount = 2u + ringState[stateBase];
  let lastKept = ringState[stateBase + 1u];
  var replaced = 0u;
  if (lastKept != 0u) {
    let gridSize = parameters[parametersOffset];
    if (isWithinTolerance(readSnappedPoint(end - 1u, gridSize), readSnappedPoint(end - 2u, gridSize), 0.0)) {
      replaced = lastKept;
      keptCount--;
    }
  }
  if (IS_POLYGON && keptCount < 3u) {
    ringState[stateBase + 2u] = ${RING_COLLAPSED};
    atomicAdd(&collapsedCount[collapsedCountOffset], 1u);
  } else {
    ringState[stateBase + 2u] = replaced;
  }`
  });
}

/**
 * Builds the tolerance-0 per-vertex apply pass: clears the flag of the replaced vertex of a ring
 * and of every vertex of a collapsed ring, using the resolution words of
 * {@link createCleanupRepeatResolveNode}.
 *
 * @internal
 */
export function createCleanupRepeatApplyNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: Omit<CleanupRepeatNodeProps<Parameters>, 'positions' | 'parameters'> & {
    positionCount: number;
    keepFlags: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: props.operation,
    variant: 'cleanup-repeat-apply',
    bindings: [
      {name: 'ringOffsets', view: props.ringOffsets, type: 'u32', access: 'read'},
      {name: 'ringState', view: props.ringState, type: 'u32', access: 'read'},
      {name: 'keepFlags', view: props.keepFlags, type: 'u32', access: 'read_write'}
    ],
    invocationCount: props.positionCount,
    condition: props.condition,
    extraResources: props.extraResources,
    declarations: `const VERTEX_COUNT: u32 = ${props.positionCount}u;
const RING_COUNT: u32 = ${props.ringOffsets.length - 1}u;
${getUpperBoundSource('findRing', 'ringOffsets', 'RING_COUNT')}`,
    body: /* wgsl */ `
  let first = ringOffsets[ringOffsetsOffset];
  if (index < first || index >= min(ringOffsets[ringOffsetsOffset + RING_COUNT], VERTEX_COUNT)) {
    return;
  }
  let ring = findRing(index);
  let start = ringOffsets[ringOffsetsOffset + ring];
  let end = min(ringOffsets[ringOffsetsOffset + ring + 1u], VERTEX_COUNT);
  if (index < start || index >= end) {
    return;
  }
  let resolution = ringState[ringStateOffset + ${CLEANUP_RING_STATE_WORDS}u * ring + 2u];
  if (resolution == ${RING_COLLAPSED} || (resolution != 0u && index - start == resolution)) {
    keepFlags[keepFlagsOffset + index] = 0u;
  }`
  });
}
