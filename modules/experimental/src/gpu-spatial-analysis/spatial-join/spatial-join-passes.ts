// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  doGraphDataViewsOverlap,
  GPUBVH,
  GPUSort,
  GraphVectorView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {createWGSLKernelNode, type WGSLKernelBinding} from '../../utils/wgsl-kernel-nodes';
import type {GPUCompactOutput, GPUUint32Rows} from '../../utils/gpu-contributor-types';
import {getGraphViewChunks} from '../../utils/gpu-contributor-utils';

/** Shared WGSL constants and helpers for spatial-join kernels. @internal */
export const SPATIAL_JOIN_WGSL_HELPERS = /* wgsl */ `
const FLOAT32_MAXIMUM: f32 = 3.402823466e+38;
const NO_FEATURE: u32 = 0xffffffffu;
fn isFiniteValue(value: f32) -> bool { return value == value && abs(value) <= FLOAT32_MAXIMUM; }`;

/** Returns whether `value` is a positive power of two. @internal */
export function isPowerOfTwo(value: number): boolean {
  return Number.isSafeInteger(value) && value >= 1 && (value & (value - 1)) === 0;
}

/** Returns the smallest power of two that is at least `value` (and at least 1). @internal */
export function getNextPowerOfTwo(value: number): number {
  let result = 1;
  while (result < value) {
    result *= 2;
  }
  return result;
}

/** Throws unless `view` has the same chunk lengths as `reference`. @internal */
export function validateMatchingChunks(
  id: string,
  name: string,
  reference: GraphDataView | GraphVectorView,
  view: GraphDataView | GraphVectorView | undefined
): void {
  if (!view) {
    return;
  }
  const referenceLengths = getGraphViewChunks(reference).map(chunk => chunk.length);
  const viewLengths = getGraphViewChunks(view).map(chunk => chunk.length);
  if (
    referenceLengths.length !== viewLengths.length ||
    referenceLengths.some((length, index) => length !== viewLengths[index])
  ) {
    throw new Error(`${id} ${name} must match the chunk topology of points`);
  }
}

/** Throws when writable views overlap inputs or each other. @internal */
export function validateDisjointOutputs(
  id: string,
  inputs: readonly (GraphDataView | GraphVectorView | undefined)[],
  outputs: readonly (GraphDataView | GraphVectorView | undefined)[]
): void {
  const toChunks = (views: readonly (GraphDataView | GraphVectorView | undefined)[]) =>
    views.flatMap(view => (view ? getGraphViewChunks(view) : []));
  const inputChunks = toChunks(inputs);
  const outputChunks = toChunks(outputs);
  for (const [outputIndex, output] of outputChunks.entries()) {
    if (
      inputChunks.some(input => doGraphDataViewsOverlap(input, output)) ||
      outputChunks.some(
        (other, otherIndex) => otherIndex !== outputIndex && doGraphDataViewsOverlap(other, output)
      )
    ) {
      throw new Error(`${id} outputs must not overlap inputs or each other`);
    }
  }
}

/**
 * Caller-owned BVH output storage that outlives one encoding. @internal
 *
 * Used by {@link getFeatureBVHNodes} so a prepared right-hand side keeps its tree between
 * encodings instead of using graph transients, whose memory the graph may alias after their last use.
 */
export type SpatialJoinBVHStorage = {
  /** `2 * leafCapacity - 1` node minima. */
  nodeMinima: GraphDataView<'float32x2'>;
  /** `2 * leafCapacity - 1` node maxima. */
  nodeMaxima: GraphDataView<'float32x2'>;
  /** `2 * leafCapacity - 1` child pairs. */
  nodeChildren: GraphDataView<'uint32x2'>;
  /** `leafCapacity` leaf feature rows. */
  leafIds: GraphDataView<'uint32'>;
  /** One-row source row count. */
  count: GraphDataView<'uint32'>;
  /** One-row overflow flag. */
  overflow: GraphDataView<'uint32'>;
};

/**
 * Creates BVH storage over feature bounds and returns the BVH and its nodes. @internal
 *
 * Storage is graph transient unless `storage` is given.
 */
export function getFeatureBVHNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  minima: GraphDataView<'float32x2'>,
  maxima: GraphDataView<'float32x2'>,
  leafCapacity: number,
  sourceIds?: GraphDataView<'uint32'>,
  storage?: SpatialJoinBVHStorage
): {bvh: GPUBVH; nodes: readonly GPUCommandNode<Parameters>[]} {
  const nodeCount = 2 * leafCapacity - 1;
  const bvh = new GPUBVH({
    id: `${id}-bvh`,
    minima,
    maxima,
    sourceIds,
    leafCapacity,
    nodeMinima:
      storage?.nodeMinima ??
      createTransientView(graph, `${id}-bvh-node-minima`, 'float32x2', nodeCount),
    nodeMaxima:
      storage?.nodeMaxima ??
      createTransientView(graph, `${id}-bvh-node-maxima`, 'float32x2', nodeCount),
    nodeChildren:
      storage?.nodeChildren ??
      createTransientView(graph, `${id}-bvh-node-children`, 'uint32x2', nodeCount),
    leafIds:
      storage?.leafIds ?? createTransientView(graph, `${id}-bvh-leaf-ids`, 'uint32', leafCapacity),
    count: storage?.count ?? createTransientView(graph, `${id}-bvh-count`, 'uint32', 1),
    overflow: storage?.overflow ?? createTransientView(graph, `${id}-bvh-overflow`, 'uint32', 1)
  });
  return {bvh, nodes: bvh.getCommandNodes(graph)};
}

/** Feature count from which `spatialSort` defaults to on (see {@link getDefaultSpatialSort}). @internal */
export const SPATIAL_SORT_MINIMUM_FEATURES = 256;

/**
 * Default of `spatialSort`: on from {@link SPATIAL_SORT_MINIMUM_FEATURES} features. Measured with
 * `spatial-join-benchmark.spec.ts`: shuffled features cost 50 to 140 times more per encoding
 * unsorted (point-in-polygon, 14,400 features and 250,000 points: 236.7 ms against 4.6 ms;
 * nearest, 50,000 segments and 100,000 points: 333.6 ms against 2.4 ms), while coherent features
 * lose about 1 ms (3.2 ms against 4.3 ms), and below a few hundred features the BVH is too shallow
 * for the order to matter.
 *
 * @internal
 */
export function getDefaultSpatialSort(featureCount: number): boolean {
  return featureCount >= SPATIAL_SORT_MINIMUM_FEATURES;
}

/** Space-filling curve that orders features before the BVH build. @internal */
export type SpatialSortCurve = 'morton' | 'hilbert';

/**
 * Default `spatialSort` curve. Hilbert beat Morton by 10 to 15% of join GPU time in paired A/B runs
 * (`spatial-join-curve-ab.spec.ts`: point-in-polygon, nearest and buffer selection, shuffled and
 * coherent input) and was never slower beyond noise.
 *
 * @internal
 */
export const DEFAULT_SPATIAL_SORT_CURVE: SpatialSortCurve = 'hilbert';

/**
 * Creates the BVH over feature bounds, optionally after a Hilbert (default) or Morton (Z-order) sort of the features.
 *
 * With `spatialSort` and at least two features, features are reordered along a 16-bit-per-axis
 * Hilbert (or Morton) curve of their bound centers (empty or invalid features last) before the BVH build, and
 * the BVH `leafIds` map back to the original feature rows. Without it, or for fewer than two
 * features, this is {@link getFeatureBVHNodes}. Nothing is read back.
 *
 * @internal
 */
export function getSortedFeatureBVHNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  operation: string,
  minima: GraphDataView<'float32x2'>,
  maxima: GraphDataView<'float32x2'>,
  leafCapacity: number,
  spatialSort: boolean,
  storage?: SpatialJoinBVHStorage,
  spatialSortCurve: SpatialSortCurve = DEFAULT_SPATIAL_SORT_CURVE
): {bvh: GPUBVH; nodes: readonly GPUCommandNode<Parameters>[]} {
  const featureCount = minima.length;
  if (!spatialSort || featureCount < 2) {
    return getFeatureBVHNodes(graph, id, minima, maxima, leafCapacity, undefined, storage);
  }
  const sceneBounds = createTransientView(graph, `${id}-sort-scene-bounds`, 'float32', 4);
  const keys = createTransientView(graph, `${id}-sort-keys`, 'uint32', featureCount);
  const rows = createTransientView(graph, `${id}-sort-rows`, 'uint32', featureCount);
  const sortedKeys = createTransientView(graph, `${id}-sort-sorted-keys`, 'uint32', featureCount);
  const sortedRows = createTransientView(graph, `${id}-sort-sorted-rows`, 'uint32', featureCount);
  const sortedMinima = createTransientView(
    graph,
    `${id}-sort-sorted-minima`,
    'float32x2',
    featureCount
  );
  const sortedMaxima = createTransientView(
    graph,
    `${id}-sort-sorted-maxima`,
    'float32x2',
    featureCount
  );
  const nodes: GPUCommandNode<Parameters>[] = [];
  const featureBindings: WGSLKernelBinding[] = [
    {name: 'featureMinima', view: minima, type: 'f32', access: 'read'},
    {name: 'featureMaxima', view: maxima, type: 'f32', access: 'read'}
  ];
  const validFeatureDeclarations = `${SPATIAL_JOIN_WGSL_HELPERS}
const FEATURE_COUNT: u32 = ${featureCount}u;
fn readFeatureCenter(row: u32) -> vec4f {
  // Returns (center.x, center.y, valid, 0). Halving before the sum avoids overflow near FLT_MAX.
  let minimum = vec2f(featureMinima[featureMinimaOffset + row * 2u], featureMinima[featureMinimaOffset + row * 2u + 1u]);
  let maximum = vec2f(featureMaxima[featureMaximaOffset + row * 2u], featureMaxima[featureMaximaOffset + row * 2u + 1u]);
  let valid = isFiniteValue(minimum.x) && isFiniteValue(minimum.y) &&
    isFiniteValue(maximum.x) && isFiniteValue(maximum.y) &&
    minimum.x <= maximum.x && minimum.y <= maximum.y;
  let center = minimum * 0.5 + maximum * 0.5;
  return vec4f(center, select(0.0, 1.0, valid), 0.0);
}`;
  // One workgroup strides over all features and reduces the centers of valid features.
  nodes.push(
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-sort-scene-bounds`,
      operation,
      variant: 'sort-scene-bounds',
      bindings: [
        ...featureBindings,
        {name: 'sceneBounds', view: sceneBounds, type: 'f32', access: 'read_write'}
      ],
      invocationCount: 256,
      guardIndex: false,
      declarations: `${validFeatureDeclarations}
var<workgroup> sharedMinima: array<vec2f, 256>;
var<workgroup> sharedMaxima: array<vec2f, 256>;`,
      body: `// Exactly one workgroup is dispatched, so index == localInvocationIndex.
  var localMinimum = vec2f(FLOAT32_MAXIMUM);
  var localMaximum = vec2f(-FLOAT32_MAXIMUM);
  for (var row = localInvocationIndex; row < FEATURE_COUNT; row += 256u) {
    let center = readFeatureCenter(row);
    if (center.z > 0.5) {
      localMinimum = min(localMinimum, center.xy);
      localMaximum = max(localMaximum, center.xy);
    }
  }
  sharedMinima[localInvocationIndex] = localMinimum;
  sharedMaxima[localInvocationIndex] = localMaximum;
  workgroupBarrier();
  for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
    if (localInvocationIndex < stride) {
      sharedMinima[localInvocationIndex] = min(sharedMinima[localInvocationIndex], sharedMinima[localInvocationIndex + stride]);
      sharedMaxima[localInvocationIndex] = max(sharedMaxima[localInvocationIndex], sharedMaxima[localInvocationIndex + stride]);
    }
    workgroupBarrier();
  }
  if (localInvocationIndex == 0u) {
    let minimum = sharedMinima[0];
    let maximum = sharedMaxima[0];
    let hasValid = minimum.x <= maximum.x && minimum.y <= maximum.y;
    sceneBounds[sceneBoundsOffset] = select(0.0, minimum.x, hasValid);
    sceneBounds[sceneBoundsOffset + 1u] = select(0.0, minimum.y, hasValid);
    sceneBounds[sceneBoundsOffset + 2u] = select(0.0, maximum.x, hasValid);
    sceneBounds[sceneBoundsOffset + 3u] = select(0.0, maximum.y, hasValid);
  }`
    })
  );
  nodes.push(
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-sort-keys`,
      operation,
      variant: spatialSortCurve === 'hilbert' ? 'sort-keys-hilbert' : 'sort-keys',
      bindings: [
        ...featureBindings,
        {name: 'sceneBounds', view: sceneBounds, type: 'f32', access: 'read'},
        {name: 'sortKeys', view: keys, type: 'u32', access: 'read_write'},
        {name: 'sortRows', view: rows, type: 'u32', access: 'read_write'}
      ],
      invocationCount: featureCount,
      declarations: `${validFeatureDeclarations}
fn spreadBits(value: u32) -> u32 {
  var x = value & 0xffffu;
  x = (x | (x << 8u)) & 0x00ff00ffu;
  x = (x | (x << 4u)) & 0x0f0f0f0fu;
  x = (x | (x << 2u)) & 0x33333333u;
  x = (x | (x << 1u)) & 0x55555555u;
  return x;
}
fn quantizeAxis(value: f32, minimum: f32, maximum: f32) -> u32 {
  let extent = maximum - minimum;
  if (!(extent > 0.0) || !isFiniteValue(extent)) { return 0u; }
  let normalized = clamp((value - minimum) / extent, 0.0, 1.0);
  return min(u32(normalized * 65535.0 + 0.5), 65535u);
}
// Classic xy2d on the 65536 x 65536 grid (same curve as GPUHilbertKeys at order 16).
fn getHilbertIndex(cellX: u32, cellY: u32) -> u32 {
  var x = cellX;
  var y = cellY;
  var key = 0u;
  for (var s = 32768u; s > 0u; s = s >> 1u) {
    let rx = select(0u, 1u, (x & s) != 0u);
    let ry = select(0u, 1u, (y & s) != 0u);
    key += s * s * ((3u * rx) ^ ry);
    if (ry == 0u) {
      if (rx == 1u) {
        x = 65535u - x;
        y = 65535u - y;
      }
      let swap = x;
      x = y;
      y = swap;
    }
  }
  return key;
}`,
      body: `sortRows[sortRowsOffset + index] = index;
  let center = readFeatureCenter(index);
  var key = 0xffffffffu;
  if (center.z > 0.5) {
    let quantizedX = quantizeAxis(center.x, sceneBounds[sceneBoundsOffset], sceneBounds[sceneBoundsOffset + 2u]);
    let quantizedY = quantizeAxis(center.y, sceneBounds[sceneBoundsOffset + 1u], sceneBounds[sceneBoundsOffset + 3u]);
    key = ${
      spatialSortCurve === 'hilbert'
        ? 'getHilbertIndex(quantizedX, quantizedY)'
        : 'spreadBits(quantizedX) | (spreadBits(quantizedY) << 1u)'
    };
  }
  sortKeys[sortKeysOffset + index] = key;`
    })
  );
  nodes.push(
    ...new GPUSort({
      id: `${id}-sort-order`,
      keys,
      values: rows,
      outputKeys: sortedKeys,
      outputValues: sortedRows,
      algorithm: 'radix'
    }).getCommandNodes(graph)
  );
  nodes.push(
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-sort-gather`,
      operation,
      variant: 'sort-gather',
      bindings: [
        ...featureBindings,
        {name: 'sortedRows', view: sortedRows, type: 'u32', access: 'read'},
        {name: 'sortedMinima', view: sortedMinima, type: 'f32', access: 'read_write'},
        {name: 'sortedMaxima', view: sortedMaxima, type: 'f32', access: 'read_write'}
      ],
      invocationCount: featureCount,
      declarations: `const FEATURE_COUNT: u32 = ${featureCount}u;`,
      body: `let row = min(sortedRows[sortedRowsOffset + index], FEATURE_COUNT - 1u);
  sortedMinima[sortedMinimaOffset + index * 2u] = featureMinima[featureMinimaOffset + row * 2u];
  sortedMinima[sortedMinimaOffset + index * 2u + 1u] = featureMinima[featureMinimaOffset + row * 2u + 1u];
  sortedMaxima[sortedMaximaOffset + index * 2u] = featureMaxima[featureMaximaOffset + row * 2u];
  sortedMaxima[sortedMaximaOffset + index * 2u + 1u] = featureMaxima[featureMaximaOffset + row * 2u + 1u];`
    })
  );
  const {bvh, nodes: bvhNodes} = getFeatureBVHNodes(
    graph,
    id,
    sortedMinima,
    sortedMaxima,
    leafCapacity,
    sortedRows,
    storage
  );
  nodes.push(...bvhNodes);
  return {bvh, nodes};
}

/** Feature geometry used to compute per-feature bounds. @internal */
export type SpatialJoinBoundsSource =
  | {
      kind: 'polygons';
      polygonPositions: GraphDataView<'float32x2'>;
      featureOffsets: GraphDataView<'uint32'>;
      polygonOffsets: GraphDataView<'uint32'>;
      ringOffsets: GraphDataView<'uint32'>;
    }
  | {kind: 'points'; positions: GraphDataView<'float32x2'>}
  | {kind: 'segments'; starts: GraphDataView<'float32x2'>; ends: GraphDataView<'float32x2'>};

/** Writes per-feature bounds; empty or invalid features get inverted (empty) bounds. @internal */
export function createSpatialJoinBoundsNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    featureCount: number;
    source: SpatialJoinBoundsSource;
    minima: GraphDataView<'float32x2'>;
    maxima: GraphDataView<'float32x2'>;
  }
): GPUCommandNode<Parameters> {
  const {source} = props;
  const bindings: WGSLKernelBinding[] = [];
  let declarations = SPATIAL_JOIN_WGSL_HELPERS;
  let body: string;
  if (source.kind === 'polygons') {
    bindings.push(
      {name: 'polygonPositions', view: source.polygonPositions, type: 'f32', access: 'read'},
      {name: 'featureOffsets', view: source.featureOffsets, type: 'u32', access: 'read'},
      {name: 'polygonOffsets', view: source.polygonOffsets, type: 'u32', access: 'read'},
      {name: 'ringOffsets', view: source.ringOffsets, type: 'u32', access: 'read'}
    );
    declarations += `
const POLYGON_COUNT: u32 = ${source.polygonOffsets.length - 1}u;
const RING_COUNT: u32 = ${source.ringOffsets.length - 1}u;
const VERTEX_COUNT: u32 = ${source.polygonPositions.length}u;`;
    body = `var minimum = vec2f(FLOAT32_MAXIMUM);
  var maximum = vec2f(-FLOAT32_MAXIMUM);
  let polygonStart = featureOffsets[featureOffsetsOffset + index];
  let polygonEnd = featureOffsets[featureOffsetsOffset + index + 1u];
  if (polygonStart <= polygonEnd && polygonEnd <= POLYGON_COUNT) {
    let ringStart = polygonOffsets[polygonOffsetsOffset + polygonStart];
    let ringEnd = polygonOffsets[polygonOffsetsOffset + polygonEnd];
    if (ringStart <= ringEnd && ringEnd <= RING_COUNT) {
      let vertexStart = ringOffsets[ringOffsetsOffset + ringStart];
      let vertexEnd = min(ringOffsets[ringOffsetsOffset + ringEnd], VERTEX_COUNT);
      for (var vertex = vertexStart; vertex < vertexEnd; vertex++) {
        let position = vec2f(
          polygonPositions[polygonPositionsOffset + vertex * 2u],
          polygonPositions[polygonPositionsOffset + vertex * 2u + 1u]
        );
        if (isFiniteValue(position.x) && isFiniteValue(position.y)) {
          minimum = min(minimum, position);
          maximum = max(maximum, position);
        }
      }
    }
  }`;
  } else if (source.kind === 'points') {
    bindings.push({name: 'positions', view: source.positions, type: 'f32', access: 'read'});
    body = `var minimum = vec2f(FLOAT32_MAXIMUM);
  var maximum = vec2f(-FLOAT32_MAXIMUM);
  let position = vec2f(positions[positionsOffset + index * 2u], positions[positionsOffset + index * 2u + 1u]);
  if (isFiniteValue(position.x) && isFiniteValue(position.y)) {
    minimum = position;
    maximum = position;
  }`;
  } else {
    bindings.push(
      {name: 'starts', view: source.starts, type: 'f32', access: 'read'},
      {name: 'ends', view: source.ends, type: 'f32', access: 'read'}
    );
    body = `var minimum = vec2f(FLOAT32_MAXIMUM);
  var maximum = vec2f(-FLOAT32_MAXIMUM);
  let start = vec2f(starts[startsOffset + index * 2u], starts[startsOffset + index * 2u + 1u]);
  let end = vec2f(ends[endsOffset + index * 2u], ends[endsOffset + index * 2u + 1u]);
  if (isFiniteValue(start.x) && isFiniteValue(start.y) && isFiniteValue(end.x) && isFiniteValue(end.y)) {
    minimum = min(start, end);
    maximum = max(start, end);
  }`;
  }
  bindings.push(
    {name: 'featureMinima', view: props.minima, type: 'f32', access: 'read_write'},
    {name: 'featureMaxima', view: props.maxima, type: 'f32', access: 'read_write'}
  );
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: props.operation,
    variant: `bounds-${source.kind}`,
    bindings,
    invocationCount: props.featureCount,
    declarations,
    body: `${body}
  featureMinima[featureMinimaOffset + index * 2u] = minimum.x;
  featureMinima[featureMinimaOffset + index * 2u + 1u] = minimum.y;
  featureMaxima[featureMaximaOffset + index * 2u] = maximum.x;
  featureMaxima[featureMaximaOffset + index * 2u + 1u] = maximum.y;`
  });
}

/** Clears join state, per-point scratch, optional counts, and the padded pair layout. @internal */
export function createSpatialJoinClearNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    state: GraphDataView<'uint32'>;
    assignment: GraphDataView<'uint32'>;
    pointCount: number;
    featureCounts?: GraphDataView<'uint32'>;
    bestDistanceBits?: GraphDataView<'uint32'>;
    pairs?: {
      points: GraphDataView<'float32x2'>;
      geometryOffsets: GraphDataView<'uint32'>;
      polygonCount: number;
    };
  }
): GPUCommandNode<Parameters> {
  const {pairs, featureCounts, bestDistanceBits} = props;
  const bindings: WGSLKernelBinding[] = [
    {name: 'state', view: props.state, type: 'atomic<u32>', access: 'read_write'},
    {name: 'assignment', view: props.assignment, type: 'u32', access: 'read_write'}
  ];
  if (pairs) {
    bindings.push(
      {name: 'pairPoints', view: pairs.points, type: 'f32', access: 'read_write'},
      {name: 'pairGeometryOffsets', view: pairs.geometryOffsets, type: 'u32', access: 'read_write'}
    );
  }
  if (bestDistanceBits) {
    bindings.push({
      name: 'bestDistanceBits',
      view: bestDistanceBits,
      type: 'u32',
      access: 'read_write'
    });
  }
  if (featureCounts) {
    bindings.push({name: 'featureCounts', view: featureCounts, type: 'u32', access: 'read_write'});
  }
  const lengths = [
    4,
    props.pointCount,
    pairs ? pairs.geometryOffsets.length : 0,
    featureCounts?.length ?? 0
  ];
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: props.operation,
    variant: 'clear',
    bindings,
    invocationCount: Math.max(...lengths),
    declarations: SPATIAL_JOIN_WGSL_HELPERS,
    body: `if (index < 4u) { atomicStore(&state[stateOffset + index], 0u); }
  if (index < ${props.pointCount}u) {
    assignment[assignmentOffset + index] = NO_FEATURE;
    ${bestDistanceBits ? 'bestDistanceBits[bestDistanceBitsOffset + index] = 0xffffffffu;' : ''}
  }
  ${
    pairs
      ? `if (index < ${pairs.points.length}u) {
    // A runtime expression keeps the quiet-NaN bit pattern out of const evaluation.
    let quietNaN = bitcast<f32>(0x7fc00000u | (index & 0u));
    pairPoints[pairPointsOffset + index * 2u] = quietNaN;
    pairPoints[pairPointsOffset + index * 2u + 1u] = quietNaN;
  }
  if (index < ${pairs.geometryOffsets.length}u) {
    pairGeometryOffsets[pairGeometryOffsetsOffset + index] =
      select(0u, ${pairs.polygonCount}u, index == ${pairs.geometryOffsets.length - 1}u);
  }`
      : ''
  }
  ${featureCounts ? `if (index < ${featureCounts.length}u) { featureCounts[featureCountsOffset + index] = 0u; }` : ''}`
  });
}

/** Returns the chunk node ID: no suffix for packed views, `-${index}` for vectors. @internal */
export function getChunkNodeId(
  id: string,
  view: GraphDataView | GraphVectorView,
  chunkIndex: number
): string {
  return view instanceof GraphVectorView ? `${id}-${chunkIndex}` : id;
}

/**
 * Builds one stackless BVH probe per nonempty point chunk that appends `[pointRow, featureRow]`
 * candidate pairs and writes each candidate's point into `candidatePoints`.
 *
 * @internal
 */
export function getSpatialJoinProbeNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    points: GraphDataView<'float32x2'> | GraphVectorView<'float32x2'>;
    bvh: Pick<GPUBVH, 'nodeMinima' | 'nodeMaxima' | 'leafIds' | 'internalNodeCount'>;
    featureCount: number;
    candidateCapacity: number;
    state: GraphDataView<'uint32'>;
    candidatePairs: GraphDataView<'uint32x2'>;
    /** Destination of the candidate point, at row `slot * pointRowScale + pointRowOffset`. */
    candidatePoints: GraphDataView<'float32x2'>;
    pointRowScale: number;
    pointRowOffset: number;
    radius?: GraphDataView<'float32'>;
  }
): GPUCommandNode<Parameters>[] {
  const nodes: GPUCommandNode<Parameters>[] = [];
  const {bvh} = props;
  let chunkFirstRow = 0;
  for (const [chunkIndex, chunk] of getGraphViewChunks(props.points).entries()) {
    if (chunk.length > 0) {
      const bindings: WGSLKernelBinding[] = [
        {name: 'points', view: chunk, type: 'f32', access: 'read'},
        {name: 'nodeMinima', view: bvh.nodeMinima, type: 'f32', access: 'read'},
        {name: 'nodeMaxima', view: bvh.nodeMaxima, type: 'f32', access: 'read'},
        {name: 'leafIds', view: bvh.leafIds, type: 'u32', access: 'read'},
        {name: 'state', view: props.state, type: 'atomic<u32>', access: 'read_write'},
        {name: 'candidatePairs', view: props.candidatePairs, type: 'u32', access: 'read_write'},
        {name: 'candidatePoints', view: props.candidatePoints, type: 'f32', access: 'read_write'}
      ];
      if (props.radius) {
        bindings.push({name: 'radius', view: props.radius, type: 'f32', access: 'read'});
      }
      const query = props.radius
        ? `let searchRadius = radius[radiusOffset];
  if (!(searchRadius >= 0.0) || searchRadius > FLOAT32_MAXIMUM) { return; }
  let queryMinimum = point - vec2f(searchRadius);
  let queryMaximum = point + vec2f(searchRadius);`
        : `let queryMinimum = point;
  let queryMaximum = point;`;
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: getChunkNodeId(props.id, props.points, chunkIndex),
          operation: props.operation,
          variant: 'probe',
          bindings,
          invocationCount: chunk.length,
          declarations: `${SPATIAL_JOIN_WGSL_HELPERS}
const CHUNK_FIRST_ROW: u32 = ${chunkFirstRow}u;
const INTERNAL_NODE_COUNT: u32 = ${bvh.internalNodeCount}u;
const FEATURE_COUNT: u32 = ${props.featureCount}u;
const CANDIDATE_CAPACITY: u32 = ${props.candidateCapacity}u;
fn nodeOverlaps(node: u32, queryMinimum: vec2f, queryMaximum: vec2f) -> bool {
  let component = node * 2u;
  let minimum = vec2f(nodeMinima[nodeMinimaOffset + component], nodeMinima[nodeMinimaOffset + component + 1u]);
  let maximum = vec2f(nodeMaxima[nodeMaximaOffset + component], nodeMaxima[nodeMaximaOffset + component + 1u]);
  return all(minimum <= queryMaximum) && all(queryMinimum <= maximum);
}`,
          body: `let row = CHUNK_FIRST_ROW + index;
  let point = vec2f(points[pointsOffset + index * 2u], points[pointsOffset + index * 2u + 1u]);
  if (!isFiniteValue(point.x) || !isFiniteValue(point.y)) { return; }
  ${query}
  var node = 0u;
  loop {
    if (nodeOverlaps(node, queryMinimum, queryMaximum)) {
      if (node < INTERNAL_NODE_COUNT) {
        node = node * 2u + 1u;
        continue;
      }
      let featureRow = leafIds[leafIdsOffset + node - INTERNAL_NODE_COUNT];
      if (featureRow < FEATURE_COUNT) {
        let slot = atomicAdd(&state[stateOffset], 1u);
        if (slot < CANDIDATE_CAPACITY) {
          candidatePairs[candidatePairsOffset + slot * 2u] = row;
          candidatePairs[candidatePairsOffset + slot * 2u + 1u] = featureRow;
          let pointRow = slot * ${props.pointRowScale}u + ${props.pointRowOffset}u;
          candidatePoints[candidatePointsOffset + pointRow * 2u] = point.x;
          candidatePoints[candidatePointsOffset + pointRow * 2u + 1u] = point.y;
        }
      }
    }
    // Climb while the node is a right child, then step to the right sibling.
    loop {
      if (node == 0u || (node & 1u) == 1u) { break; }
      node = (node - 1u) / 2u;
    }
    if (node == 0u) { break; }
    node = node + 1u;
  }`
        })
      );
    }
    chunkFirstRow += chunk.length;
  }
  return nodes;
}

/**
 * Writes per-point feature IDs (and optional distances) and accumulates per-feature counts.
 *
 * @internal
 */
export function getSpatialJoinAssignNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    assignment: GraphDataView<'uint32'>;
    pointFeatureIds: GPUUint32Rows;
    featureIds?: GraphDataView<'uint32'>;
    featureCounts?: GraphDataView<'uint32'>;
    bestDistanceBits?: GraphDataView<'uint32'>;
    distances?: GraphDataView<'float32'> | GraphVectorView<'float32'>;
  }
): GPUCommandNode<Parameters>[] {
  const nodes: GPUCommandNode<Parameters>[] = [];
  const distanceChunks = props.distances ? getGraphViewChunks(props.distances) : [];
  let chunkFirstRow = 0;
  for (const [chunkIndex, chunk] of getGraphViewChunks(props.pointFeatureIds).entries()) {
    if (chunk.length > 0) {
      const bindings: WGSLKernelBinding[] = [
        {name: 'assignment', view: props.assignment, type: 'u32', access: 'read'},
        {name: 'pointFeatureIds', view: chunk, type: 'u32', access: 'read_write'}
      ];
      if (props.featureIds) {
        bindings.push({name: 'featureIds', view: props.featureIds, type: 'u32', access: 'read'});
      }
      if (props.featureCounts) {
        bindings.push({
          name: 'featureCounts',
          view: props.featureCounts,
          type: 'atomic<u32>',
          access: 'read_write'
        });
      }
      const distanceChunk = distanceChunks[chunkIndex];
      if (props.bestDistanceBits && distanceChunk) {
        bindings.push(
          {name: 'bestDistanceBits', view: props.bestDistanceBits, type: 'u32', access: 'read'},
          {name: 'distances', view: distanceChunk, type: 'f32', access: 'read_write'}
        );
      }
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: getChunkNodeId(props.id, props.pointFeatureIds, chunkIndex),
          operation: props.operation,
          variant: 'assign',
          bindings,
          invocationCount: chunk.length,
          declarations: `${SPATIAL_JOIN_WGSL_HELPERS}
const CHUNK_FIRST_ROW: u32 = ${chunkFirstRow}u;`,
          body: `let row = CHUNK_FIRST_ROW + index;
  let featureRow = assignment[assignmentOffset + row];
  let matched = featureRow != NO_FEATURE;
  var featureId = featureRow;
  ${props.featureIds ? 'if (matched) { featureId = featureIds[featureIdsOffset + featureRow]; }' : ''}
  pointFeatureIds[pointFeatureIdsOffset + index] = featureId;
  ${props.featureCounts ? 'if (matched) { atomicAdd(&featureCounts[featureCountsOffset + featureRow], 1u); }' : ''}
  ${
    props.bestDistanceBits && distanceChunk
      ? 'distances[distancesOffset + index] = select(-1.0, bitcast<f32>(bestDistanceBits[bestDistanceBitsOffset + row]), matched);'
      : ''
  }`
        })
      );
    }
    chunkFirstRow += chunk.length;
  }
  return nodes;
}

/** Appends stable IDs of matched points into `matches.ids`, counting the total in state[2]. @internal */
export function getSpatialJoinCollectNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    points: GraphDataView | GraphVectorView;
    assignment: GraphDataView<'uint32'>;
    state: GraphDataView<'uint32'>;
    sourceIds?: GPUUint32Rows;
    matches: GPUCompactOutput;
  }
): GPUCommandNode<Parameters>[] {
  const nodes: GPUCommandNode<Parameters>[] = [];
  const sourceChunks = props.sourceIds ? getGraphViewChunks(props.sourceIds) : [];
  const capacity = props.matches.ids.length;
  let chunkFirstRow = 0;
  for (const [chunkIndex, chunk] of getGraphViewChunks(props.points).entries()) {
    if (chunk.length > 0) {
      const sourceChunk = sourceChunks[chunkIndex];
      const bindings: WGSLKernelBinding[] = [
        {name: 'assignment', view: props.assignment, type: 'u32', access: 'read'},
        {name: 'state', view: props.state, type: 'atomic<u32>', access: 'read_write'}
      ];
      if (sourceChunk) {
        bindings.push({name: 'sourceIds', view: sourceChunk, type: 'u32', access: 'read'});
      }
      if (capacity > 0) {
        bindings.push({
          name: 'matchIds',
          view: props.matches.ids,
          type: 'u32',
          access: 'read_write'
        });
      }
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: getChunkNodeId(props.id, props.points, chunkIndex),
          operation: props.operation,
          variant: 'collect-matches',
          bindings,
          invocationCount: chunk.length,
          declarations: `${SPATIAL_JOIN_WGSL_HELPERS}
const CHUNK_FIRST_ROW: u32 = ${chunkFirstRow}u;
const MATCH_CAPACITY: u32 = ${capacity}u;`,
          body: `let row = CHUNK_FIRST_ROW + index;
  if (assignment[assignmentOffset + row] == NO_FEATURE) { return; }
  let slot = atomicAdd(&state[stateOffset + 2u], 1u);
  ${
    capacity > 0
      ? `if (slot < MATCH_CAPACITY) {
    matchIds[matchIdsOffset + slot] = ${sourceChunk ? 'sourceIds[sourceIdsOffset + index]' : 'row'};
  }`
      : ''
  }`
        })
      );
    }
    chunkFirstRow += chunk.length;
  }
  return nodes;
}

/** Publishes overflow, candidate and uncertain counts, and the clamped match count. @internal */
export function createSpatialJoinFinalizeNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    state: GraphDataView<'uint32'>;
    bvhOverflow: GraphDataView<'uint32'>;
    candidateCapacity: number;
    overflow: GraphDataView<'uint32'>;
    candidateCount?: GraphDataView<'uint32'>;
    uncertainCount?: GraphDataView<'uint32'>;
    matches?: GPUCompactOutput;
  }
): GPUCommandNode<Parameters> {
  const {matches} = props;
  const bindings: WGSLKernelBinding[] = [
    {name: 'state', view: props.state, type: 'u32', access: 'read'},
    {name: 'bvhOverflow', view: props.bvhOverflow, type: 'u32', access: 'read'},
    {name: 'overflow', view: props.overflow, type: 'u32', access: 'read_write'}
  ];
  if (props.candidateCount) {
    bindings.push({
      name: 'candidateCount',
      view: props.candidateCount,
      type: 'u32',
      access: 'read_write'
    });
  }
  if (props.uncertainCount) {
    bindings.push({
      name: 'uncertainCount',
      view: props.uncertainCount,
      type: 'u32',
      access: 'read_write'
    });
  }
  if (matches) {
    bindings.push(
      {name: 'matchCount', view: matches.count, type: 'u32', access: 'read_write'},
      {name: 'matchOverflow', view: matches.overflow, type: 'u32', access: 'read_write'}
    );
    if (matches.totalCount) {
      bindings.push({
        name: 'matchTotalCount',
        view: matches.totalCount,
        type: 'u32',
        access: 'read_write'
      });
    }
  }
  const matchCapacity = matches?.ids.length ?? 0;
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: props.operation,
    variant: 'finalize',
    bindings,
    invocationCount: 1,
    declarations: `const CANDIDATE_CAPACITY: u32 = ${props.candidateCapacity}u;
const MATCH_CAPACITY: u32 = ${matchCapacity}u;`,
    body: `let candidateTotal = state[stateOffset];
  let matchTotal = state[stateOffset + 2u];
  let overflowed = bvhOverflow[bvhOverflowOffset] != 0u || candidateTotal > CANDIDATE_CAPACITY${
    matches ? ' || matchTotal > MATCH_CAPACITY' : ''
  };
  let overflowValue = select(0u, 1u, overflowed);
  overflow[overflowOffset] = overflowValue;
  ${props.candidateCount ? 'candidateCount[candidateCountOffset] = candidateTotal;' : ''}
  ${props.uncertainCount ? 'uncertainCount[uncertainCountOffset] = state[stateOffset + 1u];' : ''}
  ${
    matches
      ? `matchCount[matchCountOffset] = min(matchTotal, MATCH_CAPACITY);
  matchOverflow[matchOverflowOffset] = overflowValue;
  ${matches.totalCount ? 'matchTotalCount[matchTotalCountOffset] = matchTotal;' : ''}`
      : ''
  }`
  });
}
