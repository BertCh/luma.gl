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
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../utils/gpu-contributor-utils';
import {createWGSLKernelNode, type WGSLKernelBinding} from '../../utils/wgsl-kernel-nodes';
import {getSegmentBVHNodes} from '../segment-intersection/segment-bvh';

/** Reference count above which the global mode searches a BVH instead of scanning every row. */
const BVH_MINIMUM_REFERENCES = 64;

function getNextPowerOfTwo(value: number): number {
  let result = 1;
  while (result < value) {
    result *= 2;
  }
  return result;
}

const OPERATION = 'GPUVertexSnap';

/** Number of float32 elements in a {@link GPUVertexSnap} parameter buffer. */
export const GPU_VERTEX_SNAP_PARAMETER_LENGTH = 4;

/** Value of `output.referenceRows` for a vertex that did not snap. */
export const GPU_VERTEX_SNAP_NO_REFERENCE = 0xffffffff;

/** CPU description of the per-frame parameters of {@link GPUVertexSnap}. */
export type GPUVertexSnapParameters = {
  /** Snap radius in position units. A vertex snaps when a reference vertex is strictly closer. */
  tolerance: number;
};

/**
 * Packs {@link GPUVertexSnapParameters} into the 4-element float32 layout `[tolerance, 0, 0, 0]`.
 *
 * @param parameters Parameters to encode.
 * @param target Optional destination of at least 4 elements.
 * @throws If the tolerance is negative or not finite, or `target` is too short.
 */
export function getGPUVertexSnapParameterValues(
  parameters: GPUVertexSnapParameters,
  target: Float32Array = new Float32Array(GPU_VERTEX_SNAP_PARAMETER_LENGTH)
): Float32Array {
  if (target.length < GPU_VERTEX_SNAP_PARAMETER_LENGTH) {
    throw new Error(`Vertex snap target must hold ${GPU_VERTEX_SNAP_PARAMETER_LENGTH} elements`);
  }
  if (!Number.isFinite(parameters.tolerance) || parameters.tolerance < 0) {
    throw new Error('Vertex snap tolerance must be a non-negative finite number');
  }
  target.set([parameters.tolerance, 0, 0, 0]);
  return target;
}

/** Caller-owned outputs of {@link GPUVertexSnap}. */
export type GPUVertexSnapOutput = {
  /** Snapped positions, one row per input row, same order and layout as the input. */
  positions: GraphDataView<'float32x2'>;
  /**
   * Optional per input row: the index into `referencePositions` the vertex snapped to, or
   * {@link GPU_VERTEX_SNAP_NO_REFERENCE}.
   */
  referenceRows?: GraphDataView<'uint32'>;
};

/**
 * Properties for {@link GPUVertexSnap}.
 *
 * Per-frame (no recompile): the contents of `parameters` (tolerance) and every input buffer.
 * Compile-time: view lengths and which optional views are present.
 */
export type GPUVertexSnapProps = {
  /** Prefix for generated node IDs. Defaults to `'vertex-snap'`. */
  id?: string;
  /** Packed planar positions of the geometry to snap (any layout: rows are independent). */
  positions: GraphDataView<'float32x2'>;
  /** Packed reference vertices. */
  referencePositions: GraphDataView<'float32x2'>;
  /**
   * Optional pairwise mode: `featureCount + 1` monotonic offsets over `positions`. With
   * `referenceOffsets`, feature `f` snaps only to reference rows
   * `[referenceOffsets[f], referenceOffsets[f + 1])` (`shapely.snap(a[f], b[f])`). Without both,
   * every vertex snaps to the whole reference set (`shapely.snap(a, b)` with one `b`).
   */
  featureOffsets?: GraphDataView<'uint32'>;
  /** Optional pairwise mode: `featureCount + 1` monotonic offsets over `referencePositions`. */
  referenceOffsets?: GraphDataView<'uint32'>;
  /** Per-frame packed float32 view written with {@link getGPUVertexSnapParameterValues}. */
  parameters: GraphDataView<'float32'>;
  /** Output views. */
  output: GPUVertexSnapOutput;
};

/**
 * Vertex-to-vertex snapping (`shapely.snap`, `shapely.ops.snap`, geo `snap`/Sedona `ST_Snap`
 * subset): every vertex with a reference vertex strictly closer than `tolerance` moves onto the
 * nearest such reference vertex, the lowest reference row winning ties; other vertices are kept
 * bit-for-bit. Output has the same layout as the input, so offsets stay valid.
 *
 * **Partial by design.** GEOS also snaps reference vertices onto segments of the input and inserts
 * them (changing vertex counts) and snaps the input's segments to references; that is out of
 * scope here. Results equal `shapely.snap` whenever no reference vertex lies within the tolerance
 * of a segment without also being within the tolerance of one of its vertices (checked against
 * GEOS in the spec). Snapping can create repeated vertices; clean them afterward if needed.
 *
 * Cost: with one global reference set of more than 64 vertices, a BVH over the reference vertices
 * is built and every input vertex runs a branch-and-bound nearest search pruned by
 * `min(tolerance, best)` squared, `O(vertices * log references)` for typical tolerances. The
 * result is identical to a linear scan (box distances are exact lower bounds in f32, and ties
 * keep the lowest row). Pairwise mode and small reference sets scan the vertex's reference range
 * linearly, `O(vertices * references per feature)`. No atomics, deterministic. The
 * tolerance is read per frame. Precision: f32 squared distances (relative error about `1e-7`
 * around the tolerance boundary).
 */
export class GPUVertexSnap implements GPUCommandNodeProducer {
  /** Prefix for every node ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUVertexSnapProps;

  constructor(props: GPUVertexSnapProps) {
    this.id = props.id ?? 'vertex-snap';
    this.props = props;
    const {id} = this;
    for (const [name, view] of Object.entries({
      positions: props.positions,
      referencePositions: props.referencePositions,
      featureOffsets: props.featureOffsets,
      referenceOffsets: props.referenceOffsets,
      parameters: props.parameters,
      outputPositions: props.output.positions,
      outputReferenceRows: props.output.referenceRows
    })) {
      if ((view as unknown) instanceof GraphVectorView) {
        throw new Error(`${id} ${name} must be a single packed view, not a chunked vector`);
      }
    }
    validatePackedView(props.positions, ['float32x2'], `${id} positions`);
    validatePackedView(props.referencePositions, ['float32x2'], `${id} referencePositions`);
    if (props.positions.length < 1) {
      throw new Error(`${id} needs at least one position`);
    }
    if (Boolean(props.featureOffsets) !== Boolean(props.referenceOffsets)) {
      throw new Error(`${id} featureOffsets and referenceOffsets must be given together`);
    }
    if (props.featureOffsets && props.referenceOffsets) {
      validatePackedUint32View(props.featureOffsets, `${id} featureOffsets`);
      validatePackedUint32View(props.referenceOffsets, `${id} referenceOffsets`);
      if (props.featureOffsets.length < 2) {
        throw new Error(`${id} featureOffsets must contain at least two rows`);
      }
      if (props.referenceOffsets.length !== props.featureOffsets.length) {
        throw new Error(`${id} referenceOffsets must have as many rows as featureOffsets`);
      }
    }
    validatePackedView(props.parameters, ['float32'], `${id} parameters`);
    if (props.parameters.length < GPU_VERTEX_SNAP_PARAMETER_LENGTH) {
      throw new Error(
        `${id} parameters must hold ${GPU_VERTEX_SNAP_PARAMETER_LENGTH} float32 values`
      );
    }
    validatePackedView(props.output.positions, ['float32x2'], `${id} output.positions`);
    if (props.output.positions.length !== props.positions.length) {
      throw new Error(`${id} output.positions must hold ${props.positions.length} rows`);
    }
    if (props.output.referenceRows) {
      validatePackedUint32View(props.output.referenceRows, `${id} output.referenceRows`);
      if (props.output.referenceRows.length !== props.positions.length) {
        throw new Error(`${id} output.referenceRows must hold ${props.positions.length} rows`);
      }
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      [props.output.positions, props.output.referenceRows],
      [
        props.positions,
        props.referencePositions,
        props.featureOffsets,
        props.referenceOffsets,
        props.parameters
      ]
    );
  }

  /** Returns the single snapping node. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id} = this;
    validateGraphViewsBelongToGraph(id, graph, [
      props.positions,
      props.referencePositions,
      props.featureOffsets,
      props.referenceOffsets,
      props.parameters,
      props.output.positions,
      props.output.referenceRows
    ]);
    const pairwise = Boolean(props.featureOffsets && props.referenceOffsets);
    let bvhInternalNodeCount = 0;
    const bindings: WGSLKernelBinding[] = [
      {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
      {name: 'referencePositions', view: props.referencePositions, type: 'f32', access: 'read'}
    ];
    if (props.featureOffsets && props.referenceOffsets) {
      bindings.push(
        {name: 'featureOffsets', view: props.featureOffsets, type: 'u32', access: 'read'},
        {name: 'referenceOffsets', view: props.referenceOffsets, type: 'u32', access: 'read'}
      );
    }
    bindings.push(
      {name: 'parameters', view: props.parameters, type: 'f32', access: 'read'},
      {name: 'outputPositions', view: props.output.positions, type: 'f32', access: 'read_write'}
    );
    if (props.output.referenceRows) {
      bindings.push({
        name: 'outputReferenceRows',
        view: props.output.referenceRows,
        type: 'u32',
        access: 'read_write'
      });
    }
    const useBVH = !pairwise && props.referencePositions.length > BVH_MINIMUM_REFERENCES;
    const nodes: GPUCommandNode<Parameters>[] = [];
    if (useBVH) {
      const built = getSegmentBVHNodes(graph, {
        id,
        operation: OPERATION,
        // A reference vertex is a degenerate box.
        minima: props.referencePositions,
        maxima: props.referencePositions,
        leafCapacity: getNextPowerOfTwo(props.referencePositions.length),
        spatialSort: true
      });
      nodes.push(...built.nodes);
      bindings.push(
        {name: 'nodeMinima', view: built.bvh.nodeMinima, type: 'f32', access: 'read'},
        {name: 'nodeMaxima', view: built.bvh.nodeMaxima, type: 'f32', access: 'read'},
        {name: 'leafIds', view: built.bvh.leafIds, type: 'u32', access: 'read'}
      );
      bvhInternalNodeCount = built.bvh.internalNodeCount;
    }
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-snap`,
        operation: OPERATION,
        variant: pairwise ? 'pairwise' : useBVH ? 'global-bvh' : 'global',
        bindings,
        invocationCount: props.positions.length,
        declarations: /* wgsl */ `
const FEATURE_COUNT: u32 = ${props.featureOffsets ? props.featureOffsets.length - 1 : 0}u;
const REFERENCE_COUNT: u32 = ${props.referencePositions.length}u;
const NO_REFERENCE: u32 = 0xffffffffu;
${
  useBVH
    ? `const INTERNAL_NODE_COUNT: u32 = ${bvhInternalNodeCount}u;
fn boxDistanceSquared(node: u32, point: vec2f) -> f32 {
  let component = node * 2u;
  let minimum = vec2f(nodeMinima[nodeMinimaOffset + component], nodeMinima[nodeMinimaOffset + component + 1u]);
  let maximum = vec2f(nodeMaxima[nodeMaximaOffset + component], nodeMaxima[nodeMaximaOffset + component + 1u]);
  let gap = max(max(minimum - point, point - maximum), vec2f(0.0));
  return dot(gap, gap);
}`
    : ''
}
${
  pairwise
    ? `
// Upper-bound binary search: the feature owning a position row, or FEATURE_COUNT when none.
fn findFeature(row: u32) -> u32 {
  var low = 0u;
  var high = FEATURE_COUNT + 1u;
  while (low < high) {
    let middle = (low + high) / 2u;
    if (featureOffsets[featureOffsetsOffset + middle] <= row) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  if (low == 0u || low > FEATURE_COUNT) {
    return FEATURE_COUNT;
  }
  return low - 1u;
}`
    : ''
}`,
        body: /* wgsl */ `let tolerance = max(parameters[parametersOffset], 0.0);
  let toleranceSquared = tolerance * tolerance;
  let vertex = vec2<f32>(positions[positionsOffset + 2u * index], positions[positionsOffset + 2u * index + 1u]);
  var referenceStart = 0u;
  var referenceEnd = REFERENCE_COUNT;
  ${
    pairwise
      ? `let feature = findFeature(index);
  if (feature == FEATURE_COUNT) {
    referenceEnd = 0u;
  } else {
    referenceStart = referenceOffsets[referenceOffsetsOffset + feature];
    referenceEnd = min(referenceOffsets[referenceOffsetsOffset + feature + 1u], REFERENCE_COUNT);
  }`
      : ''
  }
  var bestRow = NO_REFERENCE;
  var bestDistance = toleranceSquared;
  ${
    useBVH
      ? `// Branch and bound over the reference BVH. A box distance never exceeds the squared distance of
  // any vertex inside it, so pruning on '>' keeps equal-distance rows and the lowest row wins ties.
  if (vertex.x == vertex.x && vertex.y == vertex.y) {
    var stack: array<u32, 64>;
    var stackLower: array<f32, 64>;
    var depth = 1u;
    stack[0] = 0u;
    stackLower[0] = boxDistanceSquared(0u, vertex);
    loop {
      if (depth == 0u) { break; }
      depth = depth - 1u;
      if (stackLower[depth] > bestDistance) { continue; }
      let node = stack[depth];
      if (node < INTERNAL_NODE_COUNT) {
        let first = node * 2u + 1u;
        let second = first + 1u;
        let firstLower = boxDistanceSquared(first, vertex);
        let secondLower = boxDistanceSquared(second, vertex);
        // Push the farther child first so the nearer one is popped next.
        if (firstLower <= secondLower) {
          if (secondLower <= bestDistance) { stack[depth] = second; stackLower[depth] = secondLower; depth = depth + 1u; }
          if (firstLower <= bestDistance) { stack[depth] = first; stackLower[depth] = firstLower; depth = depth + 1u; }
        } else {
          if (firstLower <= bestDistance) { stack[depth] = first; stackLower[depth] = firstLower; depth = depth + 1u; }
          if (secondLower <= bestDistance) { stack[depth] = second; stackLower[depth] = secondLower; depth = depth + 1u; }
        }
        continue;
      }
      let row = leafIds[leafIdsOffset + node - INTERNAL_NODE_COUNT];
      if (row >= REFERENCE_COUNT) { continue; }
      let reference = vec2<f32>(referencePositions[referencePositionsOffset + 2u * row], referencePositions[referencePositionsOffset + 2u * row + 1u]);
      let delta = reference - vertex;
      let distanceSquared = dot(delta, delta);
      if (distanceSquared < bestDistance || (distanceSquared == bestDistance && bestRow != NO_REFERENCE && row < bestRow)) {
        bestDistance = distanceSquared;
        bestRow = row;
      }
    }
  }`
      : `for (var row = referenceStart; row < referenceEnd; row++) {
    let reference = vec2<f32>(referencePositions[referencePositionsOffset + 2u * row], referencePositions[referencePositionsOffset + 2u * row + 1u]);
    let delta = reference - vertex;
    let distanceSquared = dot(delta, delta);
    // Strictly closer than the tolerance and strictly better than the best so far: ties keep the lowest row.
    if (distanceSquared < bestDistance) {
      bestDistance = distanceSquared;
      bestRow = row;
    }
  }`
  }
  var snapped = vertex;
  if (bestRow != NO_REFERENCE) {
    snapped = vec2<f32>(referencePositions[referencePositionsOffset + 2u * bestRow], referencePositions[referencePositionsOffset + 2u * bestRow + 1u]);
  }
  outputPositions[outputPositionsOffset + 2u * index] = snapped.x;
  outputPositions[outputPositionsOffset + 2u * index + 1u] = snapped.y;
  ${props.output.referenceRows ? 'outputReferenceRows[outputReferenceRowsOffset + index] = bestRow;' : ''}`
      })
    );
    return nodes;
  }
}
