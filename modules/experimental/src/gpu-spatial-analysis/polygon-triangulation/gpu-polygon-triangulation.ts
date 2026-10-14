// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {createWGSLKernelNode, type WGSLKernelBinding} from '../../utils/wgsl-kernel-nodes';
import {
  GPU_POLYGON_TRIANGULATION_DEFAULT_MONOTONE_MINIMUM_VERTICES,
  getMonotoneTriangulationNodes,
  getGPUPolygonTriangulationMonotoneMaximumVertexCount,
  validatePolygonTriangulationMonotoneCapacity
} from './polygon-triangulation-monotone';
import {getSizeRouteWGSL} from './polygon-triangulation-monotone-wgsl';
import {
  validateCapacity,
  getStorageBindingElementLimit,
  validateGraphViewsBelongToGraph,
  validateMaximumCount,
  validatePackedView
} from '../../utils/gpu-contributor-utils';
import {EXACT_GEOMETRY_WGSL} from '../../utils/exact-geometry-wgsl';
import {EXACT_ORIENTATION_WGSL} from '../../utils/exact-orientation-wgsl';
import {
  resolveGPUSpatialContext,
  type GPUSpatialContext
} from '../../analysis-core/contracts/spatial-context';
import type {GPUPolygonGeometryFields} from '../../analysis-core/contracts/index';

const OPERATION = 'GPUPolygonTriangulation';

/** Default `maximumWork` of {@link GPUPolygonTriangulation}: about 4 million elementary steps. */
export const GPU_POLYGON_TRIANGULATION_DEFAULT_MAXIMUM_WORK = 4_000_000;

/**
 * Number of `uint32` index slots {@link GPUPolygonTriangulation} writes for a layout: exactly
 * `3 * (vertices + 2 * holes - 2)` per polygon, which sums to
 * `3 * (vertexCount + 2 * ringCount - 4 * polygonCount)`.
 */
export function getPolygonTriangulationIndexCount(
  vertexCount: number,
  ringCount: number,
  polygonCount: number
): number {
  return Math.max(0, 3 * (vertexCount + 2 * ringCount - 4 * polygonCount));
}

/**
 * Largest scratch node count of {@link GPUPolygonTriangulation} on `device`: each node takes 7
 * `uint32` words (28 bytes) with the z-order hash (the default) or 4 words (16 bytes) without it,
 * and the scratch must fit one storage binding, so 4793490 (8388608 without the hash) on a default
 * device (128 MiB bindings). The node count of a layout is
 * `2 * (vertices + 2 * (rings - polygons)) + 8 * polygons`, see
 * {@link getPolygonTriangulationNodeCount}, about `2` to `3` per input vertex.
 *
 * @param device Device whose `maxStorageBufferBindingSize` bounds the scratch.
 * @param useZOrderHash Whether the z-order hash is used (the default).
 */
export function getGPUPolygonTriangulationMaximumNodeCount(
  device: Device,
  useZOrderHash: boolean = true
): number {
  return getStorageBindingElementLimit(device, useZOrderHash ? 28 : 16);
}

/**
 * Number of scratch nodes {@link GPUPolygonTriangulation} allocates for a layout:
 * `2 * (vertices + 2 * (rings - polygons)) + 8 * polygons`, twice the minimum node count for
 * diagonal splits plus a margin per polygon.
 */
export function getPolygonTriangulationNodeCount(
  vertexCount: number,
  ringCount: number,
  polygonCount: number
): number {
  return 2 * (vertexCount + 2 * (ringCount - polygonCount)) + 8 * polygonCount;
}

/**
 * Throws unless the scratch of a polygon layout fits one storage binding of `device`: its
 * {@link getPolygonTriangulationNodeCount} is at most
 * {@link getGPUPolygonTriangulationMaximumNodeCount}. Contributors that triangulate internally call
 * it under their own ID, at declaration.
 *
 * @param device Device the graph runs on.
 * @param id Contributor ID, for the message.
 * @param vertexCount Vertex rows of the layout.
 * @param ringCount Rings of the layout.
 * @param polygonCount Polygons of the layout.
 * @param useZOrderHash Whether the z-order hash is used (the default).
 */
export function validatePolygonTriangulationScratch(
  device: Device,
  id: string,
  vertexCount: number,
  ringCount: number,
  polygonCount: number,
  useZOrderHash: boolean = true
): void {
  validateMaximumCount(
    id,
    'scratch nodes',
    getPolygonTriangulationNodeCount(vertexCount, ringCount, polygonCount),
    getGPUPolygonTriangulationMaximumNodeCount(device, useZOrderHash)
  );
}

/** Caller-owned outputs of {@link GPUPolygonTriangulation}. */
export type GPUPolygonTriangulationOutput = {
  /**
   * Triangle indices, at least {@link getPolygonTriangulationIndexCount} rows.
   * Polygon `p` owns the slice starting at
   * `3 * (ringOffsets[polygonOffsets[p]] + 2 * polygonOffsets[p] - 4 * p)`, a prefix that is
   * computed from the offsets, so no count pass is needed. Indices refer to rows of
   * `positions`. Slots a polygon does not use (collinear or duplicate vertices dropped,
   * or a failed polygon) hold the degenerate triangle `(first, first, first)` of that polygon, so
   * drawing the whole buffer is always safe.
   */
  indices: GraphDataView<'uint32'>;
  /**
   * Flag per polygon, `polygonCount` rows: 1 when the triangulation completed, 0 when
   * it failed (shell with fewer than three vertices, self-intersecting or otherwise
   * untriangulable rings, scratch overflow, or `maximumWork` exceeded). A failed polygon's slice
   * is entirely degenerate. The polygons of feature `f` are
   * `[featureOffsets[f], featureOffsets[f + 1])`.
   */
  valid: GraphDataView<'uint32'>;
  /** Optional count of real (non-degenerate padding) triangles per polygon, `polygonCount` rows. */
  triangleCount?: GraphDataView<'uint32'>;
};

/**
 * Properties for {@link GPUPolygonTriangulation}: the four-level layout of
 * {@link GPUPolygonGeometryFields} (`positions`, `featureOffsets`, `polygonOffsets`,
 * `ringOffsets`). Every polygon needs a shell of at least three vertices; the first ring of a
 * polygon is the shell and later rings are holes. Rings close implicitly, a repeated first vertex
 * is tolerated. Winding of either kind is accepted. Triangulation runs per polygon, so a
 * multipolygon feature contributes one slice per polygon and `output.valid` and
 * `output.triangleCount` have one row per polygon.
 *
 * Per-frame: the contents of every input buffer. Topology: view lengths and `maximumWork`.
 */
export type GPUPolygonTriangulationProps = GPUPolygonGeometryFields & {
  /** Prefix for generated node and transient IDs. Defaults to `'polygon-triangulation'`. */
  id?: string;
  /**
   * Planar only (default: native planar coordinates). A longitude-latitude context throws at
   * construction: project positions to a planar frame first, for example with
   * `createGPULocalFrame` (`contracts/local-frame`) and `GPULocalFrameProjection`
   * (`geometry-measures`), and pass the frame's context.
   */
  spatialContext?: GPUSpatialContext;
  /** Caller-owned outputs. */
  output: GPUPolygonTriangulationOutput;
  /**
   * Work budget per polygon in elementary steps (ring walks and predicate scans). A polygon that
   * exceeds it is flagged invalid instead of hanging the device, which bounds the O(n^2) worst
   * case of ear clipping. Defaults to {@link GPU_POLYGON_TRIANGULATION_DEFAULT_MAXIMUM_WORK},
   * which covers rings of a few thousand vertices. Baked into the shader.
   */
  maximumWork?: number;
  /**
   * Index ear tests with Earcut's z-order curve hash for polygons with more than 80 vertices
   * (default `true`). Ring nodes are kept in a second list sorted by Morton code, so an ear test
   * only visits the vertices whose code lies between the codes of the ear's bounding-box corners
   * instead of the whole ring. This turns the O(n^2) ear clipping of one large ring into about
   * O(n log n) for typical outlines, with identical output. Costs 12 extra scratch bytes per
   * node (28 bytes of scratch per node instead of 16). Pass `false` to keep the footprint small
   * when every polygon is tiny. Baked into the shader.
   */
  useZOrderHash?: boolean;
  /**
   * What happens to polygons that per-invocation Earcut cannot finish (default `'monotone'`).
   *
   * - `'monotone'`: polygons with at least {@link GPUPolygonTriangulationProps.monotoneMinimumVertices}
   *   vertices skip Earcut, and every polygon for which Earcut fails (`maximumWork` exceeded, or
   *   rings it cannot split) is retried, by a parallel y-monotone decomposition followed by a
   *   linear stack triangulation of each monotone piece. A polygon that skipped Earcut and that
   *   the monotone path flags invalid (rings touching at a vertex, input beyond
   *   {@link GPUPolygonTriangulationProps.monotoneMaximumVertices}) is then given to Earcut within
   *   `maximumWork`, so the method never does worse than `'none'`. The output contract is
   *   unchanged: the same slice, `valid` and `triangleCount`. Polygons that Earcut triangulates are
   *   bit-identical to `'none'`. The path costs fixed O(vertices) scans and sorts (sized by
   *   {@link GPUPolygonTriangulationProps.monotoneMaximumVertices}) and a few hundred bytes of
   *   scratch per vertex, even when no polygon is large.
   * - `'none'`: today's behavior, a polygon over `maximumWork` is flagged `valid = 0` with a
   *   degenerate slice. Use it when all polygons are known to be small.
   *
   * Baked into the graph.
   */
  largePolygonMethod?: 'monotone' | 'none';
  /**
   * Polygons with at least this many vertices (all rings) are routed straight to the monotone path
   * and skip Earcut (default 4096). Earcut with its z-order hash is near-linear for typical
   * outlines up to a few thousand vertices; its worst case, bounded only by `maximumWork`, is
   * quadratic, so a lower value trades Earcut's output for guaranteed linear-time-per-piece work.
   * Used with `largePolygonMethod: 'monotone'`. Baked into the shader.
   */
  monotoneMinimumVertices?: number;
  /**
   * Capacity of the monotone path in cleaned vertices (default: all `positions` rows). Scratch,
   * scans and sorts are sized by it, so a smaller value cuts the fixed cost for inputs whose large
   * polygons are known to total fewer vertices. Large polygons (in row order) whose vertices do not
   * fit skip the monotone path and are given to Earcut within `maximumWork` (`valid = 0` with a
   * degenerate slice if that fails too). At most
   * {@link getGPUPolygonTriangulationMonotoneMaximumVertexCount}; only an explicit larger value
   * throws. Inputs with more rows than that limit skip the monotone path entirely and behave like
   * `'none'`. Baked into the graph.
   */
  monotoneMaximumVertices?: number;
};

/** Polygons with more vertices than this use the z-order hash (same threshold as Earcut). */
const Z_ORDER_HASH_MINIMUM_VERTICES = 80;

/**
 * Triangulates polygons with holes on the GPU: the Earcut algorithm behind deck.gl's polygon
 * tessellation, comparable to `geo` `TriangulateEarcut` and Sedona `ST_TriangulatePolygon`.
 *
 * Each polygon runs on one invocation (the ear-clipping list is sequential), so throughput comes
 * from many polygons, not from one huge ring. Polygons that are too large for that, or on which
 * Earcut fails within `maximumWork`, take the monotone path
 * ({@link GPUPolygonTriangulationProps.largePolygonMethod}): a parallel y-monotone decomposition
 * and a linear stack triangulation of each monotone piece, with the same output contract. The
 * Earcut port follows Earcut: holes are bridged into the
 * shell by Eberly's rightmost-vertex rule (holes processed left to right), ears are clipped from a
 * circular list, and stalls fall back to filtering collinear points, curing local
 * self-intersections and splitting along a valid diagonal. Orientation tests use the exact
 * predicate of `GPUSegmentIntersection`, so collinear and near-degenerate input does not depend
 * on f32 rounding. Polygons above 80 vertices use Earcut's z-order hash
 * ({@link GPUPolygonTriangulationProps.useZOrderHash}), so ear tests visit only nearby vertices.
 *
 * Output is bounded without a count pass: exactly `3 * (vertices + 2 * holes - 2)` indices per
 * polygon, see {@link GPUPolygonTriangulationOutput.indices}. Like Earcut the result is not a
 * Delaunay triangulation and may contain slivers.
 *
 * **Limits.** Earcut's work is budgeted by `maximumWork`; a larger polygon is retried by the
 * monotone path, or with `largePolygonMethod: 'none'` flagged in `valid` rather than hung. Scratch memory is 56 bytes per vertex of input (twice the minimum node count for diagonal splits;
 * 32 bytes without the z-order hash) and must fit one storage binding: at most
 * {@link getGPUPolygonTriangulationMaximumNodeCount} scratch nodes (4793490 on a default device;
 * 8388608 with `useZOrderHash: false`), see {@link getPolygonTriangulationNodeCount}.
 * Polygons need at least three shell vertices and no empty rings, otherwise the index slices of
 * later polygons shift. Holes with fewer than three vertices are ignored.
 */
export class GPUPolygonTriangulation implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUPolygonTriangulationProps;
  /** Resolved spatial context (planar only). */
  readonly spatialContext: GPUSpatialContext;
  /** Number of polygons, `polygonOffsets.length - 1`. */
  readonly polygonCount: number;
  /** Number of rings, `ringOffsets.length - 1`. */
  readonly ringCount: number;
  /** Number of index rows the layout requires. */
  readonly indexCount: number;

  constructor(props: GPUPolygonTriangulationProps) {
    this.id = props.id ?? 'polygon-triangulation';
    this.spatialContext = resolveGPUSpatialContext(this.id, props.spatialContext);
    this.props = props;
    const {id} = this;
    const {output} = props;
    validatePackedView(props.positions, ['float32x2'], `${id} positions`);
    for (const name of ['featureOffsets', 'polygonOffsets', 'ringOffsets'] as const) {
      validatePackedView(props[name], ['uint32'], `${id} ${name}`);
      if (props[name].length < 1) {
        throw new Error(`${id} ${name} requires a terminal entry`);
      }
    }
    this.polygonCount = props.polygonOffsets.length - 1;
    this.ringCount = props.ringOffsets.length - 1;
    if (this.polygonCount < 1) {
      throw new Error(`${id} geometry must contain at least one polygon`);
    }
    if (props.positions.length < 1) {
      throw new Error(`${id} geometry must contain at least one vertex`);
    }
    this.indexCount = getPolygonTriangulationIndexCount(
      props.positions.length,
      this.ringCount,
      this.polygonCount
    );
    validatePackedView(output.indices, ['uint32'], `${id} output.indices`);
    if (output.indices.length < this.indexCount) {
      throw new Error(`${id} output.indices needs at least ${this.indexCount} rows`);
    }
    validatePackedView(output.valid, ['uint32'], `${id} output.valid`);
    if (output.valid.length !== this.polygonCount) {
      throw new Error(`${id} output.valid length must equal the polygon count`);
    }
    if (output.triangleCount) {
      validatePackedView(output.triangleCount, ['uint32'], `${id} output.triangleCount`);
      if (output.triangleCount.length !== this.polygonCount) {
        throw new Error(`${id} output.triangleCount length must equal the polygon count`);
      }
    }
    if (props.maximumWork !== undefined) {
      validateCapacity(id, 'maximumWork', props.maximumWork);
    }
    if (
      props.largePolygonMethod !== undefined &&
      !['monotone', 'none'].includes(props.largePolygonMethod)
    ) {
      throw new Error(`${id} largePolygonMethod must be 'monotone' or 'none'`);
    }
    if (props.monotoneMinimumVertices !== undefined) {
      validateCapacity(id, 'monotoneMinimumVertices', props.monotoneMinimumVertices);
    }
    if (props.monotoneMaximumVertices !== undefined) {
      validateCapacity(id, 'monotoneMaximumVertices', props.monotoneMaximumVertices);
    }
  }

  private getViews(): (GraphDataView | undefined)[] {
    const {positions, featureOffsets, polygonOffsets, ringOffsets, output} = this.props;
    const {indices, valid, triangleCount} = output;
    return [positions, featureOffsets, polygonOffsets, ringOffsets, indices, valid, triangleCount];
  }

  /**
   * Returns the Earcut node and, with `largePolygonMethod: 'monotone'`, the routing node before it
   * and the monotone-path nodes after it.
   */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, polygonCount, ringCount} = this;
    const {output} = props;
    validateGraphViewsBelongToGraph(id, graph, this.getViews());
    validatePolygonTriangulationScratch(
      graph.device,
      id,
      props.positions.length,
      ringCount,
      polygonCount,
      props.useZOrderHash ?? true
    );
    const vertexCount = props.positions.length;
    // Scratch nodes: 4 words each, with a capacity of 2 * (vertices + 2 * holes) + 8 per polygon.
    const nodeCount = getPolygonTriangulationNodeCount(vertexCount, ringCount, polygonCount);
    const useZOrderHash = props.useZOrderHash ?? true;
    // Words 0..4n-1 hold the ring nodes; the z-order list (z, previous, next) of node n follows at
    // 4 * nodeCount + 3 * n.
    const scratch = createTransientView(
      graph,
      `${id}-scratch`,
      'uint32',
      Math.max((useZOrderHash ? 7 : 4) * nodeCount, 4)
    );
    // The monotone path binds up to 40 bytes per input row; larger inputs keep today's behavior
    // (Earcut within `maximumWork`), so the default never makes an accepted input throw.
    const useMonotone =
      (props.largePolygonMethod ?? 'monotone') === 'monotone' &&
      vertexCount <= getGPUPolygonTriangulationMonotoneMaximumVertexCount(graph.device);
    const bindings: WGSLKernelBinding[] = [
      {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
      {name: 'ringOffsets', view: props.ringOffsets, type: 'u32', access: 'read'},
      {name: 'polygonOffsets', view: props.polygonOffsets, type: 'u32', access: 'read'},
      {name: 'indices', view: output.indices, type: 'u32', access: 'read_write'},
      {name: 'valid', view: output.valid, type: 'u32', access: 'read_write'},
      {name: 'scratch', view: scratch, type: 'u32', access: 'read_write'}
    ];
    if (output.triangleCount) {
      bindings.push({
        name: 'triangleCount',
        view: output.triangleCount,
        type: 'u32',
        access: 'read_write'
      });
    }
    // Monotone method: the Earcut kernel also reads and writes the routing word of each polygon.
    const routing = useMonotone
      ? createTransientView(graph, `${id}-routing`, 'uint32', Math.max(polygonCount, 1))
      : undefined;
    if (routing) {
      bindings.push({name: 'routing', view: routing, type: 'u32', access: 'read_write'});
    }
    const nodes: GPUCommandNode<Parameters>[] = [];
    let monotoneNodes: GPUCommandNode<Parameters>[] = [];
    if (routing) {
      const monotoneProps = {
        id: `${id}-monotone`,
        operation: OPERATION,
        positions: props.positions,
        polygonOffsets: props.polygonOffsets,
        ringOffsets: props.ringOffsets,
        routing,
        output,
        capacity: Math.max(1, Math.min(props.monotoneMaximumVertices ?? vertexCount, vertexCount)),
        minimumVertices:
          props.monotoneMinimumVertices ??
          GPU_POLYGON_TRIANGULATION_DEFAULT_MONOTONE_MINIMUM_VERTICES
      };
      if (props.monotoneMaximumVertices !== undefined) {
        validatePolygonTriangulationMonotoneCapacity(
          graph.device,
          id,
          props.monotoneMaximumVertices
        );
      }
      monotoneNodes = getMonotoneTriangulationNodes(graph, monotoneProps);
    }
    const monotoneMinimumVertices =
      props.monotoneMinimumVertices ?? GPU_POLYGON_TRIANGULATION_DEFAULT_MONOTONE_MINIMUM_VERTICES;
    const createEarcutNode = (nodeId: string, routingPass?: 'first' | 'fallback') =>
      createWGSLKernelNode<Parameters>(graph, {
        id: nodeId,
        operation: OPERATION,
        variant: 'earcut',
        bindings,
        invocationCount: polygonCount,
        workgroupSize: 64,
        declarations: `${EXACT_ORIENTATION_WGSL}
// Only sosPointSide and exactCompareSign are called; exactOrientSigns (the helper's orientSign
// call site) is never called here and is dropped as dead code.
${EXACT_GEOMETRY_WGSL}
${getTriangulationWGSL({
  vertexCount,
  polygonCount,
  indexLength: output.indices.length,
  nodeCount,
  useZOrderHash,
  maximumWork: props.maximumWork ?? GPU_POLYGON_TRIANGULATION_DEFAULT_MAXIMUM_WORK,
  writeTriangleCount: Boolean(output.triangleCount),
  routingPass,
  minimumVertices: monotoneMinimumVertices
})}`,
        body: 'triangulatePolygon(index);'
      });
    nodes.push(createEarcutNode(`${id}-earcut`, routing ? 'first' : undefined));
    // Polygons that the monotone path flagged invalid get a second chance with Earcut (route 4),
    // so the method never does worse than `'none'`: a valid polygon with rings that touch, or one
    // beyond the monotone capacity, is still triangulated within `maximumWork`.
    if (routing) {
      monotoneNodes.push(createEarcutNode(`${id}-earcut-fallback`, 'fallback'));
    }
    return [...nodes, ...monotoneNodes];
  }
}

function getTriangulationWGSL(options: {
  vertexCount: number;
  polygonCount: number;
  indexLength: number;
  nodeCount: number;
  useZOrderHash: boolean;
  maximumWork: number;
  writeTriangleCount: boolean;
  routingPass?: 'first' | 'fallback';
  minimumVertices: number;
}): string {
  return /* wgsl */ `
const VERTEX_COUNT: u32 = ${options.vertexCount}u;
const POLYGON_COUNT: u32 = ${options.polygonCount}u;
const INDEX_LENGTH: u32 = ${options.indexLength}u;
const SCRATCH_NODE_COUNT: u32 = ${options.nodeCount}u;
const Z_BASE: u32 = ${4 * options.nodeCount}u;
const USE_Z_HASH: bool = ${options.useZOrderHash};
const Z_HASH_MINIMUM_VERTICES: u32 = ${Z_ORDER_HASH_MINIMUM_VERTICES}u;
const WORK_LIMIT: u32 = ${options.maximumWork}u;
const NONE: u32 = 0xffffffffu;
${options.routingPass === 'first' ? getSizeRouteWGSL(options.minimumVertices) : ''}
var<private> nodeBase: u32;
var<private> nodeCursor: u32;
var<private> nodeLimit: u32;
var<private> workDone: u32;
var<private> failed: bool;
var<private> incomplete: bool;
var<private> filteredOut: bool;
var<private> triangleBase: u32;
var<private> triangleSlots: u32;
var<private> trianglesWritten: u32;
var<private> hashed: bool;
var<private> hashMinimum: vec2f;
var<private> hashScale: f32;

// Node n lives in scratch words 4n..4n+3: next, previous, vertex row, spare (hole queue, ring stack).
fn nextOf(node: u32) -> u32 { return scratch[scratchOffset + 4u * node]; }
fn prevOf(node: u32) -> u32 { return scratch[scratchOffset + 4u * node + 1u]; }
fn vertexOf(node: u32) -> u32 { return scratch[scratchOffset + 4u * node + 2u]; }
fn setNext(node: u32, value: u32) { scratch[scratchOffset + 4u * node] = value; }
// z-order list of node n: z, previous in z order, next in z order (NONE-terminated).
fn zOf(node: u32) -> u32 { return scratch[scratchOffset + Z_BASE + 3u * node]; }
fn prevZOf(node: u32) -> u32 { return scratch[scratchOffset + Z_BASE + 3u * node + 1u]; }
fn nextZOf(node: u32) -> u32 { return scratch[scratchOffset + Z_BASE + 3u * node + 2u]; }
fn setZ(node: u32, value: u32) { scratch[scratchOffset + Z_BASE + 3u * node] = value; }
fn setPrevZ(node: u32, value: u32) { scratch[scratchOffset + Z_BASE + 3u * node + 1u] = value; }
fn setNextZ(node: u32, value: u32) { scratch[scratchOffset + Z_BASE + 3u * node + 2u] = value; }
fn setPrev(node: u32, value: u32) { scratch[scratchOffset + 4u * node + 1u] = value; }
fn vertexAt(vertex: u32) -> vec2f {
  return vec2f(positions[positionsOffset + vertex * 2u], positions[positionsOffset + vertex * 2u + 1u]);
}
fn pointOf(node: u32) -> vec2f { return vertexAt(vertexOf(node)); }
fn samePoint(first: vec2f, second: vec2f) -> bool { return first.x == second.x && first.y == second.y; }
fn spend(amount: u32) {
  workDone += amount;
  if (workDone > WORK_LIMIT) { failed = true; }
}

// Sign of Earcut's area(p, q, r), which is minus the orientation: negative at a convex vertex
// of the (counter-clockwise) working order. Uncertain predicates count as zero.
fn areaSign(p: vec2f, q: vec2f, r: vec2f) -> i32 {
  let result = orientSign(p, q, r);
  if (result == 2) { return 0; }
  return -result;
}
fn orientOf(p: vec2f, q: vec2f, r: vec2f) -> i32 {
  let result = orientSign(p, q, r);
  if (result == 2) { return 0; }
  return result;
}

fn createNode(vertex: u32) -> u32 {
  if (nodeCursor >= nodeLimit) {
    failed = true;
    return nodeBase;
  }
  let node = nodeCursor;
  nodeCursor += 1u;
  scratch[scratchOffset + 4u * node + 2u] = vertex;
  if (hashed) {
    setPrevZ(node, NONE);
    setNextZ(node, NONE);
  }
  return node;
}
fn insertNode(vertex: u32, last: u32) -> u32 {
  let node = createNode(vertex);
  if (last == NONE) {
    setPrev(node, node);
    setNext(node, node);
  } else {
    let following = nextOf(last);
    setNext(node, following);
    setPrev(node, last);
    setPrev(following, node);
    setNext(last, node);
  }
  return node;
}
fn removeNode(node: u32) {
  if (hashed) {
    let previousZ = prevZOf(node);
    let followingZ = nextZOf(node);
    if (previousZ != NONE) { setNextZ(previousZ, followingZ); }
    if (followingZ != NONE) { setPrevZ(followingZ, previousZ); }
  }
  setPrev(nextOf(node), prevOf(node));
  setNext(prevOf(node), nextOf(node));
}
fn equalNodes(first: u32, second: u32) -> bool { return samePoint(pointOf(first), pointOf(second)); }

// Winding of a ring at its lexicographically smallest vertex: 1 counter-clockwise, -1 clockwise,
// 0 undetermined, 2 uncertain.
fn ringWinding(start: u32, count: u32) -> i32 {
  var extreme = 0u;
  for (var k = 1u; k < count; k++) {
    let candidate = vertexAt(start + k);
    let best = vertexAt(start + extreme);
    if (candidate.x < best.x || (candidate.x == best.x && candidate.y < best.y)) { extreme = k; }
  }
  let center = vertexAt(start + extreme);
  var previous = extreme;
  var next = extreme;
  var foundPrevious = false;
  var foundNext = false;
  for (var step = 1u; step < count; step++) {
    if (!foundPrevious) {
      previous = (extreme + count - step) % count;
      foundPrevious = !samePoint(vertexAt(start + previous), center);
    }
    if (!foundNext) {
      next = (extreme + step) % count;
      foundNext = !samePoint(vertexAt(start + next), center);
    }
  }
  if (!foundPrevious || !foundNext) { return 0; }
  return orientSign(vertexAt(start + previous), center, vertexAt(start + next));
}

// Circular list of a ring in counter-clockwise (shell) or clockwise (hole) working order.
fn linkedList(start: u32, count: u32, wantCounterClockwise: bool) -> u32 {
  if (count == 0u) { return NONE; }
  let winding = ringWinding(start, count);
  var forward = true;
  if (winding == 1 || winding == -1) { forward = (winding == 1) == wantCounterClockwise; }
  var last = NONE;
  for (var k = 0u; k < count; k++) {
    let row = select(start + count - 1u - k, start + k, forward);
    last = insertNode(row, last);
  }
  if (last != NONE && equalNodes(last, nextOf(last))) {
    removeNode(last);
    last = nextOf(last);
  }
  return last;
}

// Removes collinear and coincident nodes; with end == start the whole ring until stable.
fn filterPoints(start: u32, endIn: u32) -> u32 {
  var end = endIn;
  let full = end == start;
  var p = start;
  loop {
    if (failed) { break; }
    spend(1u);
    var again = false;
    let following = nextOf(p);
    if (p != following &&
        (equalNodes(p, following) || areaSign(pointOf(prevOf(p)), pointOf(p), pointOf(following)) == 0)) {
      if (full || p == end) { end = prevOf(p); }
      filteredOut = true;
      removeNode(p);
      p = prevOf(p);
      again = true;
    } else if (full || p != end) {
      p = following;
      again = !full;
    }
    if (!(again || p != end)) { break; }
  }
  return end;
}

fn pointInTriangle(a: vec2f, b: vec2f, c: vec2f, p: vec2f) -> bool {
  return orientOf(p, c, a) >= 0 && orientOf(p, a, b) >= 0 && orientOf(p, b, c) >= 0;
}

fn isEar(ear: u32) -> bool {
  let a = prevOf(ear);
  let c = nextOf(ear);
  let pa = pointOf(a);
  let pb = pointOf(ear);
  let pc = pointOf(c);
  let low = min(pa, min(pb, pc));
  let high = max(pa, max(pb, pc));
  var p = nextOf(c);
  while (p != a) {
    spend(1u);
    if (failed) { return false; }
    let pp = pointOf(p);
    if (pp.x >= low.x && pp.x <= high.x && pp.y >= low.y && pp.y <= high.y && !samePoint(pa, pp) &&
        pointInTriangle(pa, pb, pc, pp) &&
        areaSign(pointOf(prevOf(p)), pp, pointOf(nextOf(p))) >= 0) {
      return false;
    }
    p = nextOf(p);
  }
  return true;
}

// Morton code of a point on a 32768 x 32768 grid over the shell's bounding box. Quantisation is
// monotone in each coordinate (and clamped), so every point inside a box has a code between the
// codes of the box's low and high corners.
fn spreadBits(value: u32) -> u32 {
  var x = value;
  x = (x | (x << 8u)) & 0x00ff00ffu;
  x = (x | (x << 4u)) & 0x0f0f0f0fu;
  x = (x | (x << 2u)) & 0x33333333u;
  x = (x | (x << 1u)) & 0x55555555u;
  return x;
}
fn zOrderOf(point: vec2f) -> u32 {
  let scaled = clamp((point - hashMinimum) * hashScale, vec2f(0.0), vec2f(32767.0));
  return spreadBits(u32(scaled.x)) | (spreadBits(u32(scaled.y)) << 1u);
}

// Earcut's indexCurves: every node of the ring gets a code and joins a list sorted by it (stable
// bottom-up merge sort of the linked list).
fn indexCurves(start: u32) {
  var p = start;
  loop {
    spend(1u);
    setZ(p, zOrderOf(pointOf(p)));
    setPrevZ(p, prevOf(p));
    setNextZ(p, nextOf(p));
    p = nextOf(p);
    if (p == start) { break; }
  }
  setNextZ(prevZOf(start), NONE);
  setPrevZ(start, NONE);
  var list = start;
  var inSize = 1u;
  loop {
    var left = list;
    list = NONE;
    var tail = NONE;
    var merges = 0u;
    while (left != NONE) {
      merges += 1u;
      var right = left;
      var leftSize = 0u;
      for (var i = 0u; i < inSize; i++) {
        leftSize += 1u;
        right = nextZOf(right);
        if (right == NONE) { break; }
      }
      var rightSize = inSize;
      while (leftSize > 0u || (rightSize > 0u && right != NONE)) {
        spend(1u);
        var element = NONE;
        if (leftSize != 0u && (rightSize == 0u || right == NONE || zOf(left) <= zOf(right))) {
          element = left;
          left = nextZOf(left);
          leftSize -= 1u;
        } else {
          element = right;
          right = nextZOf(right);
          rightSize -= 1u;
        }
        if (tail != NONE) { setNextZ(tail, element); } else { list = element; }
        setPrevZ(element, tail);
        tail = element;
      }
      left = right;
    }
    setNextZ(tail, NONE);
    if (failed || merges <= 1u) { break; }
    inSize *= 2u;
  }
}

fn blocksEar(p: u32, a: u32, c: u32, pa: vec2f, pb: vec2f, pc: vec2f, low: vec2f, high: vec2f) -> bool {
  spend(1u);
  let pp = pointOf(p);
  return p != a && p != c && pp.x >= low.x && pp.x <= high.x && pp.y >= low.y && pp.y <= high.y &&
    !samePoint(pa, pp) && pointInTriangle(pa, pb, pc, pp) &&
    areaSign(pointOf(prevOf(p)), pp, pointOf(nextOf(p))) >= 0;
}

// Same decision as isEar, but only visits the nodes whose z-order code lies in the code range of
// the triangle's bounding box (Earcut's isEarHashed).
fn isEarHashed(ear: u32) -> bool {
  let a = prevOf(ear);
  let c = nextOf(ear);
  let pa = pointOf(a);
  let pb = pointOf(ear);
  let pc = pointOf(c);
  let low = min(pa, min(pb, pc));
  let high = max(pa, max(pb, pc));
  let minZ = zOrderOf(low);
  let maxZ = zOrderOf(high);
  var before = prevZOf(ear);
  var after = nextZOf(ear);
  while (before != NONE && zOf(before) >= minZ && after != NONE && zOf(after) <= maxZ) {
    if (failed) { return false; }
    if (blocksEar(before, a, c, pa, pb, pc, low, high)) { return false; }
    before = prevZOf(before);
    if (blocksEar(after, a, c, pa, pb, pc, low, high)) { return false; }
    after = nextZOf(after);
  }
  while (before != NONE && zOf(before) >= minZ) {
    if (failed) { return false; }
    if (blocksEar(before, a, c, pa, pb, pc, low, high)) { return false; }
    before = prevZOf(before);
  }
  while (after != NONE && zOf(after) <= maxZ) {
    if (failed) { return false; }
    if (blocksEar(after, a, c, pa, pb, pc, low, high)) { return false; }
    after = nextZOf(after);
  }
  return true;
}

fn isEarTest(ear: u32) -> bool {
  if (hashed) { return isEarHashed(ear); }
  return isEar(ear);
}

// Bounding-box test only: the caller has already established collinearity with areaSign().
fn onSegment(p: vec2f, q: vec2f, r: vec2f) -> bool {
  return q.x <= max(p.x, r.x) && q.x >= min(p.x, r.x) && q.y <= max(p.y, r.y) && q.y >= min(p.y, r.y);
}
fn segmentsIntersect(p1: vec2f, q1: vec2f, p2: vec2f, q2: vec2f, includeBoundary: bool) -> bool {
  let o1 = areaSign(p1, q1, p2);
  let o2 = areaSign(p1, q1, q2);
  let o3 = areaSign(p2, q2, p1);
  let o4 = areaSign(p2, q2, q1);
  if (((o1 > 0 && o2 < 0) || (o1 < 0 && o2 > 0)) && ((o3 > 0 && o4 < 0) || (o3 < 0 && o4 > 0))) { return true; }
  if (!includeBoundary) { return false; }
  if (o1 == 0 && onSegment(p1, p2, q1)) { return true; }
  if (o2 == 0 && onSegment(p1, q2, q1)) { return true; }
  if (o3 == 0 && onSegment(p2, p1, q2)) { return true; }
  if (o4 == 0 && onSegment(p2, q1, q2)) { return true; }
  return false;
}

fn locallyInside(a: u32, b: u32) -> bool {
  let pa = pointOf(a);
  let pb = pointOf(b);
  let before = pointOf(prevOf(a));
  let after = pointOf(nextOf(a));
  if (areaSign(before, pa, after) < 0) {
    return areaSign(pa, pb, after) >= 0 && areaSign(pa, before, pb) >= 0;
  }
  return areaSign(pa, pb, before) < 0 || areaSign(pa, after, pb) < 0;
}

// Whether the diagonal midpoint is inside the working polygon: a topology decision, so the parity
// of a ray toward +x uses the exact rule of utils/exact-geometry-wgsl.ts. The midpoint is only a
// rounded f32 placement; its side of each straddling edge is the exact orientSign (through
// orientOf, no new orientSign call site) with Simulation-of-Simplicity ties from sosPointSide.
fn middleInside(a: u32, b: u32) -> bool {
  let middle = (pointOf(a) + pointOf(b)) * 0.5;
  var p = a;
  var inside = false;
  loop {
    spend(1u);
    if (failed) { return false; }
    let following = nextOf(p);
    let edgeStart = pointOf(p);
    let edgeEnd = pointOf(following);
    if ((edgeStart.y >= middle.y) != (edgeEnd.y >= middle.y)) {
      // orientOf is orientSign (positive when the midpoint is left of edgeStart -> edgeEnd) with
      // uncertain counted as zero, which sosPointSide then breaks by perturbation.
      let side = sosPointSide(orientOf(edgeStart, edgeEnd, middle), edgeStart, edgeEnd);
      let isUpward = edgeEnd.y > edgeStart.y;
      if ((isUpward && side > 0) || (!isUpward && side < 0)) {
        inside = !inside;
      }
    }
    p = following;
    if (p == a) { break; }
  }
  return inside;
}

fn intersectsPolygon(a: u32, b: u32) -> bool {
  let pa = pointOf(a);
  let pb = pointOf(b);
  let low = min(pa, pb);
  let high = max(pa, pb);
  let indexA = vertexOf(a);
  let indexB = vertexOf(b);
  var p = a;
  loop {
    spend(1u);
    if (failed) { return true; }
    let following = nextOf(p);
    let edgeStart = pointOf(p);
    let edgeEnd = pointOf(following);
    let outside = (edgeStart.x > high.x && edgeEnd.x > high.x) || (edgeStart.x < low.x && edgeEnd.x < low.x) ||
      (edgeStart.y > high.y && edgeEnd.y > high.y) || (edgeStart.y < low.y && edgeEnd.y < low.y);
    if (!outside) {
      let indexFrom = vertexOf(p);
      let indexTo = vertexOf(following);
      if (indexFrom != indexA && indexTo != indexA && indexFrom != indexB && indexTo != indexB &&
          segmentsIntersect(edgeStart, edgeEnd, pa, pb, true)) {
        return true;
      }
    }
    p = following;
    if (p == a) { break; }
  }
  return false;
}

fn isValidDiagonal(a: u32, b: u32) -> bool {
  let pa = pointOf(a);
  let pb = pointOf(b);
  let zeroLength = samePoint(pa, pb) &&
    areaSign(pointOf(prevOf(a)), pa, pointOf(nextOf(a))) > 0 &&
    areaSign(pointOf(prevOf(b)), pb, pointOf(nextOf(b))) > 0;
  if (vertexOf(nextOf(a)) == vertexOf(b)) { return false; }
  if (!(zeroLength || (locallyInside(a, b) && locallyInside(b, a) &&
      (areaSign(pointOf(prevOf(a)), pa, pointOf(prevOf(b))) != 0 ||
       areaSign(pa, pointOf(prevOf(b)), pb) != 0)))) {
    return false;
  }
  if (intersectsPolygon(a, b)) { return false; }
  return zeroLength || middleInside(a, b);
}

// Splits (a, b) in two rings (or merges a hole ring into the shell); returns b's duplicate.
fn splitPolygon(a: u32, b: u32) -> u32 {
  let a2 = createNode(vertexOf(a));
  let b2 = createNode(vertexOf(b));
  let an = nextOf(a);
  let bp = prevOf(b);
  setNext(a, b);
  setPrev(b, a);
  setNext(a2, an);
  setPrev(an, a2);
  setNext(b2, a2);
  setPrev(a2, b2);
  setNext(bp, b2);
  setPrev(b2, bp);
  return b2;
}

fn emitTriangle(first: u32, second: u32, third: u32) {
  if (trianglesWritten >= triangleSlots) {
    failed = true;
    return;
  }
  let slot = triangleBase + 3u * trianglesWritten;
  indices[indicesOffset + slot] = first;
  indices[indicesOffset + slot + 1u] = second;
  indices[indicesOffset + slot + 2u] = third;
  trianglesWritten += 1u;
}

fn cureLocalIntersections(startIn: u32) -> u32 {
  var start = startIn;
  var p = start;
  var cured = false;
  loop {
    spend(1u);
    if (failed) { break; }
    let a = prevOf(p);
    let following = nextOf(p);
    let b = nextOf(following);
    if (segmentsIntersect(pointOf(a), pointOf(p), pointOf(following), pointOf(b), false) &&
        locallyInside(a, b) && locallyInside(b, a)) {
      emitTriangle(vertexOf(a), vertexOf(p), vertexOf(b));
      removeNode(p);
      removeNode(following);
      start = b;
      p = b;
      cured = true;
    }
    p = nextOf(p);
    if (p == start) { break; }
  }
  if (cured) { return filterPoints(p, p); }
  return p;
}

// First valid diagonal of the ring as (a, b), or (NONE, NONE).
fn findSplit(start: u32) -> vec2<u32> {
  var a = start;
  loop {
    var b = nextOf(nextOf(a));
    while (b != prevOf(a)) {
      spend(1u);
      if (failed) { return vec2<u32>(NONE, NONE); }
      if (vertexOf(a) != vertexOf(b) && isValidDiagonal(a, b)) { return vec2<u32>(a, b); }
      b = nextOf(b);
    }
    a = nextOf(a);
    if (a == start) { break; }
  }
  return vec2<u32>(NONE, NONE);
}

fn pushRing(depth: u32, head: u32) {
  if (nodeBase + depth >= nodeLimit) {
    failed = true;
    return;
  }
  scratch[scratchOffset + 4u * (nodeBase + depth) + 3u] = head;
}
fn popRing(depth: u32) -> u32 {
  return scratch[scratchOffset + 4u * (nodeBase + depth) + 3u];
}

// Earcut's earcutLinked, with the recursion of splitEarcut replaced by a stack of ring heads.
fn clipEars(first: u32) {
  var ear = first;
  var depth = 0u;
  loop {
    if (hashed) { indexCurves(ear); }
    var stop = ear;
    var cured = false;
    loop {
      if (failed) { break; }
      spend(1u);
      let before = prevOf(ear);
      let after = nextOf(ear);
      if (before == after) { break; }
      var earFound = false;
      if (areaSign(pointOf(before), pointOf(ear), pointOf(after)) < 0) {
        earFound = isEarTest(ear);
      }
      if (earFound) {
        emitTriangle(vertexOf(before), vertexOf(ear), vertexOf(after));
        removeNode(ear);
        ear = after;
        stop = after;
        continue;
      }
      ear = after;
      if (ear == stop) {
        filteredOut = false;
        ear = filterPoints(ear, ear);
        if (filteredOut) {
          stop = ear;
          continue;
        }
        if (!cured) {
          ear = cureLocalIntersections(ear);
          stop = ear;
          cured = true;
          continue;
        }
        let diagonal = findSplit(ear);
        if (diagonal.x == NONE) {
          incomplete = true;
          break;
        }
        let other = splitPolygon(diagonal.x, diagonal.y);
        let keptHead = filterPoints(diagonal.x, nextOf(diagonal.x));
        let otherHead = filterPoints(other, nextOf(other));
        pushRing(depth, otherHead);
        depth += 1u;
        ear = keptHead;
        stop = keptHead;
        cured = false;
      }
    }
    if (failed || incomplete || depth == 0u) { break; }
    depth -= 1u;
    ear = popRing(depth);
  }
}

fn isLeftOf(first: u32, second: u32) -> bool {
  let a = pointOf(first);
  let b = pointOf(second);
  if (a.x != b.x) { return a.x < b.x; }
  if (a.y != b.y) { return a.y < b.y; }
  let nextA = pointOf(nextOf(first)) - a;
  let nextB = pointOf(nextOf(second)) - b;
  let slopeA = select(select(-1.0e30, 1.0e30, nextA.y > 0.0), nextA.y / nextA.x, nextA.x != 0.0);
  let slopeB = select(select(-1.0e30, 1.0e30, nextB.y > 0.0), nextB.y / nextB.x, nextB.x != 0.0);
  return slopeA < slopeB;
}

fn sectorContainsSector(m: u32, p: u32) -> bool {
  let pm = pointOf(m);
  return areaSign(pointOf(prevOf(m)), pm, pointOf(prevOf(p))) < 0 &&
    areaSign(pointOf(nextOf(p)), pm, pointOf(nextOf(m))) < 0;
}

// David Eberly's bridge between a hole's leftmost vertex and the outer ring, or NONE. The f32
// ray intersection x below only places the candidate bridge vertex (it is not a parity or
// containment decision); validity of the resulting diagonal is decided exactly by isValidDiagonal.
fn findHoleBridge(hole: u32, outerNode: u32) -> u32 {
  let hp = pointOf(hole);
  var p = outerNode;
  var qx = -3.0e38;
  var m = NONE;
  if (equalNodes(hole, p)) { return p; }
  loop {
    spend(1u);
    if (failed) { return NONE; }
    let following = nextOf(p);
    let edgeStart = pointOf(p);
    let edgeEnd = pointOf(following);
    if (equalNodes(hole, following)) { return following; }
    if (hp.y <= edgeStart.y && hp.y >= edgeEnd.y && edgeEnd.y != edgeStart.y) {
      let x = edgeStart.x + (hp.y - edgeStart.y) * (edgeEnd.x - edgeStart.x) / (edgeEnd.y - edgeStart.y);
      if (x <= hp.x && x > qx) {
        qx = x;
        m = select(following, p, edgeStart.x < edgeEnd.x);
        if (x == hp.x) { return m; }
      }
    }
    p = following;
    if (p == outerNode) { break; }
  }
  if (m == NONE) { return NONE; }
  let pm = pointOf(m);
  let stop = m;
  var tanMin = 3.0e38;
  p = m;
  loop {
    spend(1u);
    if (failed) { return NONE; }
    let pp = pointOf(p);
    if (hp.x >= pp.x && pp.x >= pm.x && hp.x != pp.x) {
      let corner = select(vec2f(qx, hp.y), vec2f(hp.x, hp.y), hp.y < pm.y);
      let corner2 = select(vec2f(hp.x, hp.y), vec2f(qx, hp.y), hp.y < pm.y);
      if (pointInTriangle(corner, pm, corner2, pp)) {
        let tangent = abs(hp.y - pp.y) / (hp.x - pp.x);
        let touchesEdge = pp.y == hp.y && pointOf(nextOf(p)).y == hp.y && pointOf(nextOf(p)).x > hp.x;
        if ((locallyInside(p, hole) || touchesEdge) &&
            (tangent < tanMin || (tangent == tanMin &&
              (pp.x > pm.x || (pp.x == pm.x && sectorContainsSector(m, p)))))) {
          m = p;
          tanMin = tangent;
        }
      }
    }
    p = nextOf(p);
    if (p == stop) { break; }
  }
  return m;
}

fn getLeftmost(start: u32) -> u32 {
  var leftmost = start;
  var p = start;
  loop {
    spend(1u);
    if (failed) { return leftmost; }
    let a = pointOf(p);
    let b = pointOf(leftmost);
    if (a.x < b.x || (a.x == b.x && a.y < b.y)) { leftmost = p; }
    p = nextOf(p);
    if (p == start) { break; }
  }
  return leftmost;
}

// Links every hole ring (rings ringFirst + 1 ..< ringLast) into the outer ring.
fn eliminateHoles(ringFirst: u32, ringLast: u32, outerIn: u32) -> u32 {
  var outerNode = outerIn;
  var queued = 0u;
  for (var ring = ringFirst + 1u; ring < ringLast; ring++) {
    let start = ringOffsets[ringOffsetsOffset + ring];
    let end = min(ringOffsets[ringOffsetsOffset + ring + 1u], VERTEX_COUNT);
    if (end < start + 3u) { continue; }
    let list = linkedList(start, end - start, false);
    if (list == NONE || failed) { continue; }
    scratch[scratchOffset + 4u * (nodeBase + queued) + 3u] = getLeftmost(list);
    queued += 1u;
  }
  for (var round = 0u; round < queued; round++) {
    // Selection sort by Earcut's compareXYSlope: leftmost, then lowest, then smallest slope.
    var best = NONE;
    var bestSlot = 0u;
    for (var slot = 0u; slot < queued; slot++) {
      spend(1u);
      let candidate = scratch[scratchOffset + 4u * (nodeBase + slot) + 3u];
      if (candidate == NONE) { continue; }
      if (best == NONE || isLeftOf(candidate, best)) {
        best = candidate;
        bestSlot = slot;
      }
    }
    if (best == NONE || failed) { break; }
    scratch[scratchOffset + 4u * (nodeBase + bestSlot) + 3u] = NONE;
    let bridge = findHoleBridge(best, outerNode);
    if (bridge == NONE) { continue; }
    let bridgeReverse = splitPolygon(bridge, best);
    filterPoints(bridgeReverse, nextOf(bridgeReverse));
    outerNode = filterPoints(bridge, nextOf(bridge));
  }
  return filterPoints(outerNode, outerNode);
}

fn baseNodeOf(polygon: u32) -> i32 {
  let ringPrefix = i32(polygonOffsets[polygonOffsetsOffset + polygon]);
  let vertexPrefix = i32(ringOffsets[ringOffsetsOffset + u32(ringPrefix)]);
  return 2 * (vertexPrefix + 2 * (ringPrefix - i32(polygon))) + 8 * i32(polygon);
}
fn baseSlotOf(polygon: u32) -> i32 {
  let ringPrefix = i32(polygonOffsets[polygonOffsetsOffset + polygon]);
  let vertexPrefix = i32(ringOffsets[ringOffsetsOffset + u32(ringPrefix)]);
  return 3 * (vertexPrefix + 2 * ringPrefix - 4 * i32(polygon));
}

fn triangulatePolygon(polygon: u32) {
  ${options.routingPass === 'first' ? 'routing[routingOffset + polygon] = sizeRouteOf(polygon);\n  if (routing[routingOffset + polygon] == 1u) { return; }' : ''}
  ${options.routingPass === 'fallback' ? 'if (routing[routingOffset + polygon] != 4u) { return; }' : ''}
  failed = false;
  incomplete = false;
  filteredOut = false;
  workDone = 0u;
  trianglesWritten = 0u;
  let slotStart = baseSlotOf(polygon);
  let slotEnd = min(baseSlotOf(polygon + 1u), i32(INDEX_LENGTH));
  let slotCount = max(slotEnd - slotStart, 0);
  triangleBase = u32(max(slotStart, 0));
  triangleSlots = u32(slotCount) / 3u;
  let nodeStart = baseNodeOf(polygon);
  let nodeEnd = min(baseNodeOf(polygon + 1u), i32(SCRATCH_NODE_COUNT));
  nodeBase = u32(max(nodeStart, 0));
  nodeCursor = nodeBase;
  nodeLimit = u32(max(nodeEnd, nodeStart));
  let ringFirst = polygonOffsets[polygonOffsetsOffset + polygon];
  let ringLast = polygonOffsets[polygonOffsetsOffset + polygon + 1u];
  let shellStart = ringOffsets[ringOffsetsOffset + ringFirst];
  let shellEnd = min(ringOffsets[ringOffsetsOffset + ringFirst + 1u], VERTEX_COUNT);
  let padVertex = shellStart;
  let polygonEnd = min(ringOffsets[ringOffsetsOffset + ringLast], VERTEX_COUNT);
  hashed = USE_Z_HASH && polygonEnd > shellStart + Z_HASH_MINIMUM_VERTICES;
  if (ringLast <= ringFirst || shellEnd < shellStart + 3u || triangleSlots == 0u ||
      nodeLimit < nodeBase + ringLast - ringFirst) {
    failed = true;
  } else {
    var outerNode = linkedList(shellStart, shellEnd - shellStart, true);
    if (outerNode == NONE || nextOf(outerNode) == prevOf(outerNode)) {
      failed = true;
    } else {
      if (hashed) {
        var low = vertexAt(shellStart);
        var high = low;
        for (var row = shellStart + 1u; row < shellEnd; row++) {
          low = min(low, vertexAt(row));
          high = max(high, vertexAt(row));
        }
        let extent = max(high.x - low.x, high.y - low.y);
        hashMinimum = low;
        hashScale = select(0.0, 32767.0 / extent, extent > 0.0);
      }
      if (ringLast - ringFirst > 1u) { outerNode = eliminateHoles(ringFirst, ringLast, outerNode); }
      clipEars(outerNode);
    }
  }
  let ok = !failed && !incomplete;
  ${options.routingPass === 'first' ? 'if (!ok && ringLast > ringFirst && shellEnd >= shellStart + 3u) { routing[routingOffset + polygon] = 2u; }' : ''}
  let written = select(0u, trianglesWritten, ok);
  // Failed polygons overwrite everything with degenerate triangles; partial output is unusable.
  for (var slot = written; slot < triangleSlots; slot++) {
    let base = indicesOffset + triangleBase + 3u * slot;
    indices[base] = padVertex;
    indices[base + 1u] = padVertex;
    indices[base + 2u] = padVertex;
  }
  valid[validOffset + polygon] = select(0u, 1u, ok);
  ${options.writeTriangleCount ? 'triangleCount[triangleCountOffset + polygon] = written;' : ''}
}
`;
}
