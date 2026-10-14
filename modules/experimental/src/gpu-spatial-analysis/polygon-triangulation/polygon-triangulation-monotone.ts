// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {
  createTransientView,
  GPUScan,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {
  createWGSLActiveCountDispatch,
  createWGSLKernelNode,
  type WGSLKernelBinding,
  type WGSLKernelProps
} from '../../utils/wgsl-kernel-nodes';
import {createStableLexicographicIndexSortNodes} from '../../utils/stable-lexicographic-index-sort';
import {
  getStorageBindingElementLimit,
  validateMaximumCount
} from '../../utils/gpu-contributor-utils';
import type {PolygonTriangulationLayout} from './polygon-triangulation-layout';
import {EXACT_ORIENTATION_WGSL} from '../../utils/exact-orientation-wgsl';
import {
  getPolygonEdgeSlabBindings,
  getPolygonEdgeSlabLoopWGSL,
  getPolygonEdgeSlabNodes,
  getPolygonEdgeSlabQueryWGSL
} from '../spatial-join/polygon-edge-slabs';
import {
  getCompactSource,
  getCountsSource,
  getDiagonalsSource,
  getFinalSource,
  getFlipSource,
  getIndexSource,
  getJumpInitSource,
  getJumpRoundSource,
  getJumpStateSource,
  getKeysSource,
  getLeftEdgeBody,
  getLinkSource,
  getMeasureSource,
  getMembersSource,
  getMonotoneConstantsWGSL,
  getNextSource,
  getPadSource,
  getPieceCountSource,
  getPiecePlanSource,
  getPiecesSource,
  getPolygonPlanSource,
  getRankSource,
  getRepresentativeCountSource,
  getRingSource,
  getRouteCountsSource,
  getRouteOverflowSource,
  getTriangulateSource,
  MONOTONE_COUNT,
  MONOTONE_COUNT_WORDS,
  MONOTONE_LEFT_EDGE_WGSL,
  MONOTONE_ORIENT_WGSL,
  MONOTONE_PIECE_STRIDE,
  MONOTONE_POSITION_WGSL,
  MONOTONE_STATS_STRIDE,
  MONOTONE_VERTEX_A_WGSL,
  MONOTONE_VERTEX_B_WGSL,
  type MonotoneKernelSource,
  type MonotoneSizes
} from './polygon-triangulation-monotone-wgsl';

/**
 * Default `monotoneMinimumVertices` of `GPUPolygonTriangulation`: polygons with at least this many
 * vertices (all rings) skip Earcut and take the monotone path. Earcut with its z-order hash is
 * near-linear for typical outlines up to a few thousand vertices, within the default
 * `maximumWork`; above that its cost grows and one invocation per polygon stops scaling.
 */
export const GPU_POLYGON_TRIANGULATION_DEFAULT_MONOTONE_MINIMUM_VERTICES = 4096;

/**
 * Bytes per cleaned vertex of the largest binding of the monotone path: the two pointer-jumping
 * arrays hold five half-edges of two words per vertex. Every other binding is smaller, so the
 * path's vertex capacity is bounded by `maxStorageBufferBindingSize / 40`.
 */
export const GPU_POLYGON_TRIANGULATION_MONOTONE_BYTES_PER_VERTEX = 40;

/**
 * Largest cleaned-vertex capacity of the monotone path on `device`
 * (`floor(maxStorageBufferBindingSize / 40)`, 3355443 on a default device). It is also the largest
 * number of input vertex rows the path accepts, because the shared edge-slab index binds
 * `40` bytes per row.
 *
 * @param device Device whose `maxStorageBufferBindingSize` bounds the bindings.
 */
export function getGPUPolygonTriangulationMonotoneMaximumVertexCount(device: Device): number {
  return getStorageBindingElementLimit(device, GPU_POLYGON_TRIANGULATION_MONOTONE_BYTES_PER_VERTEX);
}

/**
 * Throws unless an explicit monotone capacity fits the bindings of `device`.
 *
 * @param device Device the graph runs on.
 * @param id Contributor ID, for the message.
 * @param capacity Cleaned-vertex capacity of the monotone path.
 */
export function validatePolygonTriangulationMonotoneCapacity(
  device: Device,
  id: string,
  capacity: number
): void {
  validateMaximumCount(
    id,
    'monotoneMaximumVertices',
    capacity,
    getGPUPolygonTriangulationMonotoneMaximumVertexCount(device)
  );
}

/** Inputs of {@link getMonotoneTriangulationNodes}. @internal */
export type MonotoneTriangulationProps = {
  /** Prefix of node and transient IDs. */
  id: string;
  /** `workload.operation` of the nodes. */
  operation: string;
  positions: GraphDataView<'float32x2'>;
  polygonOffsets: GraphDataView<'uint32'>;
  ringOffsets: GraphDataView<'uint32'>;
  /** Optional prepared offsets and degeneracy flags from the public triangulator. @internal */
  layout?: PolygonTriangulationLayout;
  /**
   * Routing word per polygon, `polygonCount` rows: 0 Earcut only, 1 size-routed (Earcut skipped),
   * 2 Earcut failed, 3 overflowed the capacity, 4 the monotone path failed (Earcut fallback pass).
   * Written by the Earcut kernel's first pass (0, 1, 2) and by this module (3, 4).
   */
  routing: GraphDataView<'uint32'>;
  /** Caller-owned outputs of the triangulation. */
  output: {
    indices: GraphDataView<'uint32'>;
    valid: GraphDataView<'uint32'>;
    triangleCount?: GraphDataView<'uint32'>;
  };
  /** Cleaned-vertex capacity (scratch, sorts and scans are sized by it). */
  capacity: number;
  /** Polygons with at least this many vertices skip Earcut (see `getSizeRouteWGSL`). */
  minimumVertices: number;
  /**
   * Number of unrolled pairs of pointer-jumping rounds. Defaults to the pairs needed for
   * `capacity`; tests build at N and N + 1 pairs to prove that the round count does not reach the
   * WGSL (library rule 12).
   */
  jumpPairCount?: number;
  /** Receives every step in execution order, for the test harness. */
  recorder?: MonotoneStepRecorder;
};

/** One recorded step of the monotone path, for the CPU harness of the specs. @internal */
export type MonotoneStep =
  | {
      kind: 'kernel';
      id: string;
      props: WGSLKernelProps<unknown>;
      /** Present for kernels behind an active-count dispatch: how many threads to run. */
      gate?: {countRow: number; invocationsPerItem: number; workgroupSize: number};
    }
  | {kind: 'scan'; id: string; input: GraphDataView<'uint32'>; output: GraphDataView<'uint32'>}
  | {
      kind: 'sort';
      id: string;
      keys: readonly GraphDataView<'uint32'>[];
      indices: GraphDataView<'uint32'>;
      sortedIndices: GraphDataView<'uint32'>;
    }
  | {
      kind: 'slabs';
      id: string;
      params: GraphDataView<'uint32'>;
      offsets: GraphDataView<'uint32'>;
      entries: GraphDataView<'uint32x2'>;
    };

/** Receives the steps of {@link getMonotoneTriangulationNodes}. @internal */
export type MonotoneStepRecorder = (step: MonotoneStep) => void;

const WORKGROUP_SIZE = 256;
const PIECE_WORKGROUP_SIZE = 64;

function getSizes(props: MonotoneTriangulationProps): MonotoneSizes {
  return {
    preparedLayout: Boolean(props.layout),
    vertexCount: props.positions.length,
    polygonCount: props.polygonOffsets.length - 1,
    ringCount: props.ringOffsets.length - 1,
    capacity: props.capacity,
    indexLength: props.output.indices.length,
    minimumVertices: props.minimumVertices
  };
}

/** Number of bits needed to store `value`. */
function getBitCount(value: number): number {
  return Math.max(1, 32 - Math.clz32(value));
}

/**
 * Pairs of pointer-jumping rounds needed for polygons of at most `capacity` cleaned vertices: a
 * cycle has at most `3 n` half-edges, so `ceil(log2(3 n + 3))` rounds close it.
 */
export function getMonotoneJumpPairCount(capacity: number): number {
  const rounds = Math.ceil(Math.log2(3 * capacity + 3));
  return Math.ceil(rounds / 2);
}

/**
 * Builds the nodes of the monotone path, to run after the Earcut kernel: route counts and scan,
 * overflow, the clean/rank/orient/classify sweep, the left-edge ray shooting through the shared
 * edge-slab index, the helper diagonals, the half-edge next-pointers, the pointer-jumping piece
 * labels and the piece triangulation, then validity and padding. Only polygons routed by size or
 * by an Earcut failure are processed; the dispatch of every kernel after the routing is sized by a
 * GPU-resident count, so inputs without large polygons launch (almost) no work after the routing
 * scan and the fixed-size sorts and scans.
 *
 * @internal
 */
export function getMonotoneTriangulationNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: MonotoneTriangulationProps
): GPUCommandNode<Parameters>[] {
  const {id, operation, positions, routing, output} = props;
  const polygonOffsets = props.layout?.polygonOffsets ?? props.polygonOffsets;
  const ringOffsets = props.layout?.ringOffsets ?? props.ringOffsets;
  const sizes = getSizes(props);
  const {vertexCount, polygonCount, ringCount, capacity} = sizes;
  const nodes: GPUCommandNode<Parameters>[] = [];
  const record = props.recorder;

  const view = (name: string, length: number) =>
    createTransientView(graph, `${id}-${name}`, 'uint32', Math.max(length, 1));
  const maxRows = (length: number) => Math.max(length, 1);

  // State and per-polygon tables.
  const counts = view('counts', MONOTONE_COUNT_WORDS);
  const routedCounts = view('routed-counts', polygonCount + 1);
  const routedOffsets = view('routed-offsets', polygonCount + 1);
  const polygonStats = view('polygon-stats', MONOTONE_STATS_STRIDE * polygonCount);
  const polygonPlan = view('polygon-plan', 4 * polygonCount);
  // Per routed vertex (raw rows): keep flags and scan, polygon/ring record, row to compact map.
  const keepFlags = view('keep-flags', vertexCount + 1);
  const keepOffsets = view('keep-offsets', vertexCount + 1);
  const jInfo = view('vertex-info', 4 * vertexCount);
  const rowToClean = view('row-to-clean', vertexCount);
  const ringInfo = view('ring-info', 4 * ringCount);
  const ringTop = view('ring-top', ringCount);
  // Per cleaned vertex.
  const vertexA = view('vertex-a', 4 * capacity);
  const vertexB = view('vertex-b', 4 * capacity);
  const vertexPosition = createTransientView(
    graph,
    `${id}-vertex-position`,
    'float32x2',
    maxRows(capacity)
  );
  const sortKeyHigh = view('sort-key-high', capacity);
  const sortKeyLow = view('sort-key-low', capacity);
  const identity = view('identity', capacity);
  const memberGroup = view('member-group', 3 * capacity);
  const memberIdentity = view('member-identity', 3 * capacity);
  const vertexDiagonal = view('vertex-diagonal', 4 * capacity);
  const nextHalfEdge = view('next-half-edge', 5 * capacity);
  const jumpA = view('jump-a', 10 * capacity);
  const jumpB = view('jump-b', 10 * capacity);
  const repCounts = view('rep-counts', capacity + 1);
  const repOffsets = view('rep-offsets', capacity + 1);
  const pieceTable = view('piece-table', MONOTONE_PIECE_STRIDE * capacity);
  const pieceTriangles = view('piece-triangles', capacity + 1);
  const pieceTriangleOffsets = view('piece-triangle-offsets', capacity + 1);
  const pieceScratch = view('piece-scratch', 6 * capacity);

  const kernel = (
    kernelProps: WGSLKernelProps<Parameters>,
    gate?: {countRow: number; invocationsPerItem: number; workgroupSize: number}
  ) => {
    nodes.push(createWGSLKernelNode<Parameters>(graph, kernelProps));
    record?.({
      kind: 'kernel',
      id: kernelProps.id,
      props: kernelProps as unknown as WGSLKernelProps<unknown>,
      gate
    });
  };
  const simple = (
    name: string,
    variant: string,
    bindings: WGSLKernelBinding[],
    invocationCount: number,
    source: MonotoneKernelSource,
    extra: Partial<WGSLKernelProps<Parameters>> = {}
  ) =>
    kernel({
      id: `${id}-${name}`,
      operation,
      variant,
      bindings,
      invocationCount: Math.max(invocationCount, 1),
      declarations: source.declarations,
      body: source.body,
      ...extra
    });
  const scan = (
    name: string,
    input: GraphDataView<'uint32'>,
    scanOutput: GraphDataView<'uint32'>
  ) => {
    nodes.push(
      ...new GPUScan({
        id: `${id}-${name}`,
        input,
        output: scanOutput,
        mode: 'exclusive'
      }).getCommandNodes(graph)
    );
    record?.({kind: 'scan', id: `${id}-${name}`, input, output: scanOutput});
  };
  const gateFor = (
    name: string,
    countRow: number,
    maximumItemCount: number,
    workgroupSize: number,
    invocationsPerItem: number = 1
  ) => {
    const dispatch = createWGSLActiveCountDispatch<Parameters>(graph, {
      id: `${id}-${name}-dispatch`,
      operation,
      count: counts,
      countRow,
      maximumItemCount,
      workgroupSize,
      invocationsPerItem
    });
    nodes.push(dispatch.updateNode);
    return {
      condition: dispatch.condition,
      workgroupSize,
      gate: {countRow, invocationsPerItem, workgroupSize}
    };
  };
  const gated = (
    gate: ReturnType<typeof gateFor>,
    name: string,
    variant: string,
    bindings: WGSLKernelBinding[],
    invocationCount: number,
    source: MonotoneKernelSource
  ) =>
    kernel(
      {
        id: `${id}-${name}`,
        operation,
        variant,
        bindings,
        invocationCount: Math.max(invocationCount, 1),
        workgroupSize: gate.workgroupSize,
        condition: {...gate.condition, id: `${id}-${name}-gate`},
        declarations: source.declarations,
        body: source.body
      },
      gate.gate
    );

  const read = (
    name: string,
    bindingView: GraphDataView,
    type: WGSLKernelBinding['type'] = 'u32'
  ) => ({name, view: bindingView, type, access: 'read'}) as WGSLKernelBinding;
  const write = (
    name: string,
    bindingView: GraphDataView,
    type: WGSLKernelBinding['type'] = 'u32'
  ) => ({name, view: bindingView, type, access: 'read_write'}) as WGSLKernelBinding;
  const atomic = (name: string, bindingView: GraphDataView) =>
    write(name, bindingView, 'atomic<u32>');

  // ---- Routing: counts per polygon, scan, overflow, published total. ----
  simple(
    'route-counts',
    'monotone-route-counts',
    [
      read('polygonOffsets', polygonOffsets),
      read('ringOffsets', ringOffsets),
      read('routing', routing),
      write('routedCounts', routedCounts),
      write('polygonStats', polygonStats),
      write('counts', counts)
    ],
    polygonCount + 1,
    getRouteCountsSource(sizes)
  );
  scan('route-scan', routedCounts, routedOffsets);
  simple(
    'route-overflow',
    'monotone-route-overflow',
    [
      write('routing', routing),
      write('routedCounts', routedCounts),
      read('routedOffsets', routedOffsets),
      write('counts', counts)
    ],
    polygonCount,
    getRouteOverflowSource(sizes)
  );
  const routedGate = gateFor('routed', MONOTONE_COUNT.routed, vertexCount, WORKGROUP_SIZE);

  // ---- Shared edge-slab index of the routed polygons (unslabbed when not routed). ----
  const slabNodes = getPolygonEdgeSlabNodes(graph, {
    id: `${id}-slabs`,
    operation,
    owner: 'polygon',
    positions,
    polygonOffsets: props.polygonOffsets,
    ringOffsets: props.ringOffsets,
    gate: {counts: routedCounts, minimum: 1}
  });
  nodes.push(...slabNodes.nodes);
  record?.({
    kind: 'slabs',
    id: `${id}-slabs`,
    params: slabNodes.slabs.params,
    offsets: slabNodes.slabs.slabOffsets,
    entries: slabNodes.slabs.slabEntries
  });

  // ---- Clean: keep flags, ring records, compaction. ----
  gated(
    routedGate,
    'index',
    'monotone-index',
    [
      read('polygonOffsets', polygonOffsets),
      read('ringOffsets', ringOffsets),
      read('positions', positions, 'f32'),
      read('routing', routing),
      read('routedOffsets', routedOffsets),
      write('keepFlags', keepFlags),
      write('jInfo', jInfo),
      atomic('polygonStats', polygonStats)
    ],
    vertexCount,
    getIndexSource(sizes)
  );
  scan('keep-scan', keepFlags, keepOffsets);
  gated(
    routedGate,
    'ring',
    'monotone-ring',
    [
      read('jInfo', jInfo),
      read('ringOffsets', ringOffsets),
      read('routedOffsets', routedOffsets),
      read('keepOffsets', keepOffsets),
      write('ringInfo', ringInfo),
      atomic('ringTop', ringTop),
      atomic('polygonStats', polygonStats),
      atomic('counts', counts)
    ],
    vertexCount,
    getRingSource(sizes)
  );
  simple(
    'counts',
    'monotone-counts',
    [write('counts', counts), read('keepOffsets', keepOffsets)],
    1,
    getCountsSource(sizes)
  );
  const cleanGate = gateFor('clean', MONOTONE_COUNT.clean, capacity, WORKGROUP_SIZE);
  const memberGate = gateFor('members', MONOTONE_COUNT.clean, capacity, WORKGROUP_SIZE, 3);
  gated(
    routedGate,
    'compact',
    'monotone-compact',
    [
      read('jInfo', jInfo),
      read('keepFlags', keepFlags),
      read('keepOffsets', keepOffsets),
      read('ringInfo', ringInfo),
      read('positions', positions, 'f32'),
      write('vertexA', vertexA),
      write('vertexPosition', vertexPosition, 'f32'),
      write('rowToClean', rowToClean)
    ],
    vertexCount,
    getCompactSource(sizes)
  );

  // ---- Sweep order: sort keys and a stable lexicographic sort (y descending, x ascending). ----
  simple(
    'keys',
    'monotone-keys',
    [
      read('counts', counts),
      write('vertexA', vertexA),
      read('vertexPosition', vertexPosition, 'f32'),
      write('sortKeyHigh', sortKeyHigh),
      write('sortKeyLow', sortKeyLow),
      write('identity', identity)
    ],
    capacity,
    getKeysSource(sizes)
  );
  const sweepSort = createStableLexicographicIndexSortNodes<Parameters>(graph, {
    id: `${id}-sweep-sort`,
    operation,
    indices: identity,
    keys: [{view: sortKeyHigh}, {view: sortKeyLow}]
  });
  nodes.push(...sweepSort.nodes);
  const sortedVertices = sweepSort.sortedIndices;
  record?.({
    kind: 'sort',
    id: `${id}-sweep-sort`,
    keys: [sortKeyHigh, sortKeyLow],
    indices: identity,
    sortedIndices: sortedVertices
  });

  // ---- Rank, orientation, classification. ----
  gated(
    cleanGate,
    'rank',
    'monotone-rank',
    [
      read('sortedIndices', sortedVertices),
      read('vertexA', vertexA),
      write('vertexB', vertexB),
      atomic('ringTop', ringTop)
    ],
    capacity,
    getRankSource(sizes)
  );
  gated(
    cleanGate,
    'flip',
    'monotone-flip',
    [
      read('vertexA', vertexA),
      read('vertexB', vertexB),
      write('ringInfo', ringInfo),
      atomic('ringTop', ringTop),
      read('vertexPosition', vertexPosition, 'f32'),
      read('polygonOffsets', polygonOffsets),
      atomic('polygonStats', polygonStats)
    ],
    capacity,
    withOrientation(getFlipSource(sizes))
  );
  gated(
    cleanGate,
    'link',
    'monotone-link',
    [
      write('vertexA', vertexA),
      write('vertexB', vertexB),
      read('ringInfo', ringInfo),
      read('vertexPosition', vertexPosition, 'f32'),
      write('vertexDiagonal', vertexDiagonal),
      write('nextHalfEdge', nextHalfEdge),
      atomic('polygonStats', polygonStats)
    ],
    capacity,
    withOrientation(getLinkSource(sizes))
  );

  // ---- Left edges through the shared edge-slab index. ----
  const slabBindings = getPolygonEdgeSlabBindings(slabNodes.slabs);
  const leftEdgeDeclarations = `${EXACT_ORIENTATION_WGSL}${getMonotoneConstantsWGSL(sizes)}${MONOTONE_VERTEX_A_WGSL}${MONOTONE_VERTEX_B_WGSL}${MONOTONE_POSITION_WGSL}${MONOTONE_ORIENT_WGSL}${getPolygonEdgeSlabQueryWGSL(slabNodes.slabs)}${MONOTONE_LEFT_EDGE_WGSL}`;
  gated(
    cleanGate,
    'left-edge',
    'monotone-left-edge',
    [
      ...slabBindings,
      read('rowToClean', rowToClean),
      read('vertexA', vertexA),
      write('vertexB', vertexB),
      read('vertexPosition', vertexPosition, 'f32')
    ],
    capacity,
    {
      declarations: leftEdgeDeclarations,
      body: getLeftEdgeBody(candidate =>
        getPolygonEdgeSlabLoopWGSL({
          owner: 'polygon',
          lowY: 'point.y',
          highY: 'point.y',
          body: candidate
        })
      )
    }
  );

  // ---- Helper diagonals: members sorted by (edge, rank), consecutive pairs. ----
  simple(
    'members',
    'monotone-members',
    [
      read('counts', counts),
      read('sortedIndices', sortedVertices),
      read('vertexA', vertexA),
      read('vertexB', vertexB),
      write('memberGroup', memberGroup),
      write('memberIdentity', memberIdentity),
      atomic('polygonStats', polygonStats)
    ],
    capacity,
    getMembersSource(sizes)
  );
  const memberSort = createStableLexicographicIndexSortNodes<Parameters>(graph, {
    id: `${id}-member-sort`,
    operation,
    indices: memberIdentity,
    keys: [{view: memberGroup, keyBits: getBitCount(capacity)}]
  });
  nodes.push(...memberSort.nodes);
  record?.({
    kind: 'sort',
    id: `${id}-member-sort`,
    keys: [memberGroup],
    indices: memberIdentity,
    sortedIndices: memberSort.sortedIndices
  });
  gated(
    memberGate,
    'diagonals',
    'monotone-diagonals',
    [
      read('sortedMembers', memberSort.sortedIndices),
      read('memberGroup', memberGroup),
      read('sortedIndices', sortedVertices),
      read('vertexA', vertexA),
      write('vertexDiagonal', vertexDiagonal),
      atomic('polygonStats', polygonStats)
    ],
    3 * capacity,
    getDiagonalsSource(sizes)
  );

  // ---- Half-edge next pointers and piece labels by pointer jumping. ----
  gated(
    cleanGate,
    'next',
    'monotone-next',
    [
      read('vertexA', vertexA),
      read('vertexB', vertexB),
      read('vertexPosition', vertexPosition, 'f32'),
      read('vertexDiagonal', vertexDiagonal),
      write('nextHalfEdge', nextHalfEdge),
      atomic('polygonStats', polygonStats)
    ],
    capacity,
    withOrientation(getNextSource(sizes))
  );
  gated(
    cleanGate,
    'jump-init',
    'monotone-jump-init',
    [
      read('vertexA', vertexA),
      read('vertexDiagonal', vertexDiagonal),
      read('nextHalfEdge', nextHalfEdge),
      write('jumpA', jumpA)
    ],
    capacity,
    getJumpInitSource(sizes)
  );
  // Rounds run in pairs (A to B, then B to A), so the labels always end in `jumpA`; a pair that
  // the GPU state disarms launches nothing. The number of unrolled pairs is a topology constant.
  const jumpPairCount = props.jumpPairCount ?? getMonotoneJumpPairCount(capacity);
  const jumpRoundSource = getJumpRoundSource(sizes);
  const jumpStateSource = getJumpStateSource(sizes);
  for (let pair = 0; pair < jumpPairCount; pair++) {
    simple(
      `jump-${pair}-state`,
      'monotone-jump-state',
      [write('counts', counts)],
      1,
      jumpStateSource
    );
    const jumpGate = gateFor(`jump-${pair}`, MONOTONE_COUNT.jumpActive, capacity, WORKGROUP_SIZE);
    gated(
      jumpGate,
      `jump-${pair}-forward`,
      'monotone-jump',
      [read('jumpSource', jumpA), write('jumpTarget', jumpB)],
      capacity,
      jumpRoundSource
    );
    gated(
      jumpGate,
      `jump-${pair}-backward`,
      'monotone-jump',
      [read('jumpSource', jumpB), write('jumpTarget', jumpA)],
      capacity,
      jumpRoundSource
    );
  }

  // ---- Pieces: representatives, scan, heads, measure. ----
  gated(
    cleanGate,
    'rep-count',
    'monotone-rep-count',
    [read('jumpA', jumpA), write('repCounts', repCounts)],
    capacity,
    getRepresentativeCountSource(sizes)
  );
  scan('rep-scan', repCounts, repOffsets);
  gated(
    cleanGate,
    'pieces',
    'monotone-pieces',
    [read('jumpA', jumpA), read('repOffsets', repOffsets), write('pieceTable', pieceTable)],
    capacity,
    getPiecesSource(sizes)
  );
  simple(
    'piece-count',
    'monotone-piece-count',
    [write('counts', counts), read('repOffsets', repOffsets)],
    1,
    getPieceCountSource(sizes)
  );
  const pieceGate = gateFor('pieces', MONOTONE_COUNT.pieces, capacity, PIECE_WORKGROUP_SIZE);
  gated(
    pieceGate,
    'measure',
    'monotone-measure',
    [
      read('counts', counts),
      write('pieceTable', pieceTable),
      read('nextHalfEdge', nextHalfEdge),
      read('vertexA', vertexA),
      atomic('polygonStats', polygonStats),
      write('pieceTriangles', pieceTriangles)
    ],
    capacity,
    getMeasureSource(sizes)
  );
  scan('piece-scan', pieceTriangles, pieceTriangleOffsets);
  simple(
    'polygon-plan',
    'monotone-polygon-plan',
    [
      read('polygonOffsets', polygonOffsets),
      read('ringOffsets', ringOffsets),
      read('routing', routing),
      read('routedOffsets', routedOffsets),
      read('keepOffsets', keepOffsets),
      read('repOffsets', repOffsets),
      read('pieceTriangleOffsets', pieceTriangleOffsets),
      write('polygonPlan', polygonPlan)
    ],
    polygonCount,
    getPolygonPlanSource(sizes)
  );
  gated(
    pieceGate,
    'piece-plan',
    'monotone-piece-plan',
    [
      read('counts', counts),
      write('pieceTable', pieceTable),
      read('pieceTriangleOffsets', pieceTriangleOffsets),
      read('polygonPlan', polygonPlan),
      read('vertexA', vertexA),
      atomic('polygonStats', polygonStats)
    ],
    capacity,
    getPiecePlanSource(sizes)
  );
  gated(
    pieceGate,
    'triangulate',
    'monotone-triangulate',
    [
      read('pieceTable', pieceTable),
      read('nextHalfEdge', nextHalfEdge),
      read('vertexA', vertexA),
      read('vertexB', vertexB),
      read('vertexPosition', vertexPosition, 'f32'),
      write('pieceScratch', pieceScratch),
      write('indices', output.indices),
      atomic('polygonStats', polygonStats)
    ],
    capacity,
    withOrientation(getTriangulateSource(sizes))
  );

  // ---- Validity and padding. ----
  simple(
    'final',
    'monotone-final',
    [
      write('routing', routing),
      write('polygonStats', polygonStats),
      write('valid', output.valid),
      ...(output.triangleCount ? [write('triangleCount', output.triangleCount)] : [])
    ],
    polygonCount,
    getFinalSource(sizes, Boolean(output.triangleCount))
  );
  gated(
    routedGate,
    'pad',
    'monotone-pad',
    [
      read('jInfo', jInfo),
      read('polygonOffsets', polygonOffsets),
      read('ringOffsets', ringOffsets),
      read('routedOffsets', routedOffsets),
      read('polygonStats', polygonStats),
      write('indices', output.indices)
    ],
    vertexCount,
    getPadSource(sizes)
  );
  return nodes;
}

/** Prepends the exact orientation helper that the kernel's `orientOf` wrapper calls. */
function withOrientation(source: MonotoneKernelSource): MonotoneKernelSource {
  return {declarations: `${EXACT_ORIENTATION_WGSL}${source.declarations}`, body: source.body};
}
