// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUBVH,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {createWGSLKernelNode, type WGSLKernelBinding} from '../../utils/wgsl-kernel-nodes';
import type {GPUFloat32Positions} from '../../utils/gpu-contributor-types';
import {getGraphViewChunks} from '../../utils/gpu-contributor-utils';
import {
  NEAREST_COMMON_WGSL,
  NEAREST_HEADER_WORDS,
  NEAREST_NODE_WORDS,
  getNearestPairWGSL,
  getNearestSideWGSL,
  getNearestTraversalWGSL
} from './nearest-wgsl';
import type {
  GPUNearestFeatureGeometry,
  GPUNearestQueryGeometry,
  NearestSide
} from './nearest-types';
import type {GPUSpatialJoinPrepared} from './spatial-join-prepared';
import {
  createSpatialJoinBoundsNode,
  getChunkNodeId,
  getSortedFeatureBVHNodes
} from './spatial-join-passes';

/** Query side of the k-nearest mode: chunked points or one geometry set. @internal */
export type NearestQuerySource =
  | {kind: 'points'; points: GPUFloat32Positions}
  | {kind: 'geometry'; geometry: GPUNearestQueryGeometry};

/** The BVH facts the nearest kernels read; satisfied by `GPUBVH` and by prepared storage. @internal */
export type NearestBVH = Pick<
  GPUBVH,
  | 'nodeMinima'
  | 'nodeMaxima'
  | 'leafIds'
  | 'overflow'
  | 'internalNodeCount'
  | 'nodeCount'
  | 'levelCount'
>;

/** Inputs of {@link getNearestNeighborNodes}. @internal */
export type NearestNeighborNodeProps = {
  /** Prepared right-hand side whose BVH and ring ranges replace the per-encoding build. */
  prepared?: GPUSpatialJoinPrepared;
  id: string;
  operation: string;
  queries: NearestQuerySource;
  queryCount: number;
  features: GPUNearestFeatureGeometry;
  featureCount: number;
  k: number;
  capacity: number;
  leafCapacity: number;
  spatialSort: boolean;
  maxDistance?: GraphDataView<'float32'>;
  featureIds?: GraphDataView<'uint32'>;
  neighborIds: GraphDataView<'uint32'>;
  neighborCounts: GraphDataView<'uint32'>;
  neighborDistances?: GraphDataView<'float32'>;
  neighborFootPoints?: GraphDataView<'float32x2'>;
  neighborSegmentIndices?: GraphDataView<'uint32'>;
  overflow: GraphDataView<'uint32'>;
};

/** Returns the number of feature rows of a nearest geometry. @internal */
export function getNearestGeometryCount(geometry: GPUNearestFeatureGeometry): number {
  switch (geometry.kind) {
    case 'points':
      return geometry.positions.length;
    case 'segments':
      return geometry.starts.length;
    case 'lines':
      return geometry.lineOffsets.length - 1;
    case 'polygons':
      return geometry.featureOffsets.length - 1;
  }
}

/** Returns every input view of a nearest geometry. @internal */
export function getNearestGeometryViews(geometry: GPUNearestFeatureGeometry): GraphDataView[] {
  switch (geometry.kind) {
    case 'points':
      return [geometry.positions];
    case 'segments':
      return [geometry.starts, geometry.ends];
    case 'lines':
      return [geometry.positions, geometry.lineOffsets];
    case 'polygons':
      return [
        geometry.positions,
        geometry.featureOffsets,
        geometry.polygonOffsets,
        geometry.ringOffsets
      ];
  }
}

function getSideVertexCount(geometry: GPUNearestFeatureGeometry): number {
  return geometry.kind === 'segments' ? geometry.starts.length * 2 : geometry.positions.length;
}

/** Storage bindings read by the accessors of one geometry side. */
function getSideBindings(
  prefix: 'q' | 'r',
  geometry: GPUNearestFeatureGeometry,
  featureRings: GraphDataView<'uint32x2'> | undefined
): WGSLKernelBinding[] {
  const read = (name: string, view: GraphDataView, type: 'f32' | 'u32'): WGSLKernelBinding => ({
    name: `${prefix}${name}`,
    view,
    type,
    access: 'read'
  });
  switch (geometry.kind) {
    case 'points':
      return [read('Positions', geometry.positions, 'f32')];
    case 'segments':
      return [read('Starts', geometry.starts, 'f32'), read('Ends', geometry.ends, 'f32')];
    case 'lines':
      return [
        read('Positions', geometry.positions, 'f32'),
        read('LineOffsets', geometry.lineOffsets, 'u32')
      ];
    case 'polygons':
      return [
        read('Positions', geometry.positions, 'f32'),
        read('FeatureRings', featureRings!, 'u32'),
        read('RingOffsets', geometry.ringOffsets, 'u32')
      ];
  }
}

/** Writes `[ringStart, ringEnd)` of every polygon feature, clamped against malformed offsets. */
function createPolygonRingsNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  operation: string,
  geometry: Extract<GPUNearestFeatureGeometry, {kind: 'polygons'}>,
  featureRings: GraphDataView<'uint32x2'>
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id,
    operation,
    variant: 'polygon-rings',
    bindings: [
      {name: 'featureOffsets', view: geometry.featureOffsets, type: 'u32', access: 'read'},
      {name: 'polygonOffsets', view: geometry.polygonOffsets, type: 'u32', access: 'read'},
      {name: 'featureRings', view: featureRings, type: 'u32', access: 'read_write'}
    ],
    invocationCount: featureRings.length,
    declarations: `const POLYGON_COUNT: u32 = ${geometry.polygonOffsets.length - 1}u;
const RING_COUNT: u32 = ${geometry.ringOffsets.length - 1}u;`,
    body: `let polygonStart = featureOffsets[featureOffsetsOffset + index];
  let polygonEnd = featureOffsets[featureOffsetsOffset + index + 1u];
  var ringStart = 0u;
  var ringEnd = 0u;
  if (polygonStart <= polygonEnd && polygonEnd <= POLYGON_COUNT) {
    let first = polygonOffsets[polygonOffsetsOffset + polygonStart];
    let last = polygonOffsets[polygonOffsetsOffset + polygonEnd];
    if (first <= last && last <= RING_COUNT) { ringStart = first; ringEnd = last; }
  }
  featureRings[featureRingsOffset + index * 2u] = ringStart;
  featureRings[featureRingsOffset + index * 2u + 1u] = ringEnd;`
  });
}

/** Writes per-feature bounds for every feature kind, including linestrings. */
function createBoundsNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  operation: string,
  features: GPUNearestFeatureGeometry,
  featureCount: number,
  minima: GraphDataView<'float32x2'>,
  maxima: GraphDataView<'float32x2'>
): GPUCommandNode<Parameters> {
  if (features.kind === 'lines') {
    return createWGSLKernelNode<Parameters>(graph, {
      id,
      operation,
      variant: 'bounds-lines',
      bindings: [
        {name: 'positions', view: features.positions, type: 'f32', access: 'read'},
        {name: 'lineOffsets', view: features.lineOffsets, type: 'u32', access: 'read'},
        {name: 'featureMinima', view: minima, type: 'f32', access: 'read_write'},
        {name: 'featureMaxima', view: maxima, type: 'f32', access: 'read_write'}
      ],
      invocationCount: featureCount,
      declarations: `${NEAREST_COMMON_WGSL}
const VERTEX_COUNT: u32 = ${features.positions.length}u;`,
      body: `var minimum = vec2f(FLOAT32_MAXIMUM);
  var maximum = vec2f(-FLOAT32_MAXIMUM);
  let vertexStart = lineOffsets[lineOffsetsOffset + index];
  let vertexEnd = min(lineOffsets[lineOffsetsOffset + index + 1u], VERTEX_COUNT);
  for (var vertex = vertexStart; vertex < vertexEnd; vertex++) {
    let position = vec2f(positions[positionsOffset + vertex * 2u], positions[positionsOffset + vertex * 2u + 1u]);
    if (isFiniteValue(position.x) && isFiniteValue(position.y)) {
      minimum = min(minimum, position);
      maximum = max(maximum, position);
    }
  }
  featureMinima[featureMinimaOffset + index * 2u] = minimum.x;
  featureMinima[featureMinimaOffset + index * 2u + 1u] = minimum.y;
  featureMaxima[featureMaximaOffset + index * 2u] = maximum.x;
  featureMaxima[featureMaximaOffset + index * 2u + 1u] = maximum.y;`
    });
  }
  return createSpatialJoinBoundsNode<Parameters>(graph, {
    id,
    operation,
    featureCount,
    source:
      features.kind === 'polygons'
        ? {
            kind: 'polygons',
            polygonPositions: features.positions,
            featureOffsets: features.featureOffsets,
            polygonOffsets: features.polygonOffsets,
            ringOffsets: features.ringOffsets
          }
        : features,
    minima,
    maxima
  });
}

/**
 * Returns the feature BVH and the nodes that build it. With a prepared handle there are no
 * nodes: the handle's own (conditional) build nodes fill its persistent storage.
 *
 * @internal
 */
export function getNearestBVHNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    features: GPUNearestFeatureGeometry;
    featureCount: number;
    leafCapacity: number;
    spatialSort: boolean;
    prepared?: GPUSpatialJoinPrepared;
  }
): {bvh: NearestBVH; nodes: readonly GPUCommandNode<Parameters>[]} {
  const {id, operation, features, featureCount, prepared} = props;
  if (prepared) {
    if (!prepared.isDeclaredIn(graph)) {
      throw new Error(`${id} prepared handle must be added to the graph before the join`);
    }
    const {storage} = prepared;
    return {
      bvh: {
        nodeMinima: storage.nodeMinima,
        nodeMaxima: storage.nodeMaxima,
        leafIds: storage.leafIds,
        overflow: storage.overflow,
        internalNodeCount: prepared.leafCapacity - 1,
        nodeCount: 2 * prepared.leafCapacity - 1,
        levelCount: Math.log2(prepared.leafCapacity) + 1
      },
      nodes: []
    };
  }
  const minima = createTransientView(graph, `${id}-feature-minima`, 'float32x2', featureCount);
  const maxima = createTransientView(graph, `${id}-feature-maxima`, 'float32x2', featureCount);
  const nodes: GPUCommandNode<Parameters>[] = [];
  if (featureCount > 0) {
    nodes.push(
      createBoundsNode(graph, `${id}-bounds`, operation, features, featureCount, minima, maxima)
    );
  }
  const built = getSortedFeatureBVHNodes(
    graph,
    id,
    operation,
    minima,
    maxima,
    props.leafCapacity,
    props.spatialSort
  );
  nodes.push(...built.nodes);
  return {bvh: built.bvh, nodes};
}

/**
 * Builds the best-first k-nearest pipeline: bounds, BVH, packed node table, polygon ring ranges,
 * header clear, one traversal per query chunk, optional foot-point pass, and column writes.
 *
 * Every pass is a fixed-size dispatch; nothing is read back.
 *
 * @internal
 */
export function getNearestNeighborNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: NearestNeighborNodeProps
): {nodes: GPUCommandNode<Parameters>[]; bvh: NearestBVH} {
  const {id, operation, features, featureCount, queryCount, k, capacity} = props;
  const nodes: GPUCommandNode<Parameters>[] = [];

  const {bvh, nodes: bvhNodes} = getNearestBVHNodes(graph, {
    id,
    operation,
    features,
    featureCount,
    leafCapacity: props.leafCapacity,
    spatialSort: props.spatialSort,
    prepared: props.prepared
  });
  nodes.push(...bvhNodes);

  // One packed node table: min.xy, max.xy as f32 bits, and the feature row of leaves.
  const nodeData = createTransientView(
    graph,
    `${id}-node-data`,
    'uint32',
    bvh.nodeCount * NEAREST_NODE_WORDS
  );
  nodes.push(
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-node-data`,
      operation,
      variant: 'node-data',
      bindings: [
        {name: 'nodeMinima', view: bvh.nodeMinima, type: 'f32', access: 'read'},
        {name: 'nodeMaxima', view: bvh.nodeMaxima, type: 'f32', access: 'read'},
        {name: 'leafIds', view: bvh.leafIds, type: 'u32', access: 'read'},
        {name: 'nodeData', view: nodeData, type: 'u32', access: 'read_write'}
      ],
      invocationCount: bvh.nodeCount,
      declarations: `const INTERNAL_NODE_COUNT: u32 = ${bvh.internalNodeCount}u;
const NO_FEATURE: u32 = 0xffffffffu;`,
      body: `let base = nodeDataOffset + index * ${NEAREST_NODE_WORDS}u;
  nodeData[base] = bitcast<u32>(nodeMinima[nodeMinimaOffset + index * 2u]);
  nodeData[base + 1u] = bitcast<u32>(nodeMinima[nodeMinimaOffset + index * 2u + 1u]);
  nodeData[base + 2u] = bitcast<u32>(nodeMaxima[nodeMaximaOffset + index * 2u]);
  nodeData[base + 3u] = bitcast<u32>(nodeMaxima[nodeMaximaOffset + index * 2u + 1u]);
  nodeData[base + 4u] = select(NO_FEATURE, leafIds[leafIdsOffset + index - INTERNAL_NODE_COUNT], index >= INTERNAL_NODE_COUNT);`
    })
  );

  // Polygon ring ranges for the feature side and for a polygon query side.
  let featureRings: GraphDataView<'uint32x2'> | undefined;
  if (features.kind === 'polygons' && props.prepared) {
    featureRings = props.prepared.storage.featureRings;
  } else if (features.kind === 'polygons') {
    featureRings = createTransientView(graph, `${id}-feature-rings`, 'uint32x2', featureCount);
    if (featureCount > 0) {
      nodes.push(
        createPolygonRingsNode(graph, `${id}-feature-rings`, operation, features, featureRings)
      );
    }
  }
  const queryGeometry = props.queries.kind === 'geometry' ? props.queries.geometry : undefined;
  let queryRings: GraphDataView<'uint32x2'> | undefined;
  if (queryGeometry?.kind === 'polygons') {
    queryRings = createTransientView(graph, `${id}-query-rings`, 'uint32x2', queryCount);
    if (queryCount > 0) {
      nodes.push(
        createPolygonRingsNode(graph, `${id}-query-rings`, operation, queryGeometry, queryRings)
      );
    }
  }

  // Result buffer: per query [count, flags, bound bits, (row, distance-squared bits) * capacity].
  const stride = NEAREST_HEADER_WORDS + 2 * capacity;
  const topk = createTransientView(graph, `${id}-topk`, 'uint32', Math.max(queryCount, 1) * stride);
  const clearBindings: WGSLKernelBinding[] = [
    {name: 'topk', view: topk, type: 'u32', access: 'read_write'},
    {name: 'overflow', view: props.overflow, type: 'u32', access: 'read_write'},
    {name: 'bvhOverflow', view: bvh.overflow, type: 'u32', access: 'read'}
  ];
  if (props.maxDistance) {
    clearBindings.push({name: 'maxDistance', view: props.maxDistance, type: 'f32', access: 'read'});
  }
  nodes.push(
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-clear`,
      operation,
      variant: 'clear',
      bindings: clearBindings,
      invocationCount: Math.max(queryCount, 1),
      declarations: `const FLOAT32_MAXIMUM: f32 = 3.402823466e+38;
const QUERY_COUNT: u32 = ${queryCount}u;
const STRIDE: u32 = ${stride}u;`,
      body: `if (index == 0u) { overflow[overflowOffset] = bvhOverflow[bvhOverflowOffset]; }
  if (index < QUERY_COUNT) {
    // A NaN, negative or infinite maxDistance sets a negative bound that matches nothing.
    ${
      props.maxDistance
        ? `let limit = maxDistance[maxDistanceOffset];
    var bound = -1.0;
    if (limit >= 0.0 && limit <= FLOAT32_MAXIMUM) { bound = min(limit * limit, FLOAT32_MAXIMUM); }`
        : 'let bound = FLOAT32_MAXIMUM;'
    }
    let header = topkOffset + index * STRIDE;
    topk[header] = 0u;
    topk[header + 1u] = 0u;
    topk[header + 2u] = bitcast<u32>(bound);
  }`
    })
  );

  // Query dispatches: one per nonempty point chunk, or one for the whole geometry set.
  type QueryDispatch = {
    chunkId: string;
    side: NearestSide;
    bindings: WGSLKernelBinding[];
    count: number;
    firstRow: number;
    kind: NearestSide['kind'];
  };
  const dispatches: QueryDispatch[] = [];
  if (props.queries.kind === 'points') {
    let firstRow = 0;
    for (const [chunkIndex, chunk] of getGraphViewChunks(props.queries.points).entries()) {
      if (chunk.length > 0) {
        const geometry = {kind: 'points', positions: chunk} as const;
        dispatches.push({
          chunkId: getChunkNodeId(id, props.queries.points, chunkIndex),
          side: {prefix: 'q', kind: 'points', vertexCount: chunk.length},
          bindings: getSideBindings('q', geometry, undefined),
          count: chunk.length,
          firstRow,
          kind: 'points'
        });
      }
      firstRow += chunk.length;
    }
  } else if (queryCount > 0) {
    dispatches.push({
      chunkId: id,
      side: {
        prefix: 'q',
        kind: props.queries.geometry.kind,
        vertexCount: getSideVertexCount(props.queries.geometry)
      },
      bindings: getSideBindings('q', props.queries.geometry, queryRings),
      count: queryCount,
      firstRow: 0,
      kind: props.queries.geometry.kind
    });
  }
  const featureSide: NearestSide = {
    prefix: 'r',
    kind: features.kind,
    vertexCount: getSideVertexCount(features)
  };
  const featureBindings = getSideBindings('r', features, featureRings);
  const nodeDataBinding: WGSLKernelBinding = {
    name: 'nodeData',
    view: nodeData,
    type: 'u32',
    access: 'read'
  };
  const topkBinding = (access: 'read' | 'read_write'): WGSLKernelBinding => ({
    name: 'topk',
    view: topk,
    type: 'u32',
    access
  });
  const stackSize = bvh.levelCount + 2;
  const needsFootPass = Boolean(props.neighborFootPoints || props.neighborSegmentIndices);
  const extras = needsFootPass
    ? createTransientView(graph, `${id}-extras`, 'uint32', Math.max(queryCount, 1) * capacity * 3)
    : undefined;

  for (const dispatch of dispatches) {
    const traversal = getNearestTraversalWGSL({
      k,
      capacity,
      featureCount,
      internalNodeCount: bvh.internalNodeCount,
      stackSize,
      chunkFirstRow: dispatch.firstRow,
      queryKind: dispatch.kind
    });
    const sideWGSL = `${NEAREST_COMMON_WGSL}${getNearestSideWGSL(dispatch.side)}${getNearestSideWGSL(featureSide)}${getNearestPairWGSL(dispatch.side, featureSide)}`;
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${dispatch.chunkId}-traverse`,
        operation,
        variant: `traverse-${dispatch.kind}-${features.kind}`,
        bindings: [
          ...dispatch.bindings,
          nodeDataBinding,
          ...featureBindings,
          topkBinding('read_write')
        ],
        invocationCount: dispatch.count,
        declarations: `${sideWGSL}${traversal.declarations}`,
        body: traversal.body
      })
    );
    if (extras) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${dispatch.chunkId}-foot-points`,
          operation,
          variant: `foot-${dispatch.kind}-${features.kind}`,
          bindings: [
            ...dispatch.bindings,
            ...featureBindings,
            topkBinding('read'),
            {name: 'extras', view: extras, type: 'u32', access: 'read_write'}
          ],
          invocationCount: dispatch.count * capacity,
          declarations: `${sideWGSL}
const CAPACITY: u32 = ${capacity}u;
const STRIDE: u32 = ${stride}u;
const CHUNK_FIRST_ROW: u32 = ${dispatch.firstRow}u;`,
          body: `let localQuery = index / CAPACITY;
  let slot = index % CAPACITY;
  let row = CHUNK_FIRST_ROW + localQuery;
  let header = topkOffset + row * STRIDE;
  if (slot >= topk[header]) { return; }
  let result = pairResult(localQuery, topk[header + 3u + slot * 2u]);
  let output = extrasOffset + (row * CAPACITY + slot) * 3u;
  extras[output] = bitcast<u32>(result.foot.x);
  extras[output + 1u] = bitcast<u32>(result.foot.y);
  extras[output + 2u] = result.segment;`
        })
      );
    }
  }

  // Column writes: IDs, distances, counts, overflow flag.
  if (queryCount > 0) {
    const columnBindings: WGSLKernelBinding[] = [
      topkBinding('read'),
      {name: 'neighborIds', view: props.neighborIds, type: 'u32', access: 'read_write'},
      {name: 'neighborCounts', view: props.neighborCounts, type: 'u32', access: 'read_write'},
      {name: 'overflow', view: props.overflow, type: 'atomic<u32>', access: 'read_write'}
    ];
    if (props.featureIds) {
      columnBindings.push({
        name: 'featureIds',
        view: props.featureIds,
        type: 'u32',
        access: 'read'
      });
    }
    if (props.neighborDistances) {
      columnBindings.push({
        name: 'neighborDistances',
        view: props.neighborDistances,
        type: 'f32',
        access: 'read_write'
      });
    }
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-columns`,
        operation,
        variant: 'columns',
        bindings: columnBindings,
        invocationCount: queryCount * capacity,
        declarations: `const CAPACITY: u32 = ${capacity}u;
const STRIDE: u32 = ${stride}u;
const NO_FEATURE: u32 = 0xffffffffu;`,
        body: `let query = index / CAPACITY;
  let slot = index % CAPACITY;
  let header = topkOffset + query * STRIDE;
  let count = topk[header];
  if (slot == 0u) {
    neighborCounts[neighborCountsOffset + query] = count;
    if (topk[header + 1u] != 0u) { atomicOr(&overflow[overflowOffset], 1u); }
  }
  var featureId = NO_FEATURE;
  var distance = -1.0;
  if (slot < count) {
    featureId = topk[header + 3u + slot * 2u];
    ${props.featureIds ? 'featureId = featureIds[featureIdsOffset + featureId];' : ''}
    distance = sqrt(bitcast<f32>(topk[header + 4u + slot * 2u]));
  }
  neighborIds[neighborIdsOffset + index] = featureId;
  ${props.neighborDistances ? 'neighborDistances[neighborDistancesOffset + index] = distance;' : ''}`
      })
    );
    if (extras) {
      const bindings: WGSLKernelBinding[] = [
        topkBinding('read'),
        {name: 'extras', view: extras, type: 'u32', access: 'read'}
      ];
      if (props.neighborFootPoints) {
        bindings.push({
          name: 'footPoints',
          view: props.neighborFootPoints,
          type: 'u32',
          access: 'read_write'
        });
      }
      if (props.neighborSegmentIndices) {
        bindings.push({
          name: 'segmentIndices',
          view: props.neighborSegmentIndices,
          type: 'u32',
          access: 'read_write'
        });
      }
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-foot-columns`,
          operation,
          variant: 'foot-columns',
          bindings,
          invocationCount: queryCount * capacity,
          declarations: `const CAPACITY: u32 = ${capacity}u;
const STRIDE: u32 = ${stride}u;
const NO_SEGMENT: u32 = 0xffffffffu;`,
          body: `let query = index / CAPACITY;
  let slot = index % CAPACITY;
  let valid = slot < topk[topkOffset + query * STRIDE];
  let source = extrasOffset + index * 3u;
  ${
    props.neighborFootPoints
      ? `// Unused slots hold a quiet NaN so a missing neighbor is never mistaken for the origin.
  footPoints[footPointsOffset + index * 2u] = select(0x7fc00000u, extras[source], valid);
  footPoints[footPointsOffset + index * 2u + 1u] = select(0x7fc00000u, extras[source + 1u], valid);`
      : ''
  }
  ${props.neighborSegmentIndices ? 'segmentIndices[segmentIndicesOffset + index] = select(NO_SEGMENT, extras[source + 2u], valid);' : ''}`
        })
      );
    }
  }
  return {nodes, bvh};
}
