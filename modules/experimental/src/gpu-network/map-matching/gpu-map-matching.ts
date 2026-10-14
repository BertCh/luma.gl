// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUScan,
  validatePackedUint32View,
  validatePackedView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {
  createFillNode,
  createWGSLKernelNode,
  type WGSLKernelBinding
} from '../../utils/wgsl-kernel-nodes';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../utils/gpu-contributor-utils';

import type {GPUSpatialContext} from '../../gpu-spatial-analysis/contracts/spatial-context';
import {
  getNetworkGeodesicRadius,
  getNetworkMetricWGSL,
  resolveNetworkSpatialContext
} from '../network-spatial-context';

const OPERATION = 'GPUMapMatching';
/** Bits of `-1.0f`: an out-of-range node ID and a negative (invalid) edge length. */
const INVALID_EDGE_WORD = 0xbf800000;

/** Value written to `matchedEdges` for a point with no candidate edge or no feasible match. */
export const GPU_MAP_MATCHING_NONE = 0xffffffff;

/** Number of `float32` words in the per-frame parameter view of {@link GPUMapMatching}. */
export const GPU_MAP_MATCHING_PARAMETER_LENGTH = 8;

/**
 * Per-frame parameters of {@link GPUMapMatching}, in the units of the planar coordinates, or in
 * meters with a great-circle `spatialContext`.
 */
export type GPUMapMatchingParameters = {
  /** GPS noise standard deviation of the Gaussian emission model. Must be positive. */
  sigma: number;
  /** Scale of the exponential transition model, `exp(-|straight - route| / beta)`. Must be positive. */
  beta: number;
  /**
   * Candidate search radius. Edges farther than this from a point are not candidates. Values above
   * `2 * cellSize` are clamped to `2 * cellSize` (on longitude/latitude, to the meters of
   * `2 * cellSize` degrees of latitude).
   */
  searchRadius: number;
  /** A route may be at most `routeFactor * straightDistance + routeSlack` long. Defaults to 3. */
  routeFactor?: number;
  /** Additive route allowance, see `routeFactor`. Defaults to 50. */
  routeSlack?: number;
};

/** Encodes {@link GPUMapMatchingParameters} as the `float32` words read by {@link GPUMapMatching}. */
export function encodeGPUMapMatchingParameters(parameters: GPUMapMatchingParameters): Float32Array {
  const words = new Float32Array(GPU_MAP_MATCHING_PARAMETER_LENGTH);
  words[0] = parameters.sigma;
  words[1] = parameters.beta;
  words[2] = parameters.searchRadius;
  words[3] = parameters.routeFactor ?? 3;
  words[4] = parameters.routeSlack ?? 50;
  return words;
}

/** Caller-owned outputs of {@link GPUMapMatching}. */
export type GPUMapMatchingOutput = {
  /** CSR row of the matched edge per point, or {@link GPU_MAP_MATCHING_NONE}. */
  matchedEdges: GraphDataView<'uint32'>;
  /** Optional matched position along the edge from source (0) to target (1), or -1. */
  matchedFractions?: GraphDataView<'float32'>;
  /** Optional matched distance along the edge from its source, in network length units, or -1. */
  matchedOffsets?: GraphDataView<'float32'>;
  /** Optional distance from the point to its matched edge, or -1. */
  snapDistances?: GraphDataView<'float32'>;
  /** Optional snapped position, or the raw point when it is unmatched. */
  snappedPositions?: GraphDataView<'float32x2'>;
  /**
   * Per point: 0, or a break where a new matched sub-track starts. 1 is an ordinary break: no
   * feasible transition from any candidate of the previous point, or no candidate at all. 2 is a
   * budget break: no transition was found, but a route search into the point hit
   * `routeNodeBudget` and dropped nodes, so the missing route may be the budget's doing rather
   * than the data's (raise `routeNodeBudget` or sample more densely). First points of tracks are
   * not breaks.
   */
  breaks?: GraphDataView<'uint32'>;
  /** Optional log-likelihood per track: the sum of the best final score of each unbroken stretch. */
  trackLogLikelihoods?: GraphDataView<'float32'>;
  /** Optional one-row count of points with a matched edge. */
  matchedCount?: GraphDataView<'uint32'>;
  /** Optional one-row count of points flagged in `breaks` (ordinary and budget breaks). */
  breakCount?: GraphDataView<'uint32'>;
  /**
   * Optional one-row flag: 1 when a bounded table dropped entries, so the result may differ from
   * the exact model. Either the edge grid dropped entries (an edge spanning more than 16 cells per
   * axis, or more edge-cell entries than `entryCapacity`), which can hide candidates, or a route
   * search filled its `routeNodeBudget` table (see `routeOverflow`), which can miss or overestimate
   * a route.
   */
  overflow?: GraphDataView<'uint32'>;
  /**
   * Optional one-row count of the network nodes the route searches discovered but could not keep
   * in their `routeNodeBudget` tables, summed over every search and saturating at `0xffffffff`.
   * 0 means every route search was exact (up to f32 rounding).
   */
  routeOverflow?: GraphDataView<'uint32'>;
};

/** Properties for {@link GPUMapMatching}. */
export type GPUMapMatchingProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'map-matching'`. */
  id?: string;
  /**
   * Coordinates of `points` and `nodePositions`. Omitted means native planar coordinates.
   * `{coordinateSpace: 'longitude-latitude', metric: 'great-circle'}` takes degrees and measures
   * every distance (emission, straight-line and route lengths, `sigma`, `beta`, `searchRadius`,
   * `routeSlack`) in great-circle meters; `sphereRadius` defaults to `GPU_EARTH_MEAN_RADIUS_METERS`.
   */
  spatialContext?: GPUSpatialContext;
  /** GPS points of every track, concatenated in time order. */
  points: GraphDataView<'float32x2'>;
  /** Track-to-point offsets with `trackCount + 1` rows. */
  trackOffsets: GraphDataView<'uint32'>;
  /** Node positions, in the same coordinates as `points`. */
  nodePositions: GraphDataView<'float32x2'>;
  /** CSR row offsets with `nodeCount + 1` rows; the CSR row is the edge ID. */
  offsets: GraphDataView<'uint32'>;
  /** Edge target node per CSR row. Edges with an out-of-range target are ignored. */
  edgeTargets: GraphDataView<'uint32'>;
  /**
   * Per-frame parameters: `GPU_MAP_MATCHING_PARAMETER_LENGTH` `float32` words written with
   * {@link encodeGPUMapMatchingParameters}.
   */
  parameters: GraphDataView<'float32'>;
  /** Compile-time candidates kept per point, 1 to 8. Defaults to 4. */
  candidateCount?: number;
  /**
   * Compile-time edge grid cell size, in coordinate units (degrees on longitude/latitude). Choose
   * about the largest search radius.
   */
  cellSize: number;
  /**
   * Compile-time bounds of the edge grid, in coordinate units (nodes and points outside are
   * clamped). On longitude/latitude the maximum longitude may exceed 180 for a grid that crosses
   * the antimeridian.
   */
  bounds: {minimum: readonly [number, number]; maximum: readonly [number, number]};
  /** Compile-time capacity of the edge-cell table. Defaults to `16 * edgeCount + 256`. */
  entryCapacity?: number;
  /**
   * Compile-time size of each route search's node table, 8 to 128. Defaults to 64. When a search
   * discovers more nodes, the table keeps the open nodes with the best A* keys and drops the rest,
   * so a route can be overestimated or missed; `overflow`, `routeOverflow` and budget breaks
   * (`breaks` value 2) report it. A search needs about as many nodes as the street blocks between
   * two samples: on a 100 m street grid, 64 covers samples up to about 800 m apart and 128 up to
   * about 1.1 km.
   */
  routeNodeBudget?: number;
  /** Caller-owned output. */
  output: GPUMapMatchingOutput;
};

/**
 * Hidden Markov model map matching (Newson and Krumm 2009) of GPS tracks onto a road network.
 *
 * Per point, up to `candidateCount` distinct edges within `searchRadius` are found through a uniform
 * edge grid built each encoding (integer atomics and a prefix sum) and kept in order of distance,
 * ties by edge row. Emission is Gaussian in the perpendicular distance (`sigma`). The transition
 * from a candidate of one point to a candidate of the next is exponential in
 * `|straight distance - route distance|` (`beta`), where the route runs along the first edge from
 * its position to the target node, over the network, then along the second edge to its position.
 * Network distances come from one bounded A* search per source candidate (Euclidean heuristic toward
 * the next point's candidate edges, so the search does not settle the whole disc that plain Dijkstra
 * would): the search stops at `routeFactor * straight + routeSlack` and keeps at most
 * `routeNodeBudget` nodes in a private table. A full table keeps the open nodes with the best keys,
 * so the search can still overestimate or miss a route when samples are far apart for the budget;
 * every such search is counted (`overflow`, `routeOverflow`), never silent. The route cost is the
 * edge length, and every edge row is directed: list both directions of a two-way road (each then
 * takes a candidate slot). CSR rows past `offsets[nodeCount]` are not part of the network.
 *
 * Viterbi runs with one thread per track, so tracks are parallel and the points of one track are
 * sequential. Where no candidate of a point is reachable from any candidate of the previous one,
 * the model restarts there (a break) instead of failing the track; a break where a route search
 * into the point dropped nodes is a budget break (`breaks` value 2). Results are deterministic
 * (integer atomics only, ties by index).
 *
 * Distances are planar by default. With a great-circle `spatialContext` positions are longitude and
 * latitude degrees and every distance is great-circle meters: edge lengths and straight-line
 * distances are haversine, the closest point of an edge is found in the local equirectangular frame
 * of the GPS point, the A* heuristic is the great-circle distance (still consistent), and the
 * candidate window widens in longitude by `1 / cos` of the highest latitude it reaches and wraps
 * at +-180. Edges are straight segments between nodes, so they should be road-length. For a
 * network that crosses the antimeridian give `bounds` across it (for example longitudes 179.9 to
 * 180.1): longitudes are unwrapped around the grid center.
 */
export class GPUMapMatching implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUMapMatchingProps;
  /** Number of GPS points. */
  readonly pointCount: number;
  /** Number of tracks. */
  readonly trackCount: number;
  /** Number of network nodes. */
  readonly nodeCount: number;
  /** Number of directed edges. */
  readonly edgeCount: number;
  /** Candidates per point. */
  readonly candidateCount: number;
  /** Edge grid width in cells. */
  readonly gridWidth: number;
  /** Edge grid height in cells. */
  readonly gridHeight: number;
  /** Resolved spatial context. */
  readonly spatialContext: GPUSpatialContext;

  constructor(props: GPUMapMatchingProps) {
    this.id = props.id ?? 'map-matching';
    this.props = props;
    const {id, props: p} = {id: this.id, props};
    const {output} = props;
    this.spatialContext = resolveNetworkSpatialContext(id, props.spatialContext);
    validatePackedView(props.points, ['float32x2'], `${id} points`);
    validatePackedView(props.nodePositions, ['float32x2'], `${id} nodePositions`);
    validatePackedView(props.parameters, ['float32'], `${id} parameters`);
    for (const [name, view] of [
      ['trackOffsets', props.trackOffsets],
      ['offsets', props.offsets],
      ['edgeTargets', props.edgeTargets],
      ['output.matchedEdges', output.matchedEdges],
      ['output.breaks', output.breaks],
      ['output.matchedCount', output.matchedCount],
      ['output.breakCount', output.breakCount],
      ['output.overflow', output.overflow],
      ['output.routeOverflow', output.routeOverflow]
    ] as const) {
      if (view) {
        validatePackedUint32View(view, `${id} ${name}`);
      }
    }
    for (const [name, view] of [
      ['output.matchedFractions', output.matchedFractions],
      ['output.matchedOffsets', output.matchedOffsets],
      ['output.snapDistances', output.snapDistances],
      ['output.trackLogLikelihoods', output.trackLogLikelihoods]
    ] as const) {
      if (view) {
        validatePackedView(view, ['float32'], `${id} ${name}`);
      }
    }
    if (output.snappedPositions) {
      validatePackedView(output.snappedPositions, ['float32x2'], `${id} output.snappedPositions`);
    }
    this.pointCount = props.points.length;
    this.trackCount = props.trackOffsets.length - 1;
    this.nodeCount = props.nodePositions.length;
    this.edgeCount = props.edgeTargets.length;
    this.candidateCount = props.candidateCount ?? 4;
    if (this.trackCount < 1) {
      throw new Error(`${id} trackOffsets needs at least one track`);
    }
    if (this.pointCount < 1 || this.edgeCount < 1 || this.nodeCount < 1) {
      throw new Error(`${id} needs at least one point, node and edge`);
    }
    if (props.offsets.length !== this.nodeCount + 1) {
      throw new Error(`${id} offsets must contain one more row than nodePositions`);
    }
    if (props.parameters.length !== GPU_MAP_MATCHING_PARAMETER_LENGTH) {
      throw new Error(`${id} parameters must contain ${GPU_MAP_MATCHING_PARAMETER_LENGTH} rows`);
    }
    if (
      !Number.isInteger(this.candidateCount) ||
      this.candidateCount < 1 ||
      this.candidateCount > 8
    ) {
      throw new Error(`${id} candidateCount must be an integer from 1 to 8`);
    }
    const budget = props.routeNodeBudget ?? 64;
    if (!Number.isInteger(budget) || budget < 8 || budget > 128) {
      throw new Error(`${id} routeNodeBudget must be an integer from 8 to 128`);
    }
    if (!(props.cellSize > 0) || !Number.isFinite(props.cellSize)) {
      throw new Error(`${id} cellSize must be positive`);
    }
    const [minX, minY] = props.bounds.minimum;
    const [maxX, maxY] = props.bounds.maximum;
    if (!(maxX > minX) || !(maxY > minY)) {
      throw new Error(`${id} bounds must have a positive extent`);
    }
    this.gridWidth = Math.max(1, Math.ceil((maxX - minX) / props.cellSize));
    this.gridHeight = Math.max(1, Math.ceil((maxY - minY) / props.cellSize));
    if (this.gridWidth * this.gridHeight > 1 << 24) {
      throw new Error(`${id} cellSize yields more than 2^24 grid cells; use a larger cellSize`);
    }
    if (
      props.entryCapacity !== undefined &&
      (!Number.isSafeInteger(props.entryCapacity) || props.entryCapacity < 1)
    ) {
      throw new Error(`${id} entryCapacity must be a positive integer`);
    }
    for (const [name, view, length] of [
      ['matchedEdges', output.matchedEdges, this.pointCount],
      ['matchedFractions', output.matchedFractions, this.pointCount],
      ['matchedOffsets', output.matchedOffsets, this.pointCount],
      ['snapDistances', output.snapDistances, this.pointCount],
      ['snappedPositions', output.snappedPositions, this.pointCount],
      ['breaks', output.breaks, this.pointCount],
      ['trackLogLikelihoods', output.trackLogLikelihoods, this.trackCount],
      ['matchedCount', output.matchedCount, 1],
      ['breakCount', output.breakCount, 1],
      ['overflow', output.overflow, 1],
      ['routeOverflow', output.routeOverflow, 1]
    ] as const) {
      if (view && view.length !== length) {
        throw new Error(`${id} output.${name} must contain ${length} rows`);
      }
    }
    void p;
    const outputs = getOutputs(props);
    validateGraphOutputsDisjointFromInputs(id, outputs, getInputs(props));
    const outputBuffers = outputs.filter(view => view !== undefined).map(view => view.buffer);
    if (new Set(outputBuffers).size !== outputBuffers.length) {
      throw new Error(`${id} outputs must use separate buffers`);
    }
  }

  /**
   * Returns `clear-edge-table` and `edge-table`, the edge grid (`cell-count`, scan, `cell-fill`),
   * `candidates`, `routes`, the `forward` and `backward` Viterbi passes, `route-status`, and the
   * optional output nodes.
   */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, pointCount, trackCount, nodeCount, edgeCount, candidateCount} = this;
    const {output} = props;
    validateGraphViewsBelongToGraph(id, graph, [...getInputs(props), ...getOutputs(props)]);
    const budget = props.routeNodeBudget ?? 64;
    const cellCount = this.gridWidth * this.gridHeight;
    const entryCapacity = props.entryCapacity ?? 16 * edgeCount + 256;
    const [minX, minY] = props.bounds.minimum;
    const maxX = props.bounds.maximum[0];
    const geodesicRadius = getNetworkGeodesicRadius(this.spatialContext);
    const transient = <Format extends 'uint32' | 'float32'>(
      name: string,
      format: Format,
      length: number
    ) => createTransientView(graph, `${id}-${name}`, format, Math.max(length, 1));
    const read = (name: string, view: GraphDataView, type: WGSLKernelBinding['type'] = 'u32') =>
      ({name, view, type, access: 'read'}) as WGSLKernelBinding;
    const write = (name: string, view: GraphDataView, type: WGSLKernelBinding['type'] = 'u32') =>
      ({name, view, type, access: 'read_write'}) as WGSLKernelBinding;
    const common = `const POINT_COUNT: u32 = ${pointCount}u;
const NODE_COUNT: u32 = ${nodeCount}u;
const EDGE_COUNT: u32 = ${edgeCount}u;
const K: u32 = ${candidateCount}u;
// Route rows: K route lengths, then the number of nodes the route search dropped (an exact f32).
const ROUTE_STRIDE: u32 = ${candidateCount + 1}u;
const NONE: u32 = 0xffffffffu;
// Back pointer of a restart where a route search into the point dropped nodes: a budget break.
const BUDGET_BREAK: u32 = 0xfffffffeu;
const NEG: f32 = -1.0e30;
const GRID_WIDTH: u32 = ${this.gridWidth}u;
const GRID_HEIGHT: u32 = ${this.gridHeight}u;
const CELL_COUNT: u32 = ${cellCount}u;
const ENTRY_CAPACITY: u32 = ${entryCapacity}u;
const GRID_MINIMUM: vec2<f32> = vec2<f32>(${toLiteral(minX)}, ${toLiteral(minY)});
const CELL_SIZE: f32 = ${toLiteral(props.cellSize)};
const MAX_EDGE_SPAN: u32 = 16u;
fn getCellX(x: f32) -> u32 {
  return u32(clamp(floor((x - GRID_MINIMUM.x) / CELL_SIZE), 0.0, f32(GRID_WIDTH - 1u)));
}
fn getCellY(y: f32) -> u32 {
  return u32(clamp(floor((y - GRID_MINIMUM.y) / CELL_SIZE), 0.0, f32(GRID_HEIGHT - 1u)));
}
${getNetworkMetricWGSL(geodesicRadius)}
${getGridXWGSL(geodesicRadius !== undefined, minX, maxX)}`;
    const nodeAccess = `fn readNode(node: u32) -> vec2<f32> {
  return vec2<f32>(nodePositions[nodePositionsOffset + node * 2u], nodePositions[nodePositionsOffset + node * 2u + 1u]);
}`;
    const edgeTableAccess = `
fn edgeSource(edge: u32) -> u32 { return edgeTable[edgeTableOffset + edge * 3u]; }
fn edgeTarget(edge: u32) -> u32 { return edgeTable[edgeTableOffset + edge * 3u + 1u]; }
fn edgeLength(edge: u32) -> f32 { return bitcast<f32>(edgeTable[edgeTableOffset + edge * 3u + 2u]); }
fn isEdgeValid(edge: u32) -> bool { return edgeLength(edge) >= 0.0; }`;
    const kernel = (
      variant: string,
      bindings: WGSLKernelBinding[],
      invocationCount: number,
      body: string,
      declarations = ''
    ) =>
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-${variant}`,
        operation: OPERATION,
        variant,
        bindings,
        invocationCount,
        declarations: `${common}
${bindings.some(binding => binding.name === 'nodePositions') ? nodeAccess : ''}
${declarations}`,
        body
      });

    const edgeTable = transient('edge-table', 'uint32', edgeCount * 3);
    const cellCounts = transient('cell-counts', 'uint32', cellCount);
    const cellEnds = transient('cell-ends', 'uint32', cellCount);
    const cursors = transient('cursors', 'uint32', cellCount);
    const entries = transient('entries', 'uint32', entryCapacity);
    const candidates = transient('candidates', 'uint32', pointCount * candidateCount * 3);
    const pointStarts = transient('point-starts', 'uint32', pointCount);
    const routes = transient(
      'routes',
      'float32',
      pointCount * candidateCount * (candidateCount + 1)
    );
    const scores = transient('scores', 'float32', pointCount * candidateCount);
    const backPointers = transient('back-pointers', 'uint32', pointCount * candidateCount);
    const chosen = transient('chosen', 'uint32', pointCount);
    const breaks = output.breaks ?? transient('breaks', 'uint32', pointCount);
    const overflow = output.overflow ?? transient('overflow', 'uint32', 1);

    const nodes: GPUCommandNode<Parameters>[] = [];
    // Source node, target node and length of every CSR row; -1 length marks an invalid edge. Every
    // word starts as the bits of -1.0 (an out-of-range node and a -1 length), so the rows past
    // offsets[NODE_COUNT], which the per-node kernel never writes, are invalid edges instead of
    // stale rows of an earlier frame or of an aliased transient.
    nodes.push(
      createFillNode<Parameters>(graph, {
        id: `${id}-clear-edge-table`,
        operation: OPERATION,
        view: edgeTable,
        type: 'u32',
        value: `${INVALID_EDGE_WORD}u`
      }),
      kernel(
        'edge-table',
        [
          read('adjacency', props.offsets),
          read('edgeTargets', props.edgeTargets),
          read('nodePositions', props.nodePositions, 'f32'),
          write('edgeTable', edgeTable)
        ],
        nodeCount,
        `let end = min(adjacency[adjacencyOffset + index + 1u], EDGE_COUNT);
  let start = readNode(index);
  for (var edge = adjacency[adjacencyOffset + index]; edge < end; edge++) {
    let targetNode = edgeTargets[edgeTargetsOffset + edge];
    var length = -1.0;
    if (targetNode < NODE_COUNT) {
      length = networkDistance(start, readNode(targetNode));
    }
    edgeTable[edgeTableOffset + edge * 3u] = index;
    edgeTable[edgeTableOffset + edge * 3u + 1u] = targetNode;
    edgeTable[edgeTableOffset + edge * 3u + 2u] = bitcast<u32>(length);
  }`
      ),
      createFillNode<Parameters>(graph, {
        id: `${id}-clear-counts`,
        operation: OPERATION,
        view: cellCounts,
        type: 'u32',
        value: '0u'
      }),
      createFillNode<Parameters>(graph, {
        id: `${id}-clear-cursors`,
        operation: OPERATION,
        view: cursors,
        type: 'u32',
        value: '0u'
      }),
      createFillNode<Parameters>(graph, {
        id: `${id}-clear-overflow`,
        operation: OPERATION,
        view: overflow,
        type: 'u32',
        value: '0u'
      })
    );
    const cellRange = `let a = readNode(edgeSource(index));
  let b = readNode(edgeTarget(index));
  let gridXA = getGridX(a.x);
  let gridXB = getGridXNear(b.x, a.x, gridXA);
  let cellX0 = getCellX(min(gridXA, gridXB));
  let cellX1 = getCellX(max(gridXA, gridXB));
  let cellY0 = getCellY(min(a.y, b.y));
  let cellY1 = getCellY(max(a.y, b.y));`;
    nodes.push(
      kernel(
        'cell-count',
        [
          read('edgeTable', edgeTable),
          read('nodePositions', props.nodePositions, 'f32'),
          write('cellCounts', cellCounts, 'atomic<u32>')
        ],
        edgeCount,
        `if (!isEdgeValid(index)) {
    return;
  }
  ${cellRange}
  for (var cellY = cellY0; cellY <= min(cellY1, cellY0 + MAX_EDGE_SPAN - 1u); cellY++) {
    for (var cellX = cellX0; cellX <= min(cellX1, cellX0 + MAX_EDGE_SPAN - 1u); cellX++) {
      atomicAdd(&cellCounts[cellCountsOffset + cellY * GRID_WIDTH + cellX], 1u);
    }
  }`,
        edgeTableAccess
      ),
      ...new GPUScan({
        id: `${id}-cell-scan`,
        input: cellCounts,
        output: cellEnds,
        mode: 'inclusive'
      }).getCommandNodes(graph),
      kernel(
        'cell-fill',
        [
          read('edgeTable', edgeTable),
          read('nodePositions', props.nodePositions, 'f32'),
          read('cellEnds', cellEnds),
          write('cursors', cursors, 'atomic<u32>'),
          write('entries', entries),
          write('overflow', overflow)
        ],
        edgeCount,
        `if (!isEdgeValid(index)) {
    return;
  }
  ${cellRange}
  if (cellX1 - cellX0 >= MAX_EDGE_SPAN || cellY1 - cellY0 >= MAX_EDGE_SPAN) {
    overflow[overflowOffset] = 1u;
  }
  for (var cellY = cellY0; cellY <= min(cellY1, cellY0 + MAX_EDGE_SPAN - 1u); cellY++) {
    for (var cellX = cellX0; cellX <= min(cellX1, cellX0 + MAX_EDGE_SPAN - 1u); cellX++) {
      let cell = cellY * GRID_WIDTH + cellX;
      let start = select(0u, cellEnds[cellEndsOffset + cell - 1u], cell > 0u);
      let slot = start + atomicAdd(&cursors[cursorsOffset + cell], 1u);
      if (slot < ENTRY_CAPACITY) {
        entries[entriesOffset + slot] = index;
      } else {
        overflow[overflowOffset] = 1u;
      }
    }
  }`,
        edgeTableAccess
      ),
      // Per point: the K nearest distinct edges within the radius, sorted by (distance, edge row).
      kernel(
        'candidates',
        [
          read('points', props.points, 'f32'),
          read('nodePositions', props.nodePositions, 'f32'),
          read('edgeTable', edgeTable),
          read('cellEnds', cellEnds),
          read('entries', entries),
          read('parameters', props.parameters, 'f32'),
          write('candidates', candidates)
        ],
        pointCount,
        `let point = vec2<f32>(points[pointsOffset + index * 2u], points[pointsOffset + index * 2u + 1u]);
  ${getCandidateRadiusWGSL(geodesicRadius !== undefined)}
  var bestEdge: array<u32, ${candidateCount}>;
  var bestFraction: array<f32, ${candidateCount}>;
  var bestDistance: array<f32, ${candidateCount}>;
  for (var slot = 0u; slot < K; slot++) {
    bestEdge[slot] = NONE;
    bestFraction[slot] = 0.0;
    bestDistance[slot] = 0.0;
  }
  let cellY0 = getCellY(point.y - latitudeExtent);
  let cellY1 = getCellY(point.y + latitudeExtent);
  for (var windowIndex = 0u; windowIndex < WINDOW_COUNT; windowIndex++) {
  let longitudeRange = getLongitudeRange(getGridX(point.x), longitudeExtent, windowIndex);
  if (longitudeRange.x > longitudeRange.y) {
    continue;
  }
  let cellX0 = getCellX(longitudeRange.x);
  let cellX1 = getCellX(longitudeRange.y);
  for (var cellY = cellY0; cellY <= cellY1; cellY++) {
    for (var cellX = cellX0; cellX <= cellX1; cellX++) {
      let cell = cellY * GRID_WIDTH + cellX;
      let start = select(0u, cellEnds[cellEndsOffset + cell - 1u], cell > 0u);
      let end = min(cellEnds[cellEndsOffset + cell], ENTRY_CAPACITY);
      for (var entry = start; entry < end; entry++) {
        let edge = entries[entriesOffset + entry];
        let origin = readNode(edgeSource(edge));
        let segmentEnd = readNode(edgeTarget(edge));
        let fraction = networkSegmentFraction(point, origin, segmentEnd);
        let distance = networkSegmentDistance(point, origin, segmentEnd, fraction);
        if (!(distance <= radius)) {
          continue;
        }
        var isKnown = false;
        for (var slot = 0u; slot < K; slot++) {
          isKnown = isKnown || bestEdge[slot] == edge;
        }
        if (isKnown) {
          continue;
        }
        // Insertion into the sorted list; the last slot is dropped when full.
        var position = K;
        for (var slot = 0u; slot < K; slot++) {
          if (position == K && (bestEdge[slot] == NONE || distance < bestDistance[slot] ||
              (distance == bestDistance[slot] && edge < bestEdge[slot]))) {
            position = slot;
          }
        }
        if (position < K) {
          for (var slot = K - 1u; slot > position; slot--) {
            bestEdge[slot] = bestEdge[slot - 1u];
            bestFraction[slot] = bestFraction[slot - 1u];
            bestDistance[slot] = bestDistance[slot - 1u];
          }
          bestEdge[position] = edge;
          bestFraction[position] = fraction;
          bestDistance[position] = distance;
        }
      }
    }
  }
  }
  for (var slot = 0u; slot < K; slot++) {
    let base = candidatesOffset + (index * K + slot) * 3u;
    candidates[base] = bestEdge[slot];
    candidates[base + 1u] = bitcast<u32>(bestFraction[slot]);
    candidates[base + 2u] = bitcast<u32>(bestDistance[slot]);
  }`,
        `${edgeTableAccess}
${getLongitudeRangeWGSL(geodesicRadius !== undefined, maxX)}`
      ),
      createFillNode<Parameters>(graph, {
        id: `${id}-clear-starts`,
        operation: OPERATION,
        view: pointStarts,
        type: 'u32',
        value: '0u'
      }),
      kernel(
        'starts',
        [read('trackOffsets', props.trackOffsets), write('pointStarts', pointStarts)],
        trackCount,
        `let first = trackOffsets[trackOffsetsOffset + index];
  if (first < POINT_COUNT) {
    pointStarts[pointStartsOffset + first] = 1u;
  }`
      ),
      kernel(
        'routes',
        [
          read('pointStarts', pointStarts),
          read('points', props.points, 'f32'),
          read('nodePositions', props.nodePositions, 'f32'),
          read('adjacency', props.offsets),
          read('edgeTable', edgeTable),
          read('candidates', candidates),
          read('parameters', props.parameters, 'f32'),
          write('routes', routes, 'f32')
        ],
        pointCount * candidateCount,
        getRoutesBody(),
        getForwardDeclarations(budget, getHashSize(budget), candidateCount, edgeTableAccess)
      ),
      kernel(
        'forward',
        [
          read('trackOffsets', props.trackOffsets),
          read('points', props.points, 'f32'),
          read('candidates', candidates),
          read('parameters', props.parameters, 'f32'),
          read('routes', routes, 'f32'),
          write('scores', scores, 'f32'),
          write('backPointers', backPointers)
        ],
        trackCount,
        getForwardBody(),
        getViterbiDeclarations(candidateCount)
      ),
      kernel(
        'backward',
        [
          read('trackOffsets', props.trackOffsets),
          read('scores', scores, 'f32'),
          read('backPointers', backPointers),
          write('chosen', chosen),
          write('breaks', breaks),
          ...(output.trackLogLikelihoods
            ? [write('logLikelihoods', output.trackLogLikelihoods, 'f32')]
            : [])
        ],
        trackCount,
        getBackwardBody(Boolean(output.trackLogLikelihoods))
      )
    );

    // Route-search drops: raise `overflow` and sum the dropped nodes into `routeOverflow`.
    if (output.routeOverflow) {
      nodes.push(
        createFillNode<Parameters>(graph, {
          id: `${id}-clear-route-overflow`,
          operation: OPERATION,
          view: output.routeOverflow,
          type: 'u32',
          value: '0u'
        })
      );
    }
    nodes.push(
      kernel(
        'route-status',
        [
          read('routes', routes, 'f32'),
          write('overflow', overflow),
          ...(output.routeOverflow
            ? [write('routeOverflow', output.routeOverflow, 'atomic<u32>')]
            : [])
        ],
        pointCount * candidateCount,
        `let dropped = u32(routes[routesOffset + index * ROUTE_STRIDE + K]);
  if (dropped == 0u) {
    return;
  }
  overflow[overflowOffset] = 1u;
  ${
    output.routeOverflow
      ? `// Saturating sum: an add that wraps pins the total at the maximum, and every later add
  // that wraps from there pins it again.
  let previous = atomicAdd(&routeOverflow[routeOverflowOffset], dropped);
  if (previous + dropped < previous) {
    atomicMax(&routeOverflow[routeOverflowOffset], 0xffffffffu);
  }`
      : ''
  }`
      )
    );

    nodes.push(
      kernel(
        'matched',
        [
          read('candidates', candidates),
          read('chosen', chosen),
          read('edgeTable', edgeTable),
          write('matchedEdges', output.matchedEdges),
          ...(output.matchedFractions ? [write('fractions', output.matchedFractions, 'f32')] : []),
          ...(output.matchedOffsets ? [write('offsets', output.matchedOffsets, 'f32')] : []),
          ...(output.snapDistances ? [write('distances', output.snapDistances, 'f32')] : [])
        ],
        pointCount,
        `let slot = chosen[chosenOffset + index];
  let isMatched = slot != NONE;
  let base = candidatesOffset + (index * K + min(slot, K - 1u)) * 3u;
  let edge = candidates[base];
  matchedEdges[matchedEdgesOffset + index] = select(NONE, edge, isMatched);
  ${
    output.matchedFractions
      ? 'fractions[fractionsOffset + index] = select(-1.0, bitcast<f32>(candidates[base + 1u]), isMatched);'
      : ''
  }
  ${
    output.matchedOffsets
      ? `offsets[offsetsOffset + index] = select(-1.0, bitcast<f32>(candidates[base + 1u]) * edgeLength(select(0u, edge, isMatched)), isMatched);`
      : ''
  }
  ${
    output.snapDistances
      ? 'distances[distancesOffset + index] = select(-1.0, bitcast<f32>(candidates[base + 2u]), isMatched);'
      : ''
  }`,
        edgeTableAccess
      )
    );
    if (output.snappedPositions) {
      nodes.push(
        kernel(
          'positions',
          [
            read('points', props.points, 'f32'),
            read('candidates', candidates),
            read('chosen', chosen),
            read('edgeTable', edgeTable),
            read('nodePositions', props.nodePositions, 'f32'),
            write('snappedPositions', output.snappedPositions, 'f32')
          ],
          pointCount,
          `var position = vec2<f32>(points[pointsOffset + index * 2u], points[pointsOffset + index * 2u + 1u]);
  let slot = chosen[chosenOffset + index];
  if (slot != NONE) {
    let base = candidatesOffset + (index * K + slot) * 3u;
    let edge = candidates[base];
    let origin = readNode(edgeSource(edge));
    position = networkInterpolate(origin, readNode(edgeTarget(edge)), bitcast<f32>(candidates[base + 1u]));
  }
  snappedPositions[snappedPositionsOffset + index * 2u] = position.x;
  snappedPositions[snappedPositionsOffset + index * 2u + 1u] = position.y;`,
          edgeTableAccess
        )
      );
    }
    if (output.matchedCount || output.breakCount) {
      const countViews = [output.matchedCount, output.breakCount].filter(view => view);
      for (const view of countViews) {
        nodes.push(
          createFillNode<Parameters>(graph, {
            id: `${id}-clear-${view === output.matchedCount ? 'matched' : 'break'}-count`,
            operation: OPERATION,
            view: view!,
            type: 'u32',
            value: '0u'
          })
        );
      }
      nodes.push(
        kernel(
          'summary',
          [
            read('chosen', chosen),
            read('breaks', breaks),
            ...(output.matchedCount
              ? [write('matchedCount', output.matchedCount, 'atomic<u32>')]
              : []),
            ...(output.breakCount ? [write('breakCount', output.breakCount, 'atomic<u32>')] : [])
          ],
          pointCount,
          `${
            output.matchedCount
              ? 'if (chosen[chosenOffset + index] != NONE) { atomicAdd(&matchedCount[matchedCountOffset], 1u); }'
              : ''
          }
  ${
    output.breakCount
      ? 'if (breaks[breaksOffset + index] != 0u) { atomicAdd(&breakCount[breakCountOffset], 1u); }'
      : ''
  }`
        )
      );
    }
    return nodes;
  }
}

/**
 * Declares the candidate `radius` (distance units) and the `latitudeExtent` and `longitudeExtent`
 * (coordinate units) of the window around `point`. A great-circle window widens in longitude by
 * `1 / cos` of the highest latitude of the band, up to every longitude near a pole.
 */
function getCandidateRadiusWGSL(isGeodesic: boolean): string {
  if (!isGeodesic) {
    return `let radius = min(parameters[parametersOffset + 2u], 2.0 * CELL_SIZE);
  let latitudeExtent = radius;
  let longitudeExtent = radius;`;
  }
  return `let radius = min(
    parameters[parametersOffset + 2u],
    2.0 * CELL_SIZE * GEODESIC_DEGREES_TO_RADIANS * NETWORK_EARTH_RADIUS
  );
  let latitudeExtent = radius / NETWORK_EARTH_RADIUS * GEODESIC_RADIANS_TO_DEGREES;
  let bandLatitude = min(abs(point.y) + latitudeExtent, 90.0);
  let longitudeExtent = min(latitudeExtent / max(geodesicCosLatitude(bandLatitude), 1.0e-6), 180.0);`;
}

/**
 * Declares `getGridX(x)`, the x of a coordinate in the edge grid, and
 * `getGridXNear(x, anchorX, anchorGridX)`, the grid x of `x` continuous with an anchor whose grid
 * x is known. Planar coordinates are used as they are. Longitudes are unwrapped to be continuous
 * around the grid center, so a grid whose bounds cross the antimeridian (for example 179.9 to
 * 180.1) buckets nodes on both sides, and longitude differences take the shorter way around.
 */
function getGridXWGSL(isGeodesic: boolean, minX: number, maxX: number): string {
  if (!isGeodesic) {
    return `fn getGridX(x: f32) -> f32 {
  return x;
}
fn getGridXNear(x: f32, anchorX: f32, anchorGridX: f32) -> f32 {
  return x;
}`;
  }
  return `const GRID_CENTER_X: f32 = ${toLiteral((minX + maxX) / 2)};
fn getGridX(x: f32) -> f32 {
  return GRID_CENTER_X + geodesicWrapLongitudeDelta(x - GRID_CENTER_X);
}
fn getGridXNear(x: f32, anchorX: f32, anchorGridX: f32) -> f32 {
  return anchorGridX + geodesicWrapLongitudeDelta(x - anchorX);
}`;
}

/**
 * Declares `WINDOW_COUNT` and `getLongitudeRange(x, extent, windowIndex)`, the grid x interval of
 * candidate window `windowIndex`, empty when `x > y`. Planar searches use one window. Great-circle
 * searches add the parts of the window beyond the grid's west and east edges, wrapped by 360
 * degrees, when they land inside a grid that spans the globe; the candidate list skips edges it
 * already holds, so overlapping windows are harmless.
 */
function getLongitudeRangeWGSL(isGeodesic: boolean, maxX: number): string {
  if (!isGeodesic) {
    return `const WINDOW_COUNT: u32 = 1u;
fn getLongitudeRange(x: f32, extent: f32, windowIndex: u32) -> vec2<f32> {
  return vec2<f32>(x - extent, x + extent);
}`;
  }
  return `const WINDOW_COUNT: u32 = 3u;
const GRID_MAXIMUM_X: f32 = ${toLiteral(maxX)};
fn getLongitudeRange(x: f32, extent: f32, windowIndex: u32) -> vec2<f32> {
  let low = x - extent;
  let high = x + extent;
  let empty = vec2<f32>(1.0, 0.0);
  if (windowIndex == 1u) {
    let isWrapped = low < GRID_MINIMUM.x && low + 360.0 < GRID_MAXIMUM_X;
    return select(empty, vec2<f32>(low + 360.0, GRID_MAXIMUM_X), isWrapped);
  }
  if (windowIndex == 2u) {
    let isWrapped = high > GRID_MAXIMUM_X && high - 360.0 > GRID_MINIMUM.x;
    return select(empty, vec2<f32>(GRID_MINIMUM.x, high - 360.0), isWrapped);
  }
  return vec2<f32>(low, high);
}`;
}

function toLiteral(value: number): string {
  const text = String(Math.fround(value));
  return /[.eE]/.test(text) ? text : `${text}.0`;
}

/** Hash cells for a node table: the next power of two at least twice the table size. */
function getHashSize(budget: number): number {
  let size = 16;
  while (size < 2 * budget) {
    size *= 2;
  }
  return size;
}

/** Module declarations of the forward Viterbi kernel: private route-search tables and search. */
function getForwardDeclarations(
  budget: number,
  hashSize: number,
  candidateCount: number,
  edgeTableAccess: string
): string {
  return `${edgeTableAccess}
${getAccessDeclarations()}
const S: u32 = ${budget}u;
const HASH_MASK: u32 = ${hashSize - 1}u;
var<private> tableNode: array<u32, ${budget}>;
var<private> tableDistance: array<f32, ${budget}>;
var<private> tableKey: array<f32, ${budget}>;
var<private> tableDone: array<u32, ${budget}>;
var<private> tableHash: array<u32, ${hashSize}>;
var<private> pendingTargets: array<u32, ${candidateCount}>;
var<private> targetPositions: array<vec2<f32>, ${candidateCount}>;
var<private> targetTotal: u32 = 0u;
// Discovered nodes the full table could not keep in the last search.
var<private> droppedCount: u32 = 0u;

// Consistent A* heuristic: the distance to the nearest target node, planar or great-circle. Every
// edge is as long as the straight line (or great circle) between its nodes, so by the triangle
// inequality the heuristic never overestimates; the factor
// below 1 absorbs f32 rounding so it stays consistent in floating point.
fn getHeuristic(node: u32) -> f32 {
  if (targetTotal == 0u || node >= NODE_COUNT) {
    return 0.0;
  }
  let position = readNode(node);
  var nearest = 3.0e38;
  for (var slot = 0u; slot < targetTotal; slot++) {
    nearest = min(nearest, networkDistance(position, targetPositions[slot]));
  }
  return nearest * 0.999999;
}

fn getHashStart(node: u32) -> u32 {
  return ((node * 2654435761u) >> 7u) & HASH_MASK;
}
// Table slot of a node, or NONE. The hash has at least twice as many cells as table slots.
fn findEntry(node: u32) -> u32 {
  var cell = getHashStart(node);
  loop {
    let stored = tableHash[cell];
    if (stored == 0u) {
      return NONE;
    }
    if (tableNode[stored - 1u] == node) {
      return stored - 1u;
    }
    cell = (cell + 1u) & HASH_MASK;
  }
}
// Removes the hash cell of table slot 'slot' with backward-shift deletion: later cells of the
// probe run move into the hole unless their home cell lies cyclically after it, so every lookup
// still reaches its entry without tombstones. The hash always has an empty cell (it has at least
// twice as many cells as table slots), so the loops end.
fn removeHashEntry(slot: u32) {
  var hole = getHashStart(tableNode[slot]);
  while (tableHash[hole] != slot + 1u) {
    hole = (hole + 1u) & HASH_MASK;
  }
  tableHash[hole] = 0u;
  var cell = (hole + 1u) & HASH_MASK;
  loop {
    let stored = tableHash[cell];
    if (stored == 0u) {
      break;
    }
    let home = getHashStart(tableNode[stored - 1u]);
    if (((cell - home) & HASH_MASK) >= ((cell - hole) & HASH_MASK)) {
      tableHash[hole] = stored;
      tableHash[cell] = 0u;
      hole = cell;
    }
    cell = (cell + 1u) & HASH_MASK;
  }
}
// The open (unsettled) slot with the largest key, ties to the highest slot, or NONE.
fn findWorstOpenEntry(count: u32) -> u32 {
  var worst = NONE;
  for (var slot = 0u; slot < count; slot++) {
    if (tableDone[slot] == 0u && (worst == NONE || tableKey[slot] >= tableKey[worst])) {
      worst = slot;
    }
  }
  return worst;
}
fn addEntry(slot: u32, node: u32, distance: f32) {
  tableNode[slot] = node;
  tableDistance[slot] = distance;
  tableKey[slot] = distance + getHeuristic(node);
  tableDone[slot] = 0u;
  var cell = getHashStart(node);
  while (tableHash[cell] != 0u) {
    cell = (cell + 1u) & HASH_MASK;
  }
  tableHash[cell] = slot + 1u;
}

// Bounded A* from 'start' over the CSR network; distances within 'limit' land in the private
// table (at most S nodes; a full table counts every node it cannot keep in 'droppedCount' and keeps
// the open nodes with the best keys). Nodes are settled in order of distance plus the heuristic, so a node
// settled (done) has its exact network distance, and the search heads toward the targets instead of
// settling the whole disc of radius 'limit' as plain Dijkstra does. It stops once every node in
// 'pendingTargets' (NONE for unused slots; 'pendingCount' of them are set) is settled, or once the
// best key exceeds 'limit' (no remaining target can be within the bound). Returns the table size.
fn searchNetwork(start: u32, limit: f32, pendingCount: u32) -> u32 {
  var pending = pendingCount;
  droppedCount = 0u;
  targetTotal = 0u;
  for (var slot = 0u; slot < K; slot++) {
    if (pendingTargets[slot] != NONE) {
      targetPositions[targetTotal] = readNode(pendingTargets[slot]);
      targetTotal++;
    }
  }
  for (var slot = 0u; slot <= HASH_MASK; slot++) {
    tableHash[slot] = 0u;
  }
  var count = 0u;
  addEntry(count, start, 0.0);
  count = 1u;
  loop {
    var best = NONE;
    var bestKey = limit + 1.0;
    for (var slot = 0u; slot < count; slot++) {
      if (tableDone[slot] == 0u && tableKey[slot] < bestKey) {
        best = slot;
        bestKey = tableKey[slot];
      }
    }
    if (best == NONE || bestKey > limit) {
      break;
    }
    let bestDistance = tableDistance[best];
    tableDone[best] = 1u;
    let node = tableNode[best];
    for (var slot = 0u; slot < K; slot++) {
      if (pendingTargets[slot] == node) {
        pendingTargets[slot] = NONE;
        pending--;
      }
    }
    if (pending == 0u) {
      break;
    }
    if (node >= NODE_COUNT) {
      continue;
    }
    let end = min(adjacency[adjacencyOffset + node + 1u], EDGE_COUNT);
    for (var edge = adjacency[adjacencyOffset + node]; edge < end; edge++) {
      if (!isEdgeValid(edge)) {
        continue;
      }
      let candidateDistance = bestDistance + edgeLength(edge);
      if (candidateDistance > limit) {
        continue;
      }
      let neighbor = edgeTarget(edge);
      let found = findEntry(neighbor);
      if (found != NONE) {
        if (tableDone[found] == 0u && candidateDistance < tableDistance[found]) {
          tableDistance[found] = candidateDistance;
          tableKey[found] = candidateDistance + getHeuristic(neighbor);
        }
      } else if (count < S) {
        addEntry(count, neighbor, candidateDistance);
        count++;
      } else {
        // Full table: the node is dropped, or replaces the open entry with the worst key when its
        // own key is better. Either way the search may now miss or overestimate a route.
        droppedCount++;
        let key = candidateDistance + getHeuristic(neighbor);
        let worst = findWorstOpenEntry(count);
        if (worst != NONE && key < tableKey[worst]) {
          removeHashEntry(worst);
          addEntry(worst, neighbor, candidateDistance);
        }
      }
    }
  }
  return count;
}

fn lookupDistance(node: u32) -> f32 {
  let slot = findEntry(node);
  if (slot != NONE && tableDone[slot] != 0u) {
    return tableDistance[slot];
  }
  return -1.0;
}
`;
}

/** Candidate and point accessors shared by the route and Viterbi kernels. */
function getAccessDeclarations(): string {
  return `fn readPoint(point: u32) -> vec2<f32> {
  return vec2<f32>(points[pointsOffset + point * 2u], points[pointsOffset + point * 2u + 1u]);
}
fn getCandidateEdge(point: u32, slot: u32) -> u32 {
  return candidates[candidatesOffset + (point * K + slot) * 3u];
}
fn getCandidateFraction(point: u32, slot: u32) -> f32 {
  return bitcast<f32>(candidates[candidatesOffset + (point * K + slot) * 3u + 1u]);
}
fn getCandidateDistance(point: u32, slot: u32) -> f32 {
  return bitcast<f32>(candidates[candidatesOffset + (point * K + slot) * 3u + 2u]);
}

`;
}

/** Module declarations of the forward Viterbi kernel: candidate access and model constants. */
function getViterbiDeclarations(candidateCount: number): string {
  return `fn readPoint(point: u32) -> vec2<f32> {
  return vec2<f32>(points[pointsOffset + point * 2u], points[pointsOffset + point * 2u + 1u]);
}
fn getCandidateEdge(point: u32, slot: u32) -> u32 {
  return candidates[candidatesOffset + (point * K + slot) * 3u];
}
fn getCandidateFraction(point: u32, slot: u32) -> f32 {
  return bitcast<f32>(candidates[candidatesOffset + (point * K + slot) * 3u + 1u]);
}
fn getCandidateDistance(point: u32, slot: u32) -> f32 {
  return bitcast<f32>(candidates[candidatesOffset + (point * K + slot) * 3u + 2u]);
}

const LOG_TWO_PI_HALF: f32 = 0.9189385;
const CANDIDATE_COUNT: u32 = ${candidateCount}u;`;
}

/**
 * Body of the route kernel: one thread per (point, candidate of the previous point). Writes the
 * route length to every candidate of the point, or -1 when there is none within the bound.
 */
function getRoutesBody(): string {
  return `let point = index / K;
  let fromSlot = index % K;
  for (var to = 0u; to < K; to++) {
    routes[routesOffset + index * ROUTE_STRIDE + to] = -1.0;
  }
  routes[routesOffset + index * ROUTE_STRIDE + K] = 0.0;
  if (pointStarts[pointStartsOffset + point] != 0u) {
    return;
  }
  let previous = point - 1u;
  let fromEdge = getCandidateEdge(previous, fromSlot);
  if (fromEdge == NONE) {
    return;
  }
  let straight = networkDistance(readPoint(point), readPoint(previous));
  let bound = straight * parameters[parametersOffset + 3u] + parameters[parametersOffset + 4u];
  let fromLength = edgeLength(fromEdge);
  let fromFraction = getCandidateFraction(previous, fromSlot);
  let tail = (1.0 - fromFraction) * fromLength;
  let remaining = bound - tail;
  if (remaining >= 0.0) {
    var pendingCount = 0u;
    for (var to = 0u; to < K; to++) {
      let toEdge = getCandidateEdge(point, to);
      let isSameEdgeAhead = toEdge == fromEdge && getCandidateFraction(point, to) >= fromFraction;
      let isPending = toEdge != NONE && !isSameEdgeAhead;
      pendingTargets[to] = select(NONE, edgeSource(select(0u, toEdge, isPending)), isPending);
      pendingCount += select(0u, 1u, isPending);
    }
    searchNetwork(edgeTarget(fromEdge), remaining, pendingCount);
    routes[routesOffset + index * ROUTE_STRIDE + K] = f32(min(droppedCount, 16777216u));
  }
  for (var to = 0u; to < K; to++) {
    let toEdge = getCandidateEdge(point, to);
    if (toEdge == NONE) {
      continue;
    }
    let toFraction = getCandidateFraction(point, to);
    var route = -1.0;
    if (toEdge == fromEdge && toFraction >= fromFraction) {
      route = (toFraction - fromFraction) * fromLength;
    } else if (remaining >= 0.0) {
      let network = lookupDistance(edgeSource(toEdge));
      if (network >= 0.0) {
        route = tail + network + toFraction * edgeLength(toEdge);
      }
    }
    routes[routesOffset + index * ROUTE_STRIDE + to] = route;
  }`;
}

/** Body of the forward Viterbi kernel: one thread per track. */
function getForwardBody(): string {
  return `let first = trackOffsets[trackOffsetsOffset + index];
  let last = trackOffsets[trackOffsetsOffset + index + 1u];
  let sigma = max(parameters[parametersOffset], 1.0e-6);
  let beta = max(parameters[parametersOffset + 1u], 1.0e-6);
  let emissionConstant = -log(sigma) - LOG_TWO_PI_HALF;
  let transitionConstant = -log(beta);
  for (var point = first; point < last; point++) {
    var newScore: array<f32, CANDIDATE_COUNT>;
    var newBack: array<u32, CANDIDATE_COUNT>;
    for (var slot = 0u; slot < K; slot++) {
      newScore[slot] = NEG;
      newBack[slot] = NONE;
    }
    var isReachable = false;
    var isBudgetLimited = false;
    if (point > first) {
      let previous = point - 1u;
      let straight = networkDistance(readPoint(point), readPoint(previous));
      for (var fromSlot = 0u; fromSlot < K; fromSlot++) {
        let fromEdge = getCandidateEdge(previous, fromSlot);
        let fromScore = scores[scoresOffset + previous * K + fromSlot];
        if (fromEdge == NONE || fromScore < -1.0e29) {
          continue;
        }
        let routeRow = routesOffset + (point * K + fromSlot) * ROUTE_STRIDE;
        isBudgetLimited = isBudgetLimited || routes[routeRow + K] > 0.0;
        for (var to = 0u; to < K; to++) {
          let route = routes[routeRow + to];
          if (route < 0.0) {
            continue;
          }
          let score = fromScore + transitionConstant - abs(straight - route) / beta;
          if (score > newScore[to]) {
            newScore[to] = score;
            newBack[to] = fromSlot;
            isReachable = true;
          }
        }
      }
    }
    for (var slot = 0u; slot < K; slot++) {
      if (getCandidateEdge(point, slot) == NONE) {
        newScore[slot] = NEG;
        newBack[slot] = NONE;
        continue;
      }
      let normalized = getCandidateDistance(point, slot) / sigma;
      let emission = emissionConstant - 0.5 * normalized * normalized;
      if (isReachable) {
        if (newScore[slot] > -1.0e29) {
          newScore[slot] += emission;
        }
      } else {
        // First point of the track or a break: restart from the emission alone. A break where a
        // route search dropped nodes is marked, since the missing route may be the budget's doing.
        newScore[slot] = emission;
        newBack[slot] = select(NONE, BUDGET_BREAK, isBudgetLimited);
      }
    }
    for (var slot = 0u; slot < K; slot++) {
      scores[scoresOffset + point * K + slot] = newScore[slot];
      backPointers[backPointersOffset + point * K + slot] = newBack[slot];
    }
  }`;
}

/** Body of the backward pass: one thread per track traces the best path of every unbroken stretch. */
function getBackwardBody(hasLogLikelihood: boolean): string {
  return `let first = trackOffsets[trackOffsetsOffset + index];
  let last = trackOffsets[trackOffsetsOffset + index + 1u];
  var logLikelihood = 0.0;
  var current = NONE;
  var isPicking = true;
  for (var step = 0u; step < last - first; step++) {
    let point = last - 1u - step;
    if (isPicking) {
      var bestScore = -1.0e29;
      current = NONE;
      for (var slot = 0u; slot < K; slot++) {
        let score = scores[scoresOffset + point * K + slot];
        if (score > bestScore) {
          bestScore = score;
          current = slot;
        }
      }
      if (current != NONE) {
        logLikelihood += bestScore;
      }
    }
    chosen[chosenOffset + point] = current;
    var isBreak = current == NONE;
    var isBudgetBreak = false;
    isPicking = true;
    if (current != NONE) {
      let previous = backPointers[backPointersOffset + point * K + current];
      if (previous < K) {
        current = previous;
        isPicking = false;
      } else {
        isBreak = true;
        isBudgetBreak = previous == BUDGET_BREAK;
      }
    }
    breaks[breaksOffset + point] = select(0u, select(1u, 2u, isBudgetBreak), isBreak && point > first);
  }
  ${hasLogLikelihood ? 'logLikelihoods[logLikelihoodsOffset + index] = logLikelihood;' : ''}`;
}

/** Returns every read-only view of a map-matching contributor. */
function getInputs(props: GPUMapMatchingProps): (GraphDataView | undefined)[] {
  return [
    props.points,
    props.trackOffsets,
    props.nodePositions,
    props.offsets,
    props.edgeTargets,
    props.parameters
  ];
}

/** Returns every writable view of a map-matching contributor. */
function getOutputs(props: GPUMapMatchingProps): (GraphDataView | undefined)[] {
  const {output} = props;
  return [
    output.matchedEdges,
    output.matchedFractions,
    output.matchedOffsets,
    output.snapDistances,
    output.snappedPositions,
    output.breaks,
    output.trackLogLikelihoods,
    output.matchedCount,
    output.breakCount,
    output.overflow,
    output.routeOverflow
  ];
}
