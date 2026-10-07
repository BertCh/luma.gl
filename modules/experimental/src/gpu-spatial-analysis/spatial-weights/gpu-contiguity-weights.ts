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
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {createWGSLKernelNode, getWGSLFloatLiteral} from '../../utils/wgsl-kernel-nodes';
import {getSortKeyBits} from '../../utils/sorted-segment-sums';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../utils/gpu-contributor-utils';
import {
  getKeyGroupNodes,
  getKeyPairSortNodes,
  KEY_PAIR_INVALID,
  type KeyGroups
} from './key-pair-grouping';
import {type GPUSpatialWeights, validateGPUSpatialWeights} from './spatial-weights';

const OPERATION = 'GPUContiguityWeights';
const INVALID = `${KEY_PAIR_INVALID}u`;

/** Polygon contiguity criterion. */
export type GPUContiguityCriterion = 'queen' | 'rook';

/**
 * Properties for {@link GPUContiguityWeights}.
 *
 * Per-frame (no rebuild or recompile): the contents of `positions`, `ringOffsets` and
 * `polygonOffsets`, as long as the lengths stay the same. Compile-time: `criterion`,
 * `snapTolerance`, view lengths and `pairCapacity`.
 */
export type GPUContiguityWeightsProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'contiguity-weights'`. */
  id?: string;
  /** `'queen'`: share at least one vertex. `'rook'`: share at least one edge. */
  criterion: GPUContiguityCriterion;
  /** Flattened vertices of every ring (GeoArrow layout, as `GPUPointInPolygonJoin`). */
  positions: GraphDataView<'float32x2'>;
  /** Ring-to-vertex offsets with a terminal entry. Rings close implicitly (no repeated first vertex). */
  ringOffsets: GraphDataView<'uint32'>;
  /**
   * Polygon-to-ring offsets with a terminal entry. Every ring (shells and holes) of a polygon
   * counts as a vertex source of that polygon. Output rows are these polygons.
   */
  polygonOffsets: GraphDataView<'uint32'>;
  /**
   * Vertex snap tolerance in coordinate units. Vertices are quantised with
   * `q = floor(x / snapTolerance + 0.5)` per axis (round half up, in f32) and two vertices are the
   * same point when both `q` match. `0` (the default) compares the exact f32 values (`-0` equals
   * `+0`). Quantisation is a grid snap, not a distance test: two points closer than the tolerance
   * can still fall in different cells, so choose a tolerance well below the feature size and
   * pre-snap data where exactness matters. `extent / snapTolerance` must stay below 2^24, the f32
   * integer range.
   */
  snapTolerance?: number;
  /**
   * Caller-owned output CSR: `offsets` has `polygonCount + 1` entries; the slot capacity is
   * `neighbors.length`. Binary weights (1) and ascending neighbor IDs; `distances` is not written.
   */
  weights: GPUSpatialWeights;
  /** Caller-owned one-row flag: 1 when the pairs or the neighbors did not fit their capacity, else 0. */
  overflow: GraphDataView<'uint32'>;
  /** Optional caller-owned one-row unclamped distinct neighbor-slot total (a lower bound after a pair overflow). */
  totalNeighbors?: GraphDataView<'uint32'>;
  /**
   * Capacity of the intermediate directed polygon-pair list, which holds duplicates (polygons that
   * share several vertices or edges appear several times before deduplication). Defaults to
   * `4 * weights.neighbors.length`. Compile-time; larger values cost sort time and memory.
   */
  pairCapacity?: number;
};

/**
 * Polygon contiguity weights (the PySAL `Queen` / `Rook` equivalent) written as a
 * {@link GPUSpatialWeights} CSR.
 *
 * Definition, which the GPU result matches exactly (IDs, offsets, weights):
 * - Vertices are snapped as described by `snapTolerance`. Non-finite vertices are ignored.
 * - Queen: polygons `i != j` are neighbors when they have at least one snapped vertex in common.
 * - Rook: polygons `i != j` are neighbors when they have a ring edge in common, meaning both
 *   endpoints match after snapping, in either direction. Degenerate edges (both endpoints snapping
 *   to one point) never match.
 * - Every ring of a polygon, holes included, takes part. A polygon is never its own neighbor, and
 *   the relation is symmetric. Weights are binary (1); each row lists ascending neighbor IDs.
 * - Capacity: slots past `weights.neighbors.length` are dropped, offsets are clamped to the
 *   capacity so rows stay consistent, and `overflow` is set. Exceeding `pairCapacity` also sets
 *   `overflow` and drops pairs, so some rows can miss neighbors.
 *
 * Algorithm: vertices get a 64-bit key `(q_x, q_y)` and are sorted with two stable radix sorts, so
 * the grouping is exact (no hash collisions). Queen groups the vertices by key; rook assigns each
 * distinct point a dense ID, keys each ring edge by `(min, max)` of its endpoint IDs and groups the
 * edges the same way. Every group of equal keys emits the directed polygon pairs `(i, j)` between
 * its distinct polygons; the pair list is sorted by `(i, j)`, deduplicated and compacted into CSR
 * (a scan marks the unique pairs and one binary search per row gives the CSR offsets). Output is deterministic. A point shared by `m`
 * polygons costs `O(m^2)` pairs, which is fine for map data but not for thousands of polygons
 * touching one point.
 *
 * Polygons are the unit of the result. MultiPolygon features must be flattened to one polygon
 * row per part, or merged afterwards.
 */
export class GPUContiguityWeights implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUContiguityWeightsProps;

  constructor(props: GPUContiguityWeightsProps) {
    const id = props.id ?? 'contiguity-weights';
    this.id = id;
    this.props = props;
    if (props.criterion !== 'queen' && props.criterion !== 'rook') {
      throw new Error(`${id} criterion must be 'queen' or 'rook'`);
    }
    validatePackedView(props.positions, ['float32x2'], `${id} positions`);
    if (props.positions.length < 1 || props.positions.length >= 2 ** 31) {
      throw new Error(`${id} positions must hold between 1 and 2^31 - 1 vertices`);
    }
    validatePackedUint32View(props.ringOffsets, `${id} ringOffsets`);
    validatePackedUint32View(props.polygonOffsets, `${id} polygonOffsets`);
    if (props.ringOffsets.length < 2) {
      throw new Error(`${id} ringOffsets must hold at least two entries`);
    }
    if (props.polygonOffsets.length < 2) {
      throw new Error(`${id} polygonOffsets must hold at least two entries`);
    }
    if (
      props.snapTolerance !== undefined &&
      (!Number.isFinite(props.snapTolerance) || props.snapTolerance < 0)
    ) {
      throw new Error(`${id} snapTolerance must be a finite number >= 0`);
    }
    const polygonCount = validateGPUSpatialWeights(id, props.weights);
    if (polygonCount !== props.polygonOffsets.length - 1) {
      throw new Error(`${id} weights.offsets length must equal polygonOffsets length`);
    }
    if (
      props.pairCapacity !== undefined &&
      (!Number.isInteger(props.pairCapacity) || props.pairCapacity < 1)
    ) {
      throw new Error(`${id} pairCapacity must be a positive integer`);
    }
    validatePackedUint32View(props.overflow, `${id} overflow`);
    if (props.overflow.length < 1) {
      throw new Error(`${id} overflow must hold one uint32`);
    }
    if (props.totalNeighbors) {
      validatePackedUint32View(props.totalNeighbors, `${id} totalNeighbors`);
      if (props.totalNeighbors.length < 1) {
        throw new Error(`${id} totalNeighbors must hold one uint32`);
      }
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      [
        props.weights.offsets,
        props.weights.neighbors,
        props.weights.weights,
        props.weights.distances,
        props.overflow,
        props.totalNeighbors
      ],
      [props.positions, props.ringOffsets, props.polygonOffsets]
    );
  }

  /** Number of polygons (output rows). */
  getPolygonCount(): number {
    return this.props.polygonOffsets.length - 1;
  }

  /** Returns the contiguity nodes in dependency order. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {positions, ringOffsets, polygonOffsets, weights} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      positions,
      ringOffsets,
      polygonOffsets,
      weights.offsets,
      weights.neighbors,
      weights.weights,
      weights.distances,
      props.overflow,
      props.totalNeighbors
    ]);
    const vertexCount = positions.length;
    const polygonCount = this.getPolygonCount();
    const ringCount = ringOffsets.length - 1;
    const capacity = weights.neighbors.length;
    const pairCapacity = props.pairCapacity ?? Math.max(1, capacity * 4);
    const snapTolerance = props.snapTolerance ?? 0;

    const vertexHigh = createTransientView(graph, `${id}-vertex-high`, 'uint32', vertexCount);
    const vertexLow = createTransientView(graph, `${id}-vertex-low`, 'uint32', vertexCount);
    const vertexPolygon = createTransientView(graph, `${id}-vertex-polygon`, 'uint32', vertexCount);
    const vertexNext = createTransientView(graph, `${id}-vertex-next`, 'uint32', vertexCount);

    const nodes: GPUCommandNode<Parameters>[] = [
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-vertex-topology`,
        operation: OPERATION,
        variant: 'vertex-topology',
        bindings: [
          {name: 'positions', view: positions, type: 'f32', access: 'read'},
          {name: 'ringOffsets', view: ringOffsets, type: 'u32', access: 'read'},
          {name: 'polygonOffsets', view: polygonOffsets, type: 'u32', access: 'read'},
          {name: 'vertexHigh', view: vertexHigh, type: 'u32', access: 'read_write'},
          {name: 'vertexLow', view: vertexLow, type: 'u32', access: 'read_write'},
          {name: 'vertexPolygon', view: vertexPolygon, type: 'u32', access: 'read_write'},
          {name: 'vertexNext', view: vertexNext, type: 'u32', access: 'read_write'}
        ],
        invocationCount: vertexCount,
        declarations: `const RING_COUNT: u32 = ${ringCount}u;
const POLYGON_COUNT: u32 = ${polygonCount}u;
const SNAP_TOLERANCE: f32 = ${getWGSLFloatLiteral(snapTolerance)};

fn isFiniteFloat(value: f32) -> bool {
  return (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u;
}

fn quantise(value: f32) -> u32 {
  if (SNAP_TOLERANCE <= 0.0) {
    return bitcast<u32>(select(value, 0.0, value == 0.0));
  }
  let snapped = clamp(floor(value / SNAP_TOLERANCE + 0.5), -2147483520.0, 2147483520.0);
  return bitcast<u32>(i32(snapped)) ^ 0x80000000u;
}`,
        body: `var ring = 0u;
  var high = RING_COUNT;
  while (ring + 1u < high) {
    let middle = (ring + high) / 2u;
    if (ringOffsets[ringOffsetsOffset + middle] <= index) {
      ring = middle;
    } else {
      high = middle;
    }
  }
  var polygon = 0u;
  high = POLYGON_COUNT;
  while (polygon + 1u < high) {
    let middle = (polygon + high) / 2u;
    if (polygonOffsets[polygonOffsetsOffset + middle] <= ring) {
      polygon = middle;
    } else {
      high = middle;
    }
  }
  let x = positions[positionsOffset + index * 2u];
  let y = positions[positionsOffset + index * 2u + 1u];
  let ringBegin = ringOffsets[ringOffsetsOffset + ring];
  let ringEnd = ringOffsets[ringOffsetsOffset + ring + 1u];
  let valid = index >= ringBegin && index < ringEnd &&
    ring >= polygonOffsets[polygonOffsetsOffset] && ring < polygonOffsets[polygonOffsetsOffset + POLYGON_COUNT] &&
    ring < polygonOffsets[polygonOffsetsOffset + polygon + 1u] &&
    isFiniteFloat(x) && isFiniteFloat(y);
  vertexHigh[vertexHighOffset + index] = select(${INVALID}, quantise(x), valid);
  vertexLow[vertexLowOffset + index] = select(${INVALID}, quantise(y), valid);
  vertexPolygon[vertexPolygonOffset + index] = select(${INVALID}, polygon, valid);
  vertexNext[vertexNextOffset + index] = select(index + 1u, ringBegin, index + 1u == ringEnd);`
      })
    ];

    const vertexSort = getKeyPairSortNodes(
      graph,
      `${id}-vertex`,
      OPERATION,
      vertexCount,
      vertexHigh,
      vertexLow
    );
    const vertexGroups = getKeyGroupNodes(
      graph,
      `${id}-vertex`,
      OPERATION,
      vertexCount,
      vertexHigh,
      vertexLow,
      vertexSort.sortedItems
    );
    nodes.push(...vertexSort.nodes, ...vertexGroups.nodes);

    // Items are what share a key: vertices for queen, ring edges (starting at the vertex) for rook.
    let itemPolygon = vertexPolygon;
    let itemSorted = vertexSort.sortedItems;
    let itemGroups: KeyGroups<Parameters> = vertexGroups;
    if (props.criterion === 'rook') {
      const pointIds = createTransientView(graph, `${id}-point-ids`, 'uint32', vertexCount);
      const edgeHigh = createTransientView(graph, `${id}-edge-high`, 'uint32', vertexCount);
      const edgeLow = createTransientView(graph, `${id}-edge-low`, 'uint32', vertexCount);
      itemPolygon = createTransientView(graph, `${id}-edge-polygon`, 'uint32', vertexCount);
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-point-ids`,
          operation: OPERATION,
          variant: 'point-ids',
          bindings: [
            {name: 'sortedItems', view: vertexSort.sortedItems, type: 'u32', access: 'read'},
            {name: 'groupIndex', view: vertexGroups.groupIndex, type: 'u32', access: 'read'},
            {name: 'vertexPolygon', view: vertexPolygon, type: 'u32', access: 'read'},
            {name: 'pointIds', view: pointIds, type: 'u32', access: 'read_write'}
          ],
          invocationCount: vertexCount,
          body: `let vertex = sortedItems[sortedItemsOffset + index];
  pointIds[pointIdsOffset + vertex] = select(${INVALID},
    groupIndex[groupIndexOffset + index] - 1u, vertexPolygon[vertexPolygonOffset + vertex] != ${INVALID});`
        }),
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-edge-keys`,
          operation: OPERATION,
          variant: 'edge-keys',
          bindings: [
            {name: 'vertexPolygon', view: vertexPolygon, type: 'u32', access: 'read'},
            {name: 'vertexNext', view: vertexNext, type: 'u32', access: 'read'},
            {name: 'pointIds', view: pointIds, type: 'u32', access: 'read'},
            {name: 'edgeHigh', view: edgeHigh, type: 'u32', access: 'read_write'},
            {name: 'edgeLow', view: edgeLow, type: 'u32', access: 'read_write'},
            {name: 'edgePolygon', view: itemPolygon, type: 'u32', access: 'read_write'}
          ],
          invocationCount: vertexCount,
          body: `let first = pointIds[pointIdsOffset + index];
  let second = pointIds[pointIdsOffset + vertexNext[vertexNextOffset + index]];
  let valid = first != ${INVALID} && second != ${INVALID} && first != second;
  edgeHigh[edgeHighOffset + index] = select(${INVALID}, min(first, second), valid);
  edgeLow[edgeLowOffset + index] = select(${INVALID}, max(first, second), valid);
  edgePolygon[edgePolygonOffset + index] = select(${INVALID}, vertexPolygon[vertexPolygonOffset + index], valid);`
        })
      );
      const edgeSort = getKeyPairSortNodes(
        graph,
        `${id}-edge`,
        OPERATION,
        vertexCount,
        edgeHigh,
        edgeLow,
        // Point IDs are below vertexCount: a narrow radix sort, invalid keys still sort last.
        {high: getSortKeyBits(vertexCount), low: getSortKeyBits(vertexCount)}
      );
      itemGroups = getKeyGroupNodes(
        graph,
        `${id}-edge`,
        OPERATION,
        vertexCount,
        edgeHigh,
        edgeLow,
        edgeSort.sortedItems
      );
      itemSorted = edgeSort.sortedItems;
      nodes.push(...edgeSort.nodes, ...itemGroups.nodes);
    }

    const pairCounts = createTransientView(graph, `${id}-pair-counts`, 'uint32', vertexCount);
    const pairStarts = createTransientView(graph, `${id}-pair-starts`, 'uint32', vertexCount);
    const pairHigh = createTransientView(graph, `${id}-pair-high`, 'uint32', pairCapacity);
    const pairLow = createTransientView(graph, `${id}-pair-low`, 'uint32', pairCapacity);
    const groupBindings = [
      {name: 'sortedItems', view: itemSorted, type: 'u32' as const, access: 'read' as const},
      {name: 'itemPolygon', view: itemPolygon, type: 'u32' as const, access: 'read' as const},
      {
        name: 'groupIndex',
        view: itemGroups.groupIndex,
        type: 'u32' as const,
        access: 'read' as const
      },
      {
        name: 'groupStarts',
        view: itemGroups.groupStarts,
        type: 'u32' as const,
        access: 'read' as const
      }
    ];
    const groupLoopWGSL = (onPair: string) => `let item = sortedItems[sortedItemsOffset + index];
  let polygon = itemPolygon[itemPolygonOffset + item];
  if (polygon != ${INVALID}) {
    let groupId = groupIndex[groupIndexOffset + index] - 1u;
    let end = groupStarts[groupStartsOffset + groupId + 1u];
    for (var other = groupStarts[groupStartsOffset + groupId]; other < end; other++) {
      let otherPolygon = itemPolygon[itemPolygonOffset + sortedItems[sortedItemsOffset + other]];
      if (otherPolygon != ${INVALID} && otherPolygon != polygon) {
        ${onPair}
      }
    }
  }`;
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-pair-counts`,
        operation: OPERATION,
        variant: 'pair-counts',
        bindings: [
          ...groupBindings,
          {name: 'pairCounts', view: pairCounts, type: 'u32', access: 'read_write'}
        ],
        invocationCount: vertexCount,
        body: `var count = 0u;
  ${groupLoopWGSL('count++;')}
  pairCounts[pairCountsOffset + index] = count;`
      }),
      ...new GPUScan({
        id: `${id}-pair-scan`,
        input: pairCounts,
        output: pairStarts,
        mode: 'exclusive'
      }).getCommandNodes(graph),
      // Both key halves of every pair slot start invalid; one pass instead of two fills.
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-pair-fill`,
        operation: OPERATION,
        variant: 'pair-fill',
        bindings: [
          {name: 'pairHigh', view: pairHigh, type: 'u32', access: 'read_write'},
          {name: 'pairLow', view: pairLow, type: 'u32', access: 'read_write'}
        ],
        invocationCount: pairCapacity,
        body: `pairHigh[pairHighOffset + index] = ${INVALID};
  pairLow[pairLowOffset + index] = ${INVALID};`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-pair-emit`,
        operation: OPERATION,
        variant: 'pair-emit',
        bindings: [
          ...groupBindings,
          {name: 'pairStarts', view: pairStarts, type: 'u32', access: 'read'},
          {name: 'pairHigh', view: pairHigh, type: 'u32', access: 'read_write'},
          {name: 'pairLow', view: pairLow, type: 'u32', access: 'read_write'}
        ],
        invocationCount: vertexCount,
        declarations: `const PAIR_CAPACITY: u32 = ${pairCapacity}u;`,
        body: `var next = pairStarts[pairStartsOffset + index];
  ${groupLoopWGSL(`if (next < PAIR_CAPACITY) {
          pairHigh[pairHighOffset + next] = polygon;
          pairLow[pairLowOffset + next] = otherPolygon;
        }
        next++;`)}`
      })
    );

    // Sort the directed pairs by (row, neighbor), keep one slot per distinct pair.
    const pairSort = getKeyPairSortNodes(
      graph,
      `${id}-pair`,
      OPERATION,
      pairCapacity,
      pairHigh,
      pairLow,
      // Polygon IDs are below polygonCount: a narrow radix sort, invalid keys still sort last.
      {high: getSortKeyBits(polygonCount), low: getSortKeyBits(polygonCount)}
    );
    const isUnique = createTransientView(graph, `${id}-unique`, 'uint32', pairCapacity);
    const uniqueRank = createTransientView(graph, `${id}-unique-rank`, 'uint32', pairCapacity);
    nodes.push(
      ...pairSort.nodes,
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-unique-flags`,
        operation: OPERATION,
        variant: 'unique-flags',
        bindings: [
          {name: 'sortedPairs', view: pairSort.sortedItems, type: 'u32', access: 'read'},
          {name: 'pairHigh', view: pairHigh, type: 'u32', access: 'read'},
          {name: 'pairLow', view: pairLow, type: 'u32', access: 'read'},
          {name: 'isUnique', view: isUnique, type: 'u32', access: 'read_write'}
        ],
        invocationCount: pairCapacity,
        body: `let item = sortedPairs[sortedPairsOffset + index];
  let high = pairHigh[pairHighOffset + item];
  let low = pairLow[pairLowOffset + item];
  var first = high != ${INVALID};
  if (first && index > 0u) {
    let previous = sortedPairs[sortedPairsOffset + index - 1u];
    first = high != pairHigh[pairHighOffset + previous] || low != pairLow[pairLowOffset + previous];
  }
  isUnique[isUniqueOffset + index] = select(0u, 1u, first);`
      }),
      ...new GPUScan({
        id: `${id}-unique-scan`,
        input: isUnique,
        output: uniqueRank,
        mode: 'exclusive'
      }).getCommandNodes(graph),
      // CSR offsets without a per-row count: the pairs are sorted by row, so row r starts at the
      // number of unique pairs before the first sorted pair whose row is at least r. One binary
      // search per row replaces the atomic row counts, their clear and the scan.
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-offsets`,
        operation: OPERATION,
        variant: 'offsets',
        bindings: [
          {name: 'sortedPairs', view: pairSort.sortedItems, type: 'u32', access: 'read'},
          {name: 'pairHigh', view: pairHigh, type: 'u32', access: 'read'},
          {name: 'isUnique', view: isUnique, type: 'u32', access: 'read'},
          {name: 'uniqueRank', view: uniqueRank, type: 'u32', access: 'read'},
          {name: 'offsets', view: weights.offsets, type: 'u32', access: 'read_write'}
        ],
        invocationCount: polygonCount + 1,
        declarations: `const CAPACITY: u32 = ${capacity}u;
const PAIR_CAPACITY: u32 = ${pairCapacity}u;`,
        body: `var low = 0u;
  var high = PAIR_CAPACITY;
  while (low < high) {
    let middle = (low + high) / 2u;
    if (pairHigh[pairHighOffset + sortedPairs[sortedPairsOffset + middle]] < index) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  var start = 0u;
  if (low < PAIR_CAPACITY) {
    start = uniqueRank[uniqueRankOffset + low];
  } else {
    start = uniqueRank[uniqueRankOffset + PAIR_CAPACITY - 1u] + isUnique[isUniqueOffset + PAIR_CAPACITY - 1u];
  }
  offsets[offsetsOffset + index] = min(start, CAPACITY);`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-totals`,
        operation: OPERATION,
        variant: 'totals',
        bindings: [
          {name: 'isUnique', view: isUnique, type: 'u32', access: 'read'},
          {name: 'uniqueRank', view: uniqueRank, type: 'u32', access: 'read'},
          {name: 'pairCounts', view: pairCounts, type: 'u32', access: 'read'},
          {name: 'pairStarts', view: pairStarts, type: 'u32', access: 'read'},
          {name: 'overflow', view: props.overflow, type: 'u32', access: 'read_write'},
          ...(props.totalNeighbors
            ? [
                {
                  name: 'totalNeighbors',
                  view: props.totalNeighbors,
                  type: 'u32' as const,
                  access: 'read_write' as const
                }
              ]
            : [])
        ],
        invocationCount: 1,
        declarations: `const CAPACITY: u32 = ${capacity}u;
const PAIR_CAPACITY: u32 = ${pairCapacity}u;
const VERTEX_COUNT: u32 = ${vertexCount}u;`,
        body: `let total = uniqueRank[uniqueRankOffset + PAIR_CAPACITY - 1u] + isUnique[isUniqueOffset + PAIR_CAPACITY - 1u];
  let pairTotal = pairStarts[pairStartsOffset + VERTEX_COUNT - 1u] + pairCounts[pairCountsOffset + VERTEX_COUNT - 1u];
  overflow[overflowOffset] = select(0u, 1u, total > CAPACITY || pairTotal > PAIR_CAPACITY);
  ${props.totalNeighbors ? 'totalNeighbors[totalNeighborsOffset] = total;' : ''}`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-emit`,
        operation: OPERATION,
        variant: 'emit',
        bindings: [
          {name: 'sortedPairs', view: pairSort.sortedItems, type: 'u32', access: 'read'},
          {name: 'pairLow', view: pairLow, type: 'u32', access: 'read'},
          {name: 'isUnique', view: isUnique, type: 'u32', access: 'read'},
          {name: 'uniqueRank', view: uniqueRank, type: 'u32', access: 'read'},
          {name: 'neighbors', view: weights.neighbors, type: 'u32', access: 'read_write'},
          {name: 'weights', view: weights.weights, type: 'f32', access: 'read_write'}
        ],
        invocationCount: pairCapacity,
        declarations: `const CAPACITY: u32 = ${capacity}u;`,
        body: `let rank = uniqueRank[uniqueRankOffset + index];
  if (isUnique[isUniqueOffset + index] != 0u && rank < CAPACITY) {
    neighbors[neighborsOffset + rank] = pairLow[pairLowOffset + sortedPairs[sortedPairsOffset + index]];
    weights[weightsOffset + rank] = 1.0;
  }`
      })
    );
    return nodes;
  }
}
