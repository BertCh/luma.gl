// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createSaturatedScanTotalNodes} from '../../utils/scan-total';
import {
  createTransientView,
  GPUScan,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import type {
  GPUContributorCapacityPlan,
  GPUCapacityPlan
} from '../../analysis-core/contracts/capacity-planning';
import type {GPUPolygonGeometryPort} from '../../analysis-core/contracts/index';
import {createWGSLKernelNode, getWGSLFloatLiteral} from '../../utils/wgsl-kernel-nodes';
import {getSortKeyBits} from '../../utils/sorted-segment-sums';
import {
  validateCapacity,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validatePackedView
} from '../../utils/gpu-contributor-utils';
import {
  getKeyGroupNodes,
  getKeyGroupTransientWordCount,
  getKeyPairSortNodes,
  KEY_PAIR_INVALID,
  KEY_PAIR_SORT_WORDS_PER_ITEM,
  type KeyGroups
} from './key-pair-grouping';
import {
  describeGPUSpatialWeights,
  getGPUSpatialWeightsCapacityPlan,
  getGPUSpatialWeightsStatusViews,
  getGPUSpatialWeightsViews,
  type GPUSpatialWeights,
  type GPUSpatialWeightsCapacityPlan,
  type GPUSpatialWeightsOutput,
  validateGPUSpatialWeightsOutput
} from './spatial-weights';
import {IS_FINITE_FLOAT_WGSL} from '../../utils/wgsl-common';

const OPERATION = 'GPUContiguityWeights';
const INVALID = `${KEY_PAIR_INVALID}u`;

/**
 * Default intermediate pair capacity per output slot. The pair list holds duplicates (one directed
 * pair per shared vertex or edge) before deduplication.
 */
export const GPU_CONTIGUITY_WEIGHTS_PAIR_CAPACITY_FACTOR = 4;

/** uint32 transients per vertex declared by the vertex stage (high, low, feature, next). */
const VERTEX_WORDS = 4;
/** uint32 transients per vertex declared by the rook edge stage (point ID, edge high/low/feature). */
const ROOK_EDGE_WORDS = 4;
/** uint32 transients per vertex for the pair counts and their scan. */
const PAIR_COUNT_WORDS = 2;
/** uint32 transients per pair slot: high, low, unique flag and unique rank. */
const PAIR_WORDS = 4;

/** Polygon contiguity criterion. */
export type GPUContiguityCriterion = 'queen' | 'rook';

/**
 * Properties for {@link GPUContiguityWeights}.
 *
 * Per-frame (no rebuild or recompile): the contents of the `polygons` views, as long as the
 * lengths stay the same. Compile-time: `criterion`, `snapTolerance`, view lengths and
 * `pairCapacity`.
 */
export type GPUContiguityWeightsProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'contiguity-weights'`. */
  id?: string;
  /** `'queen'`: share at least one vertex. `'rook'`: share at least one edge. */
  criterion: GPUContiguityCriterion;
  /**
   * Polygon or multipolygon features in the four-level layout. Output rows are the FEATURES
   * (`featureOffsets.length - 1`): every ring (shells and holes) of every part of a feature is a
   * vertex source of that feature, so two multipart features are neighbors when any of their parts
   * touch. Rings close implicitly (no repeated first vertex). `positions` must be one packed view.
   */
  polygons: GPUPolygonGeometryPort;
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
   * Caller-owned output CSR: `offsets` has `featureCount + 1` entries; the slot capacity is
   * `neighbors.length`. Binary weights (1) and ascending neighbor IDs; `distances` is not written.
   * `status.overflow` (required) is 1 when the pairs or the neighbors did not fit their capacity.
   * `status.requiredCount` (optional) is the unclamped distinct neighbor-slot total: exact unless
   * the intermediate pair list overflowed, which `status.candidateOverflow` (optional) reports;
   * it is then a lower bound.
   */
  output: GPUSpatialWeightsOutput;
  /**
   * Capacity of the intermediate directed feature-pair list, which holds duplicates when features
   * share several distinct vertices or edges before final deduplication. Repeated occurrences of
   * one feature in a single equal-key group are collapsed before pairs are emitted. Defaults to
   * `GPU_CONTIGUITY_WEIGHTS_PAIR_CAPACITY_FACTOR * output.neighbors.length`. Compile-time; larger
   * values cost sort time and memory.
   */
  pairCapacity?: number;
};

/** Read-back status of one {@link GPUContiguityWeights} run, for {@link GPUContiguityWeights.getCapacityPlan}. */
export type GPUContiguityWeightsObservation = {
  /** `output.status.requiredCount`. */
  requiredCount?: number;
  /** `output.status.candidateOverflow` was 1. */
  candidateOverflow?: boolean;
};

/** Slot and pair stages of the contiguity grow loop. */
export type GPUContiguityWeightsCapacityPlan = GPUContributorCapacityPlan & {
  /** Caller-owned CSR slots of the next allocation. */
  slots: GPUSpatialWeightsCapacityPlan;
  /** Intermediate pair list of the next build (`pairCapacity`). */
  pairs: GPUCapacityPlan;
};

/**
 * Polygon contiguity weights (the PySAL `Queen` / `Rook` equivalent) written as a
 * {@link GPUSpatialWeights} CSR whose rows are the polygon FEATURES.
 *
 * Definition, which the GPU result matches exactly (IDs, offsets, weights):
 * - Vertices are snapped as described by `snapTolerance`. Non-finite vertices are ignored.
 * - Queen: features `i != j` are neighbors when they have at least one snapped vertex in common.
 * - Rook: features `i != j` are neighbors when they have a ring edge in common, meaning both
 *   endpoints match after snapping, in either direction. Degenerate edges (both endpoints snapping
 *   to one point) never match.
 * - Every ring of every part of a feature, holes included, takes part. A feature is never its own
 *   neighbor (parts of one feature that touch add nothing), and the relation is symmetric.
 *   Weights are binary (1); each row lists ascending neighbor IDs. This is the PySAL result on
 *   MultiPolygon geometries, so points -> counts -> W -> Moran chain on one feature table.
 * - Capacity: slots past `output.neighbors.length` are dropped, offsets are clamped to the
 *   capacity so rows stay consistent, and `output.status.overflow` is set. Exceeding
 *   `pairCapacity` also sets `overflow` (and `candidateOverflow`) and drops pairs, so some rows
 *   can miss neighbors.
 *
 * Algorithm: vertices get a 64-bit key `(q_x, q_y)` and are sorted with two stable radix sorts, so
 * the grouping is exact (no hash collisions). Queen groups the vertices by key; rook assigns each
 * distinct point a dense ID, keys each ring edge by `(min, max)` of its endpoint IDs and groups the
 * edges the same way. Stable sorting keeps feature IDs together inside each equal-key group, so
 * repeated vertices or edges from one feature are collapsed before the group emits the directed
 * feature pairs `(i, j)` between its distinct features. The pair list is sorted by `(i, j)`,
 * deduplicated and compacted into CSR (a scan marks the unique pairs and one binary search per row
 * gives the CSR offsets). Output is deterministic. A point shared by `m` distinct features costs
 * `O(m^2)` pairs, which is fine for map data but not for thousands of features touching one point.
 *
 * Capacity grow loop: pass `output.status.requiredCount` and `candidateOverflow`, read them after
 * a frame and call {@link GPUContiguityWeights.getCapacityPlan}: when `overflow` is 1, reallocate
 * the slot views with `plan.slots.capacity` slots, pass `pairCapacity: plan.pairs.capacity` and
 * rebuild. The required count is exact once no pair overflow happens, so the loop converges after
 * at most a few rebuilds.
 *
 * {@link GPUContiguityWeights.outputWeights} is the caller-owned `output` with `information`
 * (row count, `transform: 'binary'`, `symmetric: true`, provenance `{source:
 * 'GPUContiguityWeights', criterion}`) and `status` attached; pass it to consumers so they can
 * gate on truncation.
 */
export class GPUContiguityWeights implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUContiguityWeightsProps;
  /** The output CSR with `information` and `status` attached, for consumers. */
  readonly outputWeights: GPUSpatialWeights;

  constructor(props: GPUContiguityWeightsProps) {
    const id = props.id ?? 'contiguity-weights';
    this.id = id;
    this.props = props;
    if (props.criterion !== 'queen' && props.criterion !== 'rook') {
      throw new Error(`${id} criterion must be 'queen' or 'rook'`);
    }
    const {polygons} = props;
    validatePackedView(polygons.positions, ['float32x2'], `${id} polygons.positions`);
    validateCapacity(id, 'polygons.positions length', polygons.positions.length, 1, 2 ** 31 - 1);
    for (const name of ['ringOffsets', 'polygonOffsets', 'featureOffsets'] as const) {
      validatePackedView(polygons[name], ['uint32'], `${id} polygons.${name}`);
      validateCapacity(id, `polygons.${name} length`, polygons[name].length, 2);
    }
    if (
      props.snapTolerance !== undefined &&
      (!Number.isFinite(props.snapTolerance) || props.snapTolerance < 0)
    ) {
      throw new Error(`${id} snapTolerance must be finite and >= 0`);
    }
    const featureCount = validateGPUSpatialWeightsOutput(id, props.output);
    if (featureCount !== polygons.featureOffsets.length - 1) {
      throw new Error(`${id} output rows must equal the feature count`);
    }
    if (props.pairCapacity !== undefined) {
      validateCapacity(id, 'pairCapacity', props.pairCapacity);
    }
    this.outputWeights = describeGPUSpatialWeights(props.output, {
      information: {
        rowCount: featureCount,
        transform: 'binary',
        symmetric: true,
        hasSelfWeights: false,
        provenance: {source: OPERATION, criterion: props.criterion}
      },
      status: props.output.status
    });
    validateGraphOutputsDisjointFromInputs(
      id,
      [
        ...getGPUSpatialWeightsViews(props.output),
        ...getGPUSpatialWeightsStatusViews(props.output)
      ],
      [polygons.positions, polygons.ringOffsets, polygons.polygonOffsets, polygons.featureOffsets]
    );
  }

  /** Number of features (output rows). */
  getFeatureCount(): number {
    return this.props.polygons.featureOffsets.length - 1;
  }

  /** Intermediate pair capacity this instance builds with. */
  getPairCapacity(): number {
    return (
      this.props.pairCapacity ??
      Math.max(1, this.props.output.neighbors.length * GPU_CONTIGUITY_WEIGHTS_PAIR_CAPACITY_FACTOR)
    );
  }

  /**
   * Plans the next allocation of the grow loop from a read-back observation (or, without one, the
   * current capacities). `slots` grows with `getGPUSpatialWeightsCapacityPlan`; `pairs` keeps the
   * default ratio to the slots, and doubles when `candidateOverflow` was reported.
   * `estimatedPeakTransientBytes` counts every transient this contributor declares (from the same
   * per-vertex and per-pair word constants as `getCommandNodes`); the scratch of the gpgpu sort
   * and scan primitives, which depends on the device, is not included.
   */
  getCapacityPlan(
    observation: GPUContiguityWeightsObservation = {}
  ): GPUContiguityWeightsCapacityPlan {
    const {props} = this;
    const capacity = props.output.neighbors.length;
    const slots = getGPUSpatialWeightsCapacityPlan({
      rowCount: this.getFeatureCount(),
      capacity,
      requiredCount: observation.requiredCount,
      distances: Boolean(props.output.distances)
    });
    const currentPairs = this.getPairCapacity();
    const scaledPairs =
      props.pairCapacity === undefined
        ? slots.capacity * GPU_CONTIGUITY_WEIGHTS_PAIR_CAPACITY_FACTOR
        : currentPairs;
    const pairCapacity = observation.candidateOverflow
      ? Math.max(scaledPairs, currentPairs * 2)
      : scaledPairs;
    const vertexCount = props.polygons.positions.length;
    const vertexWords = getContiguityVertexTransientWordCount(vertexCount, props.criterion);
    const pairWords = getContiguityPairTransientWordCount(pairCapacity);
    const pairs: GPUCapacityPlan = {
      capacity: pairCapacity,
      estimatedRequiredCount: pairCapacity,
      limited: false,
      estimatedOutputBytes: 0,
      estimatedPeakTransientBytes: pairWords * Uint32Array.BYTES_PER_ELEMENT
    };
    return {
      slots,
      pairs,
      estimatedWorkItems: vertexCount + pairCapacity + this.getFeatureCount() + 1,
      estimatedOutputBytes: slots.estimatedOutputBytes,
      estimatedPeakTransientBytes: (vertexWords + pairWords) * Uint32Array.BYTES_PER_ELEMENT
    };
  }

  /** Returns the contiguity nodes in dependency order. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {output} = props;
    const {ringOffsets, polygonOffsets, featureOffsets} = props.polygons;
    // The constructor validated `positions` as one packed float32x2 view.
    const positions = props.polygons.positions as GraphDataView<'float32x2'>;
    const {overflow, requiredCount, candidateOverflow} = output.status;
    validateGraphViewsBelongToGraph(id, graph, [
      positions,
      ringOffsets,
      polygonOffsets,
      featureOffsets,
      ...getGPUSpatialWeightsViews(output),
      ...getGPUSpatialWeightsStatusViews(output)
    ]);
    const vertexCount = positions.length;
    const featureCount = this.getFeatureCount();
    const polygonCount = polygonOffsets.length - 1;
    const ringCount = ringOffsets.length - 1;
    const capacity = output.neighbors.length;
    const pairCapacity = this.getPairCapacity();
    const snapTolerance = props.snapTolerance ?? 0;

    // VERTEX_WORDS views per vertex.
    const vertexHigh = createTransientView(graph, `${id}-vertex-high`, 'uint32', vertexCount);
    const vertexLow = createTransientView(graph, `${id}-vertex-low`, 'uint32', vertexCount);
    const vertexPolygon = createTransientView(graph, `${id}-vertex-feature`, 'uint32', vertexCount);
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
          {name: 'featureOffsets', view: featureOffsets, type: 'u32', access: 'read'},
          {name: 'vertexHigh', view: vertexHigh, type: 'u32', access: 'read_write'},
          {name: 'vertexLow', view: vertexLow, type: 'u32', access: 'read_write'},
          {name: 'vertexPolygon', view: vertexPolygon, type: 'u32', access: 'read_write'},
          {name: 'vertexNext', view: vertexNext, type: 'u32', access: 'read_write'}
        ],
        invocationCount: vertexCount,
        declarations: `const RING_COUNT: u32 = ${ringCount}u;
const POLYGON_COUNT: u32 = ${polygonCount}u;
const FEATURE_COUNT: u32 = ${featureCount}u;
const SNAP_TOLERANCE: f32 = ${getWGSLFloatLiteral(snapTolerance)};

${IS_FINITE_FLOAT_WGSL}

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
  var feature = 0u;
  high = FEATURE_COUNT;
  while (feature + 1u < high) {
    let middle = (feature + high) / 2u;
    if (featureOffsets[featureOffsetsOffset + middle] <= polygon) {
      feature = middle;
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
    polygon >= featureOffsets[featureOffsetsOffset] && polygon < featureOffsets[featureOffsetsOffset + FEATURE_COUNT] &&
    polygon < featureOffsets[featureOffsetsOffset + feature + 1u] &&
    isFiniteFloat(x) && isFiniteFloat(y);
  vertexHigh[vertexHighOffset + index] = select(${INVALID}, quantise(x), valid);
  vertexLow[vertexLowOffset + index] = select(${INVALID}, quantise(y), valid);
  vertexPolygon[vertexPolygonOffset + index] = select(${INVALID}, feature, valid);
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
    // `itemPolygon` holds the feature (output row) of each item.
    let itemPolygon = vertexPolygon;
    let itemSorted = vertexSort.sortedItems;
    let itemGroups: KeyGroups<Parameters> = vertexGroups;
    if (props.criterion === 'rook') {
      // ROOK_EDGE_WORDS views per vertex.
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

    // PAIR_COUNT_WORDS views per vertex, PAIR_WORDS views per pair slot.
    const pairCounts = createTransientView(graph, `${id}-pair-counts`, 'uint32', vertexCount);
    const pairStarts = createTransientView(graph, `${id}-pair-starts`, 'uint32', vertexCount);
    const pairTotal = createTransientView(graph, `${id}-pair-total`, 'uint32', 1);
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
    let begin = groupStarts[groupStartsOffset + groupId];
    let end = groupStarts[groupStartsOffset + groupId + 1u];
    var firstPolygonOccurrence = index == begin;
    if (!firstPolygonOccurrence) {
      let previousItem = sortedItems[sortedItemsOffset + index - 1u];
      firstPolygonOccurrence = itemPolygon[itemPolygonOffset + previousItem] != polygon;
    }
    if (firstPolygonOccurrence) {
      var previousOtherPolygon = ${INVALID};
      for (var other = begin; other < end; other++) {
        let otherPolygon = itemPolygon[itemPolygonOffset + sortedItems[sortedItemsOffset + other]];
        let firstOtherPolygonOccurrence = otherPolygon != previousOtherPolygon;
        previousOtherPolygon = otherPolygon;
        if (firstOtherPolygonOccurrence && otherPolygon != ${INVALID} && otherPolygon != polygon) {
          ${onPair}
        }
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
      ...createSaturatedScanTotalNodes<Parameters>(graph, {
        id: `${id}-pair`,
        operation: OPERATION,
        counts: pairCounts,
        offsets: pairStarts,
        rowCount: vertexCount,
        total: pairTotal
      }),
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
      // Feature IDs are below featureCount: a narrow radix sort, invalid keys still sort last.
      {high: getSortKeyBits(featureCount), low: getSortKeyBits(featureCount)}
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
          {name: 'offsets', view: output.offsets, type: 'u32', access: 'read_write'}
        ],
        invocationCount: featureCount + 1,
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
          {name: 'scanTotal', view: pairTotal, type: 'u32', access: 'read'},
          {name: 'overflow', view: overflow, type: 'u32', access: 'read_write'},
          ...(requiredCount
            ? [
                {
                  name: 'requiredCount',
                  view: requiredCount,
                  type: 'u32' as const,
                  access: 'read_write' as const
                }
              ]
            : []),
          ...(candidateOverflow
            ? [
                {
                  name: 'candidateOverflow',
                  view: candidateOverflow,
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
  let pairTotal = scanTotal[scanTotalOffset];
  overflow[overflowOffset] = select(0u, 1u, total > CAPACITY || pairTotal > PAIR_CAPACITY);
  ${requiredCount ? 'requiredCount[requiredCountOffset] = total;' : ''}
  ${candidateOverflow ? 'candidateOverflow[candidateOverflowOffset] = select(0u, 1u, pairTotal > PAIR_CAPACITY);' : ''}`
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
          {name: 'neighbors', view: output.neighbors, type: 'u32', access: 'read_write'},
          {name: 'weights', view: output.weights, type: 'f32', access: 'read_write'}
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

/**
 * uint32 transients the vertex stages of {@link GPUContiguityWeights} declare for `vertexCount`
 * vertices: the vertex keys, the key-pair sort and grouping, the rook edge keys and their sort and
 * grouping, and the per-vertex pair counts. The gpgpu sort and scan scratch is not included.
 */
function getContiguityVertexTransientWordCount(
  vertexCount: number,
  criterion: GPUContiguityCriterion
): number {
  const sortAndGroup =
    KEY_PAIR_SORT_WORDS_PER_ITEM * vertexCount + getKeyGroupTransientWordCount(vertexCount);
  const rook = criterion === 'rook' ? ROOK_EDGE_WORDS * vertexCount + sortAndGroup : 0;
  return VERTEX_WORDS * vertexCount + sortAndGroup + rook + PAIR_COUNT_WORDS * vertexCount;
}

/** uint32 transients the pair stages declare for `pairCapacity` pairs (gpgpu scratch excluded). */
function getContiguityPairTransientWordCount(pairCapacity: number): number {
  // The pair scan also declares one shared total word.
  return (PAIR_WORDS + KEY_PAIR_SORT_WORDS_PER_ITEM) * pairCapacity + 1;
}
