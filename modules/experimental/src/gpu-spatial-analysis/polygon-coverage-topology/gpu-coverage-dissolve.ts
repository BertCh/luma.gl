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
import {createWGSLKernelNode, getWGSLFloatLiteral} from '../../utils/wgsl-kernel-nodes';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../utils/gpu-contributor-utils';
import {
  getKeyGroupNodes,
  getKeyPairSortNodes,
  KEY_PAIR_INVALID
} from '../spatial-weights/key-pair-grouping';
import {getBoundedKeyPairSortNodes} from '../areal-interpolation/bounded-key-pair-sort';
import {GPUSegmentRingAssembly, type GPUSegmentRingAssemblyOutput} from '../ring-assembly/index';
import {createRingOrientationNode} from './coverage-topology-kernels';

const OPERATION = 'GPUCoverageDissolve';
const INVALID = `${KEY_PAIR_INVALID}u`;
/** Bit of a vertex's `next` entry that marks a ring whose interior is on the right. */
const FLIP_BIT = '0x80000000u';
/** Default vertex matching distance of the ring assembly after snapping. */
const DEFAULT_SNAPPED_VERTEX_TOLERANCE = 1e-6;

/**
 * Caller-owned outputs of {@link GPUCoverageDissolve}: the outputs of `GPUSegmentRingAssembly`
 * (rings, holes, shells, the GeoArrow `polygons` layout, `count`, `overflow`, ...) plus the number
 * of boundary segments. Request `polygons` with `sourceIds` to receive one polygon per
 * connected region together with its label, or `ringGroups` for the label of every ring.
 */
export type GPUCoverageDissolveOutput = GPUSegmentRingAssemblyOutput & {
  /**
   * Optional one-row count of directed boundary segments kept by the dissolve (ring edges whose
   * neighbor across the edge has a different label, or no neighbor). Useful to size the ring
   * and vertex capacity.
   */
  boundarySegmentCount?: GraphDataView<'uint32'>;
};

/** Properties for {@link GPUCoverageDissolve}. */
export type GPUCoverageDissolveProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'coverage-dissolve'`. */
  id?: string;
  /**
   * Flattened ring vertices (GeoArrow layout, as `GPUCoverageSimplification`). Planar, finite
   * coordinates; use tile- or view-local units because the output keeps f32 coordinates.
   */
  positions: GraphDataView<'float32x2'>;
  /** Ring-to-vertex offsets with a terminal entry. Rings close implicitly. */
  ringOffsets: GraphDataView<'uint32'>;
  /**
   * Polygon-to-ring offsets with a terminal entry. The first ring of each polygon is its shell,
   * the others are holes. Ring winding is free: orientation is read from the signed area.
   */
  polygonOffsets: GraphDataView<'uint32'>;
  /**
   * Group label of every polygon, `polygonOffsets.length - 1` rows. Polygons with equal labels
   * dissolve into one region (GeoPandas `dissolve(by=...)`). Labels are plain `uint32` values; map
   * strings to dense IDs on the CPU.
   */
  labels: GraphDataView<'uint32'>;
  /**
   * Vertex snap tolerance with the meaning of `GPUCoverageSimplification.snapTolerance`: two
   * vertices that snap to the same point are one point of the coverage, and the output uses the
   * coordinates of the lowest vertex index of that point. Defaults to `0` (exact f32 equality).
   */
  snapTolerance?: number;
  /**
   * Compile-time vertex matching distance of the ring assembly. Because every output vertex is a
   * canonical point, this only has to be positive and smaller than the distance between distinct
   * snapped points. Defaults to `snapTolerance / 4` when snapping, else `1e-6`.
   */
  vertexTolerance?: number;
  /** Compile-time. Split rings that touch themselves at a vertex into separate rings. Defaults to true. */
  splitTouchingRings?: boolean;
  /** Caller-owned outputs, bounded by their capacities with a GPU overflow flag. */
  output: GPUCoverageDissolveOutput;
};

/**
 * Dissolves a polygon coverage by label: GeoPandas `dissolve(by=..., method='coverage')` and
 * Shapely `coverage_union_all` per label.
 *
 * **Definition.**
 * - Vertices are snapped as in `GPUCoverageSimplification` and get dense point IDs; every ring edge
 *   gets the key `(min, max)` of its endpoint IDs. An edge is *internal* when another polygon with
 *   the same label holds an edge with the same key, and a *boundary edge* otherwise, which includes
 *   outer boundary and edges shared with a polygon of another label.
 * - Boundary edges are oriented with the interior on the left (a ring's orientation is read from
 *   its signed area, shells counter-clockwise and holes clockwise) and written, with the label as
 *   group, as the directed segments of `GPUSegmentRingAssembly`. Rings never mix labels.
 *   Assembly gives holes and shells (`ringIsHole`, `ringShells`, `polygons`).
 * - Vertices are kept at every node, as GEOS does: a boundary chain that two input polygons used
 *   to share keeps its collinear vertices.
 *
 * **Why this is bounded and not vector overlay.** A union of arbitrary polygons needs noding of
 * crossing edges and output of unpredictable size. A coverage has no crossings by definition: the
 * dissolved boundary is a subset of the input edges, so the output has at most as many vertices as
 * the input, and the exact-key edge match plus one ring assembly replace any geometry computation.
 * Input that is not a valid coverage (overlaps, edges that differ along a shared boundary) has no
 * defined result; check it with `GPUCoverageValidity` first. Like GEOS `CoverageUnion`, overlapping input
 * is not repaired.
 *
 * **Output.** Rings and polygons are in `GPUSegmentRingAssembly` order (deterministic: by lowest
 * boundary-segment index, which follows input vertex order). The label of a polygon is in
 * `output.polygons.sourceIds`. Ring and vertex capacity of the output are caller-set; the
 * input vertex count always suffices for vertices (plus one closing vertex per ring).
 *
 * **Cost.** Two sorts of `O(vertices)` keys, scans, a per-ring serial area walk and the assembly.
 */
export class GPUCoverageDissolve implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUCoverageDissolveProps;

  constructor(props: GPUCoverageDissolveProps) {
    const id = props.id ?? 'coverage-dissolve';
    this.id = id;
    this.props = props;
    validatePackedView(props.positions, ['float32x2'], `${id} positions`);
    if (props.positions.length < 1 || props.positions.length >= 2 ** 30) {
      throw new Error(`${id} positions must hold between 1 and 2^30 - 1 vertices`);
    }
    validatePackedUint32View(props.ringOffsets, `${id} ringOffsets`);
    validatePackedUint32View(props.polygonOffsets, `${id} polygonOffsets`);
    if (props.ringOffsets.length < 2) {
      throw new Error(`${id} ringOffsets must hold at least two entries`);
    }
    if (props.polygonOffsets.length < 2) {
      throw new Error(`${id} polygonOffsets must hold at least two entries`);
    }
    validatePackedUint32View(props.labels, `${id} labels`);
    if (props.labels.length !== props.polygonOffsets.length - 1) {
      throw new Error(`${id} labels length must equal the polygon count`);
    }
    if (
      props.snapTolerance !== undefined &&
      (!Number.isFinite(props.snapTolerance) || props.snapTolerance < 0)
    ) {
      throw new Error(`${id} snapTolerance must be a finite number >= 0`);
    }
    if (props.vertexTolerance !== undefined && !(props.vertexTolerance > 0)) {
      throw new Error(`${id} vertexTolerance must be positive`);
    }
    const {boundarySegmentCount, ...ringOutput} = props.output;
    if (boundarySegmentCount) {
      validatePackedUint32View(boundarySegmentCount, `${id} output.boundarySegmentCount`);
      if (boundarySegmentCount.length < 1) {
        throw new Error(`${id} output.boundarySegmentCount must hold one uint32`);
      }
    }
    validatePackedView(ringOutput.positions, ['float32x2'], `${id} output.positions`);
    validateGraphOutputsDisjointFromInputs(
      id,
      [
        ringOutput.ringOffsets,
        ringOutput.positions,
        ringOutput.ringAreas,
        ringOutput.ringIsHole,
        ringOutput.ringShells,
        ringOutput.ringGroups,
        ringOutput.polygons?.positions,
        ringOutput.polygons?.ringOffsets,
        ringOutput.polygons?.polygonOffsets,
        ringOutput.polygons?.featureOffsets,
        ringOutput.polygons?.sourceIds,
        ringOutput.count,
        ringOutput.overflow,
        ringOutput.requiredCount,
        ringOutput.openSegmentCount,
        ringOutput.touchingSegmentCount,
        boundarySegmentCount
      ],
      [props.positions, props.ringOffsets, props.polygonOffsets, props.labels]
    );
  }

  /** Returns the dissolve nodes in dependency order, ending with the ring assembly. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {positions, ringOffsets, polygonOffsets, labels} = props;
    const {boundarySegmentCount, ...ringOutput} = props.output;
    validateGraphViewsBelongToGraph(id, graph, [
      positions,
      ringOffsets,
      polygonOffsets,
      labels,
      boundarySegmentCount
    ]);
    const vertexCount = positions.length;
    const ringCount = ringOffsets.length - 1;
    const polygonCount = polygonOffsets.length - 1;
    const snapTolerance = props.snapTolerance ?? 0;
    const view = (name: string, length: number) =>
      createTransientView(graph, `${id}-${name}`, 'uint32', length);

    const ringFlip = view('ring-flip', ringCount);
    const vertexHigh = view('vertex-high', vertexCount);
    const vertexLow = view('vertex-low', vertexCount);
    const vertexPolygon = view('vertex-polygon', vertexCount);
    const vertexNext = view('vertex-next', vertexCount);

    const nodes: GPUCommandNode<Parameters>[] = [
      createRingOrientationNode<Parameters>(graph, {
        id: `${id}-ring-orientation`,
        operation: OPERATION,
        positions,
        ringOffsets,
        polygonOffsets,
        ringFlip
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-vertex-topology`,
        operation: OPERATION,
        variant: 'vertex-topology',
        bindings: [
          {name: 'positions', view: positions, type: 'f32', access: 'read'},
          {name: 'ringOffsets', view: ringOffsets, type: 'u32', access: 'read'},
          {name: 'polygonOffsets', view: polygonOffsets, type: 'u32', access: 'read'},
          {name: 'ringFlip', view: ringFlip, type: 'u32', access: 'read'},
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
  let inRing = index >= ringBegin && index < ringEnd &&
    ring >= polygonOffsets[polygonOffsetsOffset] && ring < polygonOffsets[polygonOffsetsOffset + POLYGON_COUNT] &&
    ring < polygonOffsets[polygonOffsetsOffset + polygon + 1u];
  let valid = inRing && isFiniteFloat(x) && isFiniteFloat(y);
  vertexHigh[vertexHighOffset + index] = select(${INVALID}, quantise(x), valid);
  vertexLow[vertexLowOffset + index] = select(${INVALID}, quantise(y), valid);
  vertexPolygon[vertexPolygonOffset + index] = select(${INVALID}, polygon, valid);
  let next = select(index + 1u, ringBegin, index + 1u == ringEnd);
  vertexNext[vertexNextOffset + index] = next | select(0u, ${FLIP_BIT}, ringFlip[ringFlipOffset + ring] != 0u);`
      })
    ];

    // Dense point IDs, and the canonical (lowest-index) coordinates of every point.
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
    const pointIds = view('point-ids', vertexCount);
    const canonicalPositions = createTransientView(
      graph,
      `${id}-canonical-positions`,
      'float32x2',
      vertexCount
    );
    const edgeHigh = view('edge-high', vertexCount);
    const edgeLow = view('edge-low', vertexCount);
    nodes.push(
      ...vertexSort.nodes,
      ...vertexGroups.nodes,
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-point-ids`,
        operation: OPERATION,
        variant: 'point-ids',
        bindings: [
          {name: 'sortedItems', view: vertexSort.sortedItems, type: 'u32', access: 'read'},
          {name: 'groupIndex', view: vertexGroups.groupIndex, type: 'u32', access: 'read'},
          {name: 'groupStarts', view: vertexGroups.groupStarts, type: 'u32', access: 'read'},
          {name: 'vertexPolygon', view: vertexPolygon, type: 'u32', access: 'read'},
          {name: 'positions', view: positions, type: 'f32', access: 'read'},
          {name: 'pointIds', view: pointIds, type: 'u32', access: 'read_write'},
          {name: 'canonicalPositions', view: canonicalPositions, type: 'f32', access: 'read_write'}
        ],
        invocationCount: vertexCount,
        body: `let vertex = sortedItems[sortedItemsOffset + index];
  let groupId = groupIndex[groupIndexOffset + index] - 1u;
  let valid = vertexPolygon[vertexPolygonOffset + vertex] != ${INVALID};
  let canonical = sortedItems[sortedItemsOffset + groupStarts[groupStartsOffset + groupId]];
  pointIds[pointIdsOffset + vertex] = select(${INVALID}, groupId, valid);
  let source = select(vertex, canonical, valid);
  canonicalPositions[canonicalPositionsOffset + vertex * 2u] = positions[positionsOffset + source * 2u];
  canonicalPositions[canonicalPositionsOffset + vertex * 2u + 1u] = positions[positionsOffset + source * 2u + 1u];`
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
          {name: 'edgeLow', view: edgeLow, type: 'u32', access: 'read_write'}
        ],
        invocationCount: vertexCount,
        body: `let first = pointIds[pointIdsOffset + index];
  let second = pointIds[pointIdsOffset + (vertexNext[vertexNextOffset + index] & 0x7fffffffu)];
  let valid = first != ${INVALID} && second != ${INVALID} && first != second &&
    vertexPolygon[vertexPolygonOffset + index] != ${INVALID};
  edgeHigh[edgeHighOffset + index] = select(${INVALID}, min(first, second), valid);
  edgeLow[edgeLowOffset + index] = select(${INVALID}, max(first, second), valid);`
      })
    );
    // Edge keys are dense point ids below `vertexCount`, so the radix passes need only
    // log2(vertexCount) bits per half instead of 32.
    const edgeSort = getBoundedKeyPairSortNodes(
      graph,
      `${id}-edge`,
      OPERATION,
      vertexCount,
      edgeHigh,
      edgeLow,
      {lowKeyLimit: vertexCount, highKeyLimit: vertexCount}
    );
    const edgeGroups = getKeyGroupNodes(
      graph,
      `${id}-edge`,
      OPERATION,
      vertexCount,
      edgeHigh,
      edgeLow,
      edgeSort.sortedItems
    );
    nodes.push(...edgeSort.nodes, ...edgeGroups.nodes);

    // Keep an edge unless another polygon with the same label holds the same edge.
    const keepFlag = view('keep-flag', vertexCount);
    const keepRank = view('keep-rank', vertexCount);
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-edge-keep`,
        operation: OPERATION,
        variant: 'edge-keep',
        bindings: [
          {name: 'sortedItems', view: edgeSort.sortedItems, type: 'u32', access: 'read'},
          {name: 'edgeHigh', view: edgeHigh, type: 'u32', access: 'read'},
          {name: 'vertexPolygon', view: vertexPolygon, type: 'u32', access: 'read'},
          {name: 'groupIndex', view: edgeGroups.groupIndex, type: 'u32', access: 'read'},
          {name: 'groupStarts', view: edgeGroups.groupStarts, type: 'u32', access: 'read'},
          {name: 'labels', view: labels, type: 'u32', access: 'read'},
          {name: 'keepFlag', view: keepFlag, type: 'u32', access: 'read_write'}
        ],
        invocationCount: vertexCount,
        body: `let item = sortedItems[sortedItemsOffset + index];
  let polygon = vertexPolygon[vertexPolygonOffset + item];
  var keep = polygon != ${INVALID} && edgeHigh[edgeHighOffset + item] != ${INVALID};
  if (keep) {
    let label = labels[labelsOffset + polygon];
    let groupId = groupIndex[groupIndexOffset + index] - 1u;
    let end = groupStarts[groupStartsOffset + groupId + 1u];
    for (var other = groupStarts[groupStartsOffset + groupId]; other < end; other++) {
      let otherPolygon = vertexPolygon[vertexPolygonOffset + sortedItems[sortedItemsOffset + other]];
      if (otherPolygon != ${INVALID} && otherPolygon != polygon && labels[labelsOffset + otherPolygon] == label) {
        keep = false;
      }
    }
  }
  keepFlag[keepFlagOffset + item] = select(0u, 1u, keep);`
      }),
      ...new GPUScan({
        id: `${id}-keep-scan`,
        input: keepFlag,
        output: keepRank,
        mode: 'exclusive'
      }).getCommandNodes(graph)
    );

    // Directed, canonical, interior-left segments with their label, then compaction.
    const segmentsFull = createTransientView(
      graph,
      `${id}-segments-full`,
      'float32x4',
      vertexCount
    );
    const groupsFull = view('groups-full', vertexCount);
    const segments = createTransientView(graph, `${id}-segments`, 'float32x4', vertexCount);
    const segmentGroups = view('segment-groups', vertexCount);
    const segmentCount = boundarySegmentCount ?? view('segment-count', 1);
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-segment-endpoints`,
        operation: OPERATION,
        variant: 'segment-endpoints',
        bindings: [
          {name: 'canonicalPositions', view: canonicalPositions, type: 'f32', access: 'read'},
          {name: 'vertexNext', view: vertexNext, type: 'u32', access: 'read'},
          {name: 'vertexPolygon', view: vertexPolygon, type: 'u32', access: 'read'},
          {name: 'labels', view: labels, type: 'u32', access: 'read'},
          {name: 'segmentsFull', view: segmentsFull, type: 'f32', access: 'read_write'},
          {name: 'groupsFull', view: groupsFull, type: 'u32', access: 'read_write'}
        ],
        invocationCount: vertexCount,
        body: `let entry = vertexNext[vertexNextOffset + index];
  let next = entry & 0x7fffffffu;
  let flip = (entry & ${FLIP_BIT}) != 0u;
  let startVertex = select(index, next, flip);
  let endVertex = select(next, index, flip);
  segmentsFull[segmentsFullOffset + index * 4u] = canonicalPositions[canonicalPositionsOffset + startVertex * 2u];
  segmentsFull[segmentsFullOffset + index * 4u + 1u] = canonicalPositions[canonicalPositionsOffset + startVertex * 2u + 1u];
  segmentsFull[segmentsFullOffset + index * 4u + 2u] = canonicalPositions[canonicalPositionsOffset + endVertex * 2u];
  segmentsFull[segmentsFullOffset + index * 4u + 3u] = canonicalPositions[canonicalPositionsOffset + endVertex * 2u + 1u];
  let polygon = vertexPolygon[vertexPolygonOffset + index];
  groupsFull[groupsFullOffset + index] = select(0u, labels[labelsOffset + polygon], polygon != ${INVALID});`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-compact`,
        operation: OPERATION,
        variant: 'compact',
        bindings: [
          {name: 'segmentsFull', view: segmentsFull, type: 'f32', access: 'read'},
          {name: 'groupsFull', view: groupsFull, type: 'u32', access: 'read'},
          {name: 'keepFlag', view: keepFlag, type: 'u32', access: 'read'},
          {name: 'keepRank', view: keepRank, type: 'u32', access: 'read'},
          {name: 'segments', view: segments, type: 'f32', access: 'read_write'},
          {name: 'segmentGroups', view: segmentGroups, type: 'u32', access: 'read_write'},
          {name: 'segmentCount', view: segmentCount, type: 'u32', access: 'read_write'}
        ],
        invocationCount: vertexCount,
        declarations: `const VERTEX_COUNT: u32 = ${vertexCount}u;`,
        body: `let flag = keepFlag[keepFlagOffset + index];
  let rank = keepRank[keepRankOffset + index];
  if (flag != 0u) {
    for (var component = 0u; component < 4u; component++) {
      segments[segmentsOffset + rank * 4u + component] = segmentsFull[segmentsFullOffset + index * 4u + component];
    }
    segmentGroups[segmentGroupsOffset + rank] = groupsFull[groupsFullOffset + index];
  }
  if (index == VERTEX_COUNT - 1u) {
    segmentCount[segmentCountOffset] = rank + flag;
  }`
      })
    );

    const vertexTolerance =
      props.vertexTolerance ??
      (snapTolerance > 0 ? snapTolerance / 4 : DEFAULT_SNAPPED_VERTEX_TOLERANCE);
    nodes.push(
      ...new GPUSegmentRingAssembly({
        id: `${id}-assembly`,
        endpoints: segments,
        count: segmentCount,
        groups: segmentGroups,
        vertexTolerance,
        interiorSide: 'left',
        geographic: false,
        splitTouchingRings: props.splitTouchingRings,
        output: ringOutput
      }).getCommandNodes(graph)
    );
    return nodes;
  }
}
