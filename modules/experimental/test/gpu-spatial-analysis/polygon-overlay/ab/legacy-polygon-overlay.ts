// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUScan,
  validatePackedUint32View,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {createWGSLKernelNode} from '../../../../src/utils/wgsl-kernel-nodes';
import {
  validateCapacity,
  validateGraphViewsBelongToGraph,
  validateMaximumCount,
  validatePackedView
} from '../../../../src/utils/gpu-contributor-utils';
import type {
  GPUBoundedResultStatusPort,
  GPUPackedGeometryPort,
  GPUPolygonGeometryPort
} from '../../../../src/analysis-core/contracts/index';
import {
  GPULineNoding,
  GPU_TOPOLOGY_NONE,
  type GPUNodedSegmentPort
} from '../../../../src/gpu-spatial-analysis/geometry-topology/index';
// Frozen A/B baseline: the default-device maxima (128 MiB bindings, 32-byte rows).
const GPU_LINE_SPLIT_MAXIMUM_PIECE_BOUND = 2 ** 22;
const GPU_SEGMENT_INTERSECTION_MAXIMUM_VERTEX_COUNT = 2 ** 22;
import {GPUSegmentRingAssembly} from '../../../../src/analysis-core/ring-assembly/index';
import {EXACT_ORIENTATION_WGSL} from '../../../../src/utils/exact-orientation-wgsl';
import {
  EXACT_GEOMETRY_WGSL,
  getExactRingWindingWGSL
} from '../../../../src/utils/exact-geometry-wgsl';
import {
  resolveGPUSpatialContext,
  type GPUSpatialContext
} from '../../../../src/analysis-core/contracts/spatial-context';

const OPERATION = 'GPUPolygonOverlay';

/**
 * Membership of the two sides of noded segment `index`: bit 0 for the point infinitesimally left of
 * its midpoint, bit 1 for the point infinitesimally right of it. The probes are symbolic
 * (`ExactProbe`) and take their side from the exact input edge the segment lies on
 * (`sourceSegments`), so a segment lying on an edge of the operand is labelled by exact signs rather
 * than by a rounded probe point a fraction of an ulp off the edge (review finding D-05).
 */
const MEMBERSHIP_BODY_WGSL = /* wgsl */ `var value = 0u;
  if (index < segmentCount[segmentCountOffset]) {
    let base = endpointsOffset + index * 4u;
    let a = vec2f(endpoints[base], endpoints[base + 1u]);
    let b = vec2f(endpoints[base + 2u], endpoints[base + 3u]);
    let midpoint = 0.5 * (a + b);
    // The side comes from the exact input edge the piece lies on: noded endpoints are f32-rounded
    // crossings, slightly off that edge, so their own signs would not be exact.
    let source = vec4f(
      sourceSegments[sourceSegmentsOffset + index * 4u], sourceSegments[sourceSegmentsOffset + index * 4u + 1u],
      sourceSegments[sourceSegmentsOffset + index * 4u + 2u], sourceSegments[sourceSegmentsOffset + index * 4u + 3u]);
    // One call site for the exact helpers: both probes run through the same loop.
    for (var probeIndex = 0u; probeIndex < 2u; probeIndex++) {
      let probe = ExactProbe(midpoint, source.xy, source.zw, select(-1, 1, probeIndex == 0u));
      if (insidePolygons(probe)) { value = value | (1u << probeIndex); }
    }
  }
  membership[membershipOffset + index] = value;`;

const FILL_RULES: readonly GPUPolygonOverlayFillRule[] = [
  'even-odd',
  'positive',
  'negative',
  'nonzero'
];

/** Boolean set operation applied by {@link GPUPolygonOverlay}. */
export type GPUPolygonOverlayOperation =
  | 'intersection'
  | 'union'
  | 'difference'
  | 'symmetric-difference'
  | 'dissolve';

/**
 * Rule that decides whether a probe point is inside an operand.
 *
 * - `'even-odd'` (default): a polygon contains the points inside its first (shell) ring and outside
 *   every other ring of the same polygon, using crossing parity per ring. This is the OGC
 *   shell-and-holes reading of valid polygons.
 * - `'positive'`, `'negative'` and `'nonzero'`: a point is inside when the signed sum of the
 *   winding numbers of ALL rings of one polygon is greater than, less than, or different from zero
 *   (counter-clockwise rings count +1, clockwise rings -1, computed with exact orientation and the
 *   same half-open crossing rule as the parity test, so the rules agree on parity). A point is
 *   inside the operand when it is inside any polygon. Use them for raw, self-overlapping boundaries
 *   such as offset curves, where regions covered twice must stay inside (`'even-odd'` would carve
 *   them out as holes).
 */
export type GPUPolygonOverlayFillRule = 'even-odd' | 'positive' | 'negative' | 'nonzero';

/** Explicit capacities for the bounded arrangement built by {@link GPUPolygonOverlay}. */
export type GPUPolygonOverlayCapacity = {
  /**
   * Segment-intersection events retained while noding both operands. The ring count of both
   * operands plus four times this capacity is at most {@link GPU_LINE_SPLIT_MAXIMUM_PIECE_BOUND},
   * and both operands together have at most {@link GPU_SEGMENT_INTERSECTION_MAXIMUM_VERTEX_COUNT}
   * vertex rows plus one per ring.
   */
  intersections: number;
  /** Atomic segments retained after noding. */
  nodedSegments: number;
};

/** Boundary provenance emitted before the selected edges are assembled into rings. */
export type GPUPolygonOverlayBoundaryPort = {
  endpoints: GraphDataView<'float32x4'>;
  /** Zero for the left operand and one for the right operand. */
  operandIds: GraphDataView<'uint32'>;
  /** Feature that owns the source edge within its operand. */
  sourceFeatureIds: GraphDataView<'uint32'>;
  status: GPUBoundedResultStatusPort & {
    requiredCount: GraphDataView<'uint32'>;
    candidateOverflow: GraphDataView<'uint32'>;
  };
};

/** Canonical, capacity-bounded polygon result of {@link GPUPolygonOverlay}. */
export type GPUPolygonOverlayOutput = {
  geometry: GPUPackedGeometryPort<GPUPolygonGeometryPort> & {
    sourceIds: GraphDataView<'uint32'>;
  };
  status: GPUBoundedResultStatusPort & {
    requiredCount: GraphDataView<'uint32'>;
    candidateOverflow: GraphDataView<'uint32'>;
    /**
     * Optional one-row count of segment pairs the intersection predicates could not certify.
     * They set `overflow` whether or not this view is given.
     */
    uncertainCount?: GraphDataView<'uint32'>;
  };
  boundary: GPUPolygonOverlayBoundaryPort;
};

type PackedPolygons = GPUPackedGeometryPort<GPUPolygonGeometryPort> & {
  /** Optional source feature ID per feature. */
  sourceIds?: GraphDataView<'uint32'>;
};

/** Properties for {@link GPUPolygonOverlay}. */
export type GPUPolygonOverlayProps = {
  id?: string;
  /**
   * Planar only (default: native planar coordinates). A longitude-latitude context throws at
   * construction: project positions to a planar frame first, for example with
   * `createGPULocalFrame` (`contracts/local-frame`) and `GPULocalFrameProjection`
   * (`geometry-measures`), and pass the frame's context.
   */
  spatialContext?: GPUSpatialContext;
  left: PackedPolygons;
  /** Required except for `dissolve`, which unions every polygon in `left`. */
  right?: PackedPolygons;
  operation: GPUPolygonOverlayOperation;
  /**
   * Inside rule applied to both operands. Default `'even-odd'`. Compile-time. See
   * {@link GPUPolygonOverlayFillRule}.
   */
  fillRule?: GPUPolygonOverlayFillRule;
  capacity: GPUPolygonOverlayCapacity;
  /** Chebyshev vertex identity tolerance for noding and ring assembly. */
  vertexTolerance: number;
  output: GPUPolygonOverlayOutput;
  spatialSort?: boolean;
  leafCapacity?: number;
};

function validatePolygons(id: string, name: string, polygons: PackedPolygons): void {
  if (polygons.kind !== 'polygons') {
    throw new Error(`${id} ${name}.kind must be polygons`);
  }
  validatePackedView(polygons.positions, ['float32x2'], `${id} ${name}.positions`);
  for (const [field, view] of [
    ['featureOffsets', polygons.featureOffsets],
    ['polygonOffsets', polygons.polygonOffsets],
    ['ringOffsets', polygons.ringOffsets]
  ] as const) {
    validatePackedUint32View(view, `${id} ${name}.${field}`);
    if (view.length < 2) {
      throw new Error(`${id} ${name}.${field} must contain at least two entries`);
    }
  }
  if (polygons.sourceIds) {
    validatePackedUint32View(polygons.sourceIds, `${id} ${name}.sourceIds`);
  }
  if (polygons.featureOffsets.length - 1 >= 0x80000000) {
    throw new Error(`${id} ${name} has too many features for packed overlay provenance`);
  }
}

function validateStatus(id: string, name: string, status: GPUPolygonOverlayOutput['status']): void {
  for (const [field, view] of Object.entries(status)) {
    if (view) {
      validatePackedUint32View(view, `${id} ${name}.${field}`);
      if (view.length < 1) {
        throw new Error(`${id} ${name}.${field} must contain one row`);
      }
    }
  }
}

/**
 * General polygon Boolean overlay over a bounded planar arrangement.
 *
 * Both operand boundaries are noded together. Each atomic segment samples the Boolean result on
 * its two sides; a segment is retained exactly when the result changes across it and is oriented
 * with the retained surface on its left. Ring assembly then produces compact GeoArrow-style
 * polygons, including holes. `dissolve` applies the same boundary rule to the union of `left`, so
 * shared and overlapping internal boundaries disappear rather than being returned as masked faces.
 *
 * Operand membership uses the `fillRule` (default even-odd shells and holes; `'positive'`,
 * `'negative'` and `'nonzero'` use exact winding numbers and keep regions a raw boundary covers
 * twice, which `GPUBufferSurface` relies on).
 *
 * Output and arrangement capacities are explicit. `candidateOverflow` means noding was incomplete;
 * `overflow` additionally covers boundary, ring or vertex capacity. Boundary provenance remains
 * aligned with the compact selected edge list. Side tests are exact: each noded segment is labelled
 * by the two symbolic points infinitesimally beside its midpoint, classified with exact orientation
 * signs and a Simulation-of-Simplicity tie rule, so a segment on an edge of the other operand gets
 * the right side regardless of `vertexTolerance` or coordinate magnitude. Crossing points are
 * rounded to f32 by noding (`GPULineNoding`), whose uncertain pairs and non-finite vertices set
 * `overflow`.
 *
 * `output.status.requiredCount` is the unclamped number of rings (shells plus holes) the assembly
 * needed, so callers can size the ring capacity of `output.geometry`;
 * `output.boundary.status.requiredCount` is the unclamped selected-edge count.
 */
export class LegacyGPUPolygonOverlay implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPUPolygonOverlayProps;
  /** Resolved spatial context (planar only). */
  readonly spatialContext: GPUSpatialContext;
  /** Resolved inside rule. */
  readonly fillRule: GPUPolygonOverlayFillRule;

  constructor(props: GPUPolygonOverlayProps) {
    this.id = props.id ?? 'polygon-overlay';
    this.spatialContext = resolveGPUSpatialContext(this.id, props.spatialContext);
    this.props = props;
    this.fillRule = props.fillRule ?? 'even-odd';
    const {id} = this;
    const {output} = props;
    validatePolygons(id, 'left', props.left);
    if (props.operation === 'dissolve') {
      if (props.right) {
        throw new Error(`${id} dissolve accepts only the left operand`);
      }
    } else if (!props.right) {
      throw new Error(`${id} ${props.operation} requires the right operand`);
    }
    if (props.right) {
      validatePolygons(id, 'right', props.right);
    }
    if (props.fillRule && !FILL_RULES.includes(props.fillRule)) {
      throw new Error(`${id} fillRule must be even-odd, positive, negative or nonzero`);
    }
    if (!(props.vertexTolerance > 0) || !Number.isFinite(props.vertexTolerance)) {
      throw new Error(`${id} vertexTolerance must be finite and positive`);
    }
    for (const [name, value] of Object.entries(props.capacity)) {
      validateCapacity(id, `capacity.${name}`, value);
    }
    // The operands are noded together as open lines, one repeated first vertex per ring, in one
    // segment table and one piece table.
    const ringCount =
      props.left.ringOffsets.length - 1 + (props.right ? props.right.ringOffsets.length - 1 : 0);
    validateMaximumCount(
      id,
      'left + right positions rows + rings',
      props.left.positions.length + (props.right?.positions.length ?? 0) + ringCount,
      GPU_SEGMENT_INTERSECTION_MAXIMUM_VERTEX_COUNT
    );
    validateMaximumCount(
      id,
      'left + right rings + 4 x capacity.intersections',
      ringCount + 4 * props.capacity.intersections,
      GPU_LINE_SPLIT_MAXIMUM_PIECE_BOUND
    );
    validatePolygons(id, 'output.geometry', output.geometry);
    validatePackedUint32View(output.geometry.sourceIds, `${id} output.geometry.sourceIds`);
    const ringCapacity = output.geometry.ringOffsets.length - 1;
    if (
      output.geometry.polygonOffsets.length !== ringCapacity + 1 ||
      output.geometry.featureOffsets.length !== ringCapacity + 1 ||
      output.geometry.sourceIds.length !== ringCapacity
    ) {
      throw new Error(`${id} output polygon offsets and source IDs must share one ring capacity`);
    }
    const boundary = output.boundary;
    validatePackedView(boundary.endpoints, ['float32x4'], `${id} output.boundary.endpoints`);
    for (const [name, view] of [
      ['operandIds', boundary.operandIds],
      ['sourceFeatureIds', boundary.sourceFeatureIds]
    ] as const) {
      validatePackedUint32View(view, `${id} output.boundary.${name}`);
      if (view.length !== boundary.endpoints.length) {
        throw new Error(`${id} output.boundary.${name} length must equal endpoints length`);
      }
    }
    validateStatus(id, 'output.status', output.status);
    validateStatus(id, 'output.boundary.status', boundary.status);
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {left, right, output} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      left.positions,
      left.featureOffsets,
      left.polygonOffsets,
      left.ringOffsets,
      left.sourceIds,
      right?.positions,
      right?.featureOffsets,
      right?.polygonOffsets,
      right?.ringOffsets,
      right?.sourceIds,
      output.geometry.positions,
      output.geometry.featureOffsets,
      output.geometry.polygonOffsets,
      output.geometry.ringOffsets,
      output.geometry.sourceIds,
      output.boundary.endpoints,
      output.boundary.operandIds,
      output.boundary.sourceFeatureIds,
      ...Object.values(output.boundary.status),
      ...Object.values(output.status)
    ]);

    const rightVertexCount = right?.positions.length ?? 0;
    const leftRingCount = left.ringOffsets.length - 1;
    const rightRingCount = right ? right.ringOffsets.length - 1 : 0;
    // Noding consumes open line paths, while polygon rings close implicitly. Add one repeated
    // first vertex per ring so the closing boundary edge participates in the arrangement.
    const leftExpandedVertexCount = left.positions.length + leftRingCount;
    const combinedVertexCount = leftExpandedVertexCount + rightVertexCount + rightRingCount;
    const combinedRingCount = leftRingCount + rightRingCount;
    const combinedPositions = createTransientView(
      graph,
      `${id}-combined-positions`,
      'float32x2',
      combinedVertexCount
    );
    const combinedOffsets = createTransientView(
      graph,
      `${id}-combined-offsets`,
      'uint32',
      combinedRingCount + 1
    );
    const combinedSources = createTransientView(
      graph,
      `${id}-combined-sources`,
      'uint32',
      combinedRingCount
    );
    const nodes: GPUCommandNode<Parameters>[] = [];
    const rightPositionBinding = right
      ? [
          {
            name: 'rightPositions',
            view: right.positions,
            type: 'f32' as const,
            access: 'read' as const
          }
        ]
      : [];
    const combinePositionsNode = createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-combine-positions`,
      operation: OPERATION,
      variant: 'combine-positions',
      bindings: [
        {name: 'leftPositions', view: left.positions, type: 'f32', access: 'read'},
        ...rightPositionBinding,
        {name: 'leftRingOffsets', view: left.ringOffsets, type: 'u32', access: 'read'},
        ...(right
          ? [
              {
                name: 'rightRingOffsets',
                view: right.ringOffsets,
                type: 'u32' as const,
                access: 'read' as const
              }
            ]
          : []),
        {name: 'combinedOffsets', view: combinedOffsets, type: 'u32', access: 'read'},
        {name: 'positions', view: combinedPositions, type: 'f32', access: 'read_write'}
      ],
      invocationCount: combinedVertexCount,
      declarations: `const LEFT_RING_COUNT: u32 = ${leftRingCount}u;
const RING_COUNT: u32 = ${combinedRingCount}u;`,
      body: `var combinedRing = 0u;
  var high = RING_COUNT;
  while (combinedRing + 1u < high) {
    let middle = (combinedRing + high) / 2u;
    if (combinedOffsets[combinedOffsetsOffset + middle] <= index) { combinedRing = middle; } else { high = middle; }
  }
  let fromLeft = combinedRing < LEFT_RING_COUNT;
  let ring = select(combinedRing - LEFT_RING_COUNT, combinedRing, fromLeft);
  let destinationBegin = combinedOffsets[combinedOffsetsOffset + combinedRing];
  let outputBase = positionsOffset + index * 2u;
  if (fromLeft) {
    let sourceBegin = leftRingOffsets[leftRingOffsetsOffset + ring];
    let sourceEnd = leftRingOffsets[leftRingOffsetsOffset + ring + 1u];
    let local = index - destinationBegin;
    let sourceRow = select(sourceBegin + local, sourceBegin, local >= sourceEnd - sourceBegin);
    let input = leftPositionsOffset + sourceRow * 2u;
    positions[outputBase] = leftPositions[input];
    positions[outputBase + 1u] = leftPositions[input + 1u];
  }${
    right
      ? ` else {
    let sourceBegin = rightRingOffsets[rightRingOffsetsOffset + ring];
    let sourceEnd = rightRingOffsets[rightRingOffsetsOffset + ring + 1u];
    let local = index - destinationBegin;
    let sourceRow = select(sourceBegin + local, sourceBegin, local >= sourceEnd - sourceBegin);
    let input = rightPositionsOffset + sourceRow * 2u;
    positions[outputBase] = rightPositions[input];
    positions[outputBase + 1u] = rightPositions[input + 1u];
  }`
      : ''
  }`
    });

    const rightTopologyBindings = right
      ? [
          {
            name: 'rightFeatureOffsets',
            view: right.featureOffsets,
            type: 'u32' as const,
            access: 'read' as const
          },
          {
            name: 'rightPolygonOffsets',
            view: right.polygonOffsets,
            type: 'u32' as const,
            access: 'read' as const
          },
          {
            name: 'rightRingOffsets',
            view: right.ringOffsets,
            type: 'u32' as const,
            access: 'read' as const
          }
        ]
      : [];
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-combine-topology`,
        operation: OPERATION,
        variant: 'combine-topology',
        bindings: [
          {name: 'leftFeatureOffsets', view: left.featureOffsets, type: 'u32', access: 'read'},
          {name: 'leftPolygonOffsets', view: left.polygonOffsets, type: 'u32', access: 'read'},
          {name: 'leftRingOffsets', view: left.ringOffsets, type: 'u32', access: 'read'},
          ...rightTopologyBindings,
          {name: 'offsets', view: combinedOffsets, type: 'u32', access: 'read_write'},
          {name: 'sources', view: combinedSources, type: 'u32', access: 'read_write'}
        ],
        invocationCount: combinedRingCount + 1,
        declarations: `const LEFT_RING_COUNT: u32 = ${leftRingCount}u;
const LEFT_EXPANDED_VERTEX_COUNT: u32 = ${leftExpandedVertexCount}u;
const LEFT_POLYGON_COUNT: u32 = ${left.polygonOffsets.length - 1}u;
const RIGHT_POLYGON_COUNT: u32 = ${right ? right.polygonOffsets.length - 1 : 0}u;`,
        body: `if (index <= LEFT_RING_COUNT) {
    offsets[offsetsOffset + index] = leftRingOffsets[leftRingOffsetsOffset + index] + index;
  }${
    right
      ? ` else {
    let rightRing = index - LEFT_RING_COUNT;
    offsets[offsetsOffset + index] = LEFT_EXPANDED_VERTEX_COUNT + rightRingOffsets[rightRingOffsetsOffset + rightRing] + rightRing;
  }`
      : ''
  }
  if (index >= ${combinedRingCount}u) { return; }
  let fromLeft = index < LEFT_RING_COUNT;
  let ring = select(index - LEFT_RING_COUNT, index, fromLeft);
  var polygon = 0u;
  var high = select(RIGHT_POLYGON_COUNT, LEFT_POLYGON_COUNT, fromLeft);
  while (polygon + 1u < high) {
    let middle = (polygon + high) / 2u;
    var start = 0u;
    if (fromLeft) { start = leftPolygonOffsets[leftPolygonOffsetsOffset + middle]; }
    ${right ? 'else { start = rightPolygonOffsets[rightPolygonOffsetsOffset + middle]; }' : ''}
    if (start <= ring) { polygon = middle; } else { high = middle; }
  }
  var feature = 0u;
  high = select(${right ? `${right.featureOffsets.length - 1}u` : '0u'}, ${left.featureOffsets.length - 1}u, fromLeft);
  while (feature + 1u < high) {
    let middle = (feature + high) / 2u;
    var start = 0u;
    if (fromLeft) { start = leftFeatureOffsets[leftFeatureOffsetsOffset + middle]; }
    ${right ? 'else { start = rightFeatureOffsets[rightFeatureOffsetsOffset + middle]; }' : ''}
    if (start <= polygon) { feature = middle; } else { high = middle; }
  }
  sources[sourcesOffset + index] = feature | select(0x80000000u, 0u, fromLeft);`
      })
    );
    nodes.push(combinePositionsNode);

    const noded = this._createNodedPort(graph);
    nodes.push(
      ...new GPULineNoding({
        id: `${id}-noding`,
        spatialContext: this.spatialContext,
        lines: {
          kind: 'lines',
          positions: combinedPositions,
          lineOffsets: combinedOffsets,
          sourceIds: combinedSources
        },
        intersectionCapacity: props.capacity.intersections,
        output: {
          ...noded,
          status: {...noded.status, uncertainCount: output.status.uncertainCount}
        },
        spatialSort: props.spatialSort,
        leafCapacity: props.leafCapacity
      }).getCommandNodes(graph)
    );

    const selected = createTransientView(
      graph,
      `${id}-selected`,
      'uint32',
      props.capacity.nodedSegments
    );
    const orientations = createTransientView(
      graph,
      `${id}-orientations`,
      'uint32',
      props.capacity.nodedSegments
    );
    const selectionScan = createTransientView(
      graph,
      `${id}-selection-scan`,
      'uint32',
      props.capacity.nodedSegments
    );
    const leftMembership = createTransientView(
      graph,
      `${id}-left-membership`,
      'uint32',
      props.capacity.nodedSegments
    );
    const rightMembership = right
      ? createTransientView(graph, `${id}-right-membership`, 'uint32', props.capacity.nodedSegments)
      : undefined;
    // Exact input edge of every noded segment, oriented like the segment: the side probes' reference
    // line. A zero-length segment or one without provenance gets its own endpoints (a zero-length
    // one then probes as a point, so both its sides classify alike and it is never selected).
    const sourceSegments = createTransientView(
      graph,
      `${id}-source-segments`,
      'float32x4',
      props.capacity.nodedSegments
    );
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-source-segments`,
        operation: OPERATION,
        variant: 'source-segments',
        bindings: [
          {name: 'segmentCount', view: noded.status.count, type: 'u32', access: 'read'},
          {name: 'endpoints', view: noded.endpoints, type: 'f32', access: 'read'},
          {name: 'sourceEdges', view: noded.sourceEdgeIds, type: 'u32', access: 'read'},
          {name: 'positions', view: combinedPositions, type: 'f32', access: 'read'},
          {name: 'sourceSegments', view: sourceSegments, type: 'f32', access: 'read_write'}
        ],
        invocationCount: props.capacity.nodedSegments,
        declarations: `const VERTEX_COUNT: u32 = ${combinedVertexCount}u;
const NONE: u32 = ${GPU_TOPOLOGY_NONE}u;
fn readCombined(row: u32) -> vec2f {
  return vec2f(positions[positionsOffset + row * 2u], positions[positionsOffset + row * 2u + 1u]);
}`,
        body: `var value = vec4f(0.0);
  if (index < segmentCount[segmentCountOffset]) {
    let base = endpointsOffset + index * 4u;
    let a = vec2f(endpoints[base], endpoints[base + 1u]);
    let b = vec2f(endpoints[base + 2u], endpoints[base + 3u]);
    value = vec4f(a, b);
    let edge = sourceEdges[sourceEdgesOffset + index];
    if (edge != NONE && edge + 1u < VERTEX_COUNT && any(a != b)) {
      let start = readCombined(edge);
      let end = readCombined(edge + 1u);
      if (any(start != end)) {
        // Pieces run along their edge, so the direction test is far from degenerate.
        value = select(vec4f(end, start), vec4f(start, end), dot(b - a, end - start) > 0.0);
      }
    }
  }
  sourceSegments[sourceSegmentsOffset + index * 4u] = value.x;
  sourceSegments[sourceSegmentsOffset + index * 4u + 1u] = value.y;
  sourceSegments[sourceSegmentsOffset + index * 4u + 2u] = value.z;
  sourceSegments[sourceSegmentsOffset + index * 4u + 3u] = value.w;`
      })
    );
    // Per-ring [minY, maxY] extents let the membership ray cast skip rings a probe cannot cross.
    const leftRingBounds = createTransientView(
      graph,
      `${id}-left-ring-bounds`,
      'float32',
      leftRingCount * 2
    );
    nodes.push(this._createRingBoundsNode(graph, 'left', left, leftRingBounds, leftRingCount));
    let rightRingBounds: GraphDataView<'float32'> | undefined;
    if (right) {
      rightRingBounds = createTransientView(
        graph,
        `${id}-right-ring-bounds`,
        'float32',
        rightRingCount * 2
      );
      nodes.push(
        this._createRingBoundsNode(graph, 'right', right, rightRingBounds, rightRingCount)
      );
    }
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-left-membership`,
        operation: OPERATION,
        variant: 'left-membership',
        bindings: [
          {name: 'segmentCount', view: noded.status.count, type: 'u32', access: 'read'},
          {name: 'endpoints', view: noded.endpoints, type: 'f32', access: 'read'},
          {name: 'sourceSegments', view: sourceSegments, type: 'f32', access: 'read'},
          {name: 'positions', view: left.positions, type: 'f32', access: 'read'},
          {name: 'polygonOffsets', view: left.polygonOffsets, type: 'u32', access: 'read'},
          {name: 'ringOffsets', view: left.ringOffsets, type: 'u32', access: 'read'},
          {name: 'ringBounds', view: leftRingBounds, type: 'f32', access: 'read'},
          {name: 'membership', view: leftMembership, type: 'u32', access: 'read_write'}
        ],
        invocationCount: props.capacity.nodedSegments,
        declarations: this._getMembershipWGSL(left.polygonOffsets.length - 1, this.fillRule),
        body: MEMBERSHIP_BODY_WGSL
      })
    );
    if (right && rightMembership && rightRingBounds) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-right-membership`,
          operation: OPERATION,
          variant: 'right-membership',
          bindings: [
            {name: 'segmentCount', view: noded.status.count, type: 'u32', access: 'read'},
            {name: 'endpoints', view: noded.endpoints, type: 'f32', access: 'read'},
            {name: 'sourceSegments', view: sourceSegments, type: 'f32', access: 'read'},
            {name: 'positions', view: right.positions, type: 'f32', access: 'read'},
            {name: 'polygonOffsets', view: right.polygonOffsets, type: 'u32', access: 'read'},
            {name: 'ringOffsets', view: right.ringOffsets, type: 'u32', access: 'read'},
            {name: 'ringBounds', view: rightRingBounds, type: 'f32', access: 'read'},
            {name: 'membership', view: rightMembership, type: 'u32', access: 'read_write'}
          ],
          invocationCount: props.capacity.nodedSegments,
          declarations: this._getMembershipWGSL(right.polygonOffsets.length - 1, this.fillRule),
          body: MEMBERSHIP_BODY_WGSL
        })
      );
    }
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-classify-boundary`,
        operation: OPERATION,
        variant: props.operation,
        bindings: [
          {name: 'leftMembership', view: leftMembership, type: 'u32', access: 'read'},
          ...(rightMembership
            ? [
                {
                  name: 'rightMembership',
                  view: rightMembership,
                  type: 'u32' as const,
                  access: 'read' as const
                }
              ]
            : []),
          {name: 'selected', view: selected, type: 'u32', access: 'read_write'},
          {name: 'orientations', view: orientations, type: 'u32', access: 'read_write'}
        ],
        invocationCount: props.capacity.nodedSegments,
        body: `let leftBits = leftMembership[leftMembershipOffset + index];
  let rightBits = ${rightMembership ? 'rightMembership[rightMembershipOffset + index]' : '0u'};
  let leftA = (leftBits & 1u) != 0u;
  let rightA = (leftBits & 2u) != 0u;
  let leftB = (rightBits & 1u) != 0u;
  let rightB = (rightBits & 2u) != 0u;
  let leftResult = ${this._getOperationExpression('leftA', 'leftB')};
  let rightResult = ${this._getOperationExpression('rightA', 'rightB')};
  let keep = leftResult != rightResult;
  let reverse = keep && rightResult;
  selected[selectedOffset + index] = select(0u, 1u, keep);
  orientations[orientationsOffset + index] = select(0u, 1u, reverse);`
      }),
      ...new GPUScan({
        id: `${id}-selection-scan`,
        input: selected,
        output: selectionScan,
        mode: 'inclusive'
      }).getCommandNodes(graph)
    );

    const boundary = output.boundary;
    const boundaryGroups = createTransientView(
      graph,
      `${id}-boundary-groups`,
      'uint32',
      boundary.endpoints.length
    );
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-emit-boundary`,
        operation: OPERATION,
        variant: 'emit-boundary',
        bindings: [
          {name: 'selected', view: selected, type: 'u32', access: 'read'},
          {name: 'scan', view: selectionScan, type: 'u32', access: 'read'},
          {name: 'orientations', view: orientations, type: 'u32', access: 'read'},
          {name: 'sourceEndpoints', view: noded.endpoints, type: 'f32', access: 'read'},
          {name: 'endpoints', view: boundary.endpoints, type: 'f32', access: 'read_write'},
          {name: 'groups', view: boundaryGroups, type: 'u32', access: 'read_write'}
        ],
        invocationCount: props.capacity.nodedSegments,
        declarations: `const BOUNDARY_CAPACITY: u32 = ${boundary.endpoints.length}u;`,
        body: `if (selected[selectedOffset + index] == 0u) { return; }
  let outputRow = scan[scanOffset + index] - 1u;
  if (outputRow >= BOUNDARY_CAPACITY) { return; }
  let input = sourceEndpointsOffset + index * 4u;
  let outputBase = endpointsOffset + outputRow * 4u;
  let reverse = orientations[orientationsOffset + index] != 0u;
  endpoints[outputBase] = sourceEndpoints[input + select(0u, 2u, reverse)];
  endpoints[outputBase + 1u] = sourceEndpoints[input + select(1u, 3u, reverse)];
  endpoints[outputBase + 2u] = sourceEndpoints[input + select(2u, 0u, reverse)];
  endpoints[outputBase + 3u] = sourceEndpoints[input + select(3u, 1u, reverse)];
  groups[groupsOffset + outputRow] = 0u;`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-emit-boundary-provenance`,
        operation: OPERATION,
        variant: 'emit-boundary-provenance',
        bindings: [
          {name: 'selected', view: selected, type: 'u32', access: 'read'},
          {name: 'scan', view: selectionScan, type: 'u32', access: 'read'},
          {name: 'packedSources', view: noded.sourceFeatureIds, type: 'u32', access: 'read'},
          {name: 'operandIds', view: boundary.operandIds, type: 'u32', access: 'read_write'},
          {
            name: 'sourceFeatureIds',
            view: boundary.sourceFeatureIds,
            type: 'u32',
            access: 'read_write'
          }
        ],
        invocationCount: props.capacity.nodedSegments,
        declarations: `const BOUNDARY_CAPACITY: u32 = ${boundary.endpoints.length}u;`,
        body: `if (selected[selectedOffset + index] == 0u) { return; }
  let outputRow = scan[scanOffset + index] - 1u;
  if (outputRow >= BOUNDARY_CAPACITY) { return; }
  let packed = packedSources[packedSourcesOffset + index];
  operandIds[operandIdsOffset + outputRow] = packed >> 31u;
  sourceFeatureIds[sourceFeatureIdsOffset + outputRow] = packed & 0x7fffffffu;`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-boundary-status`,
        operation: OPERATION,
        variant: 'boundary-status',
        bindings: [
          {name: 'scan', view: selectionScan, type: 'u32', access: 'read'},
          {name: 'nodingOverflow', view: noded.status.overflow, type: 'u32', access: 'read'},
          {
            name: 'nodingCandidateOverflow',
            view: noded.status.candidateOverflow,
            type: 'u32',
            access: 'read'
          },
          {name: 'count', view: boundary.status.count, type: 'u32', access: 'read_write'},
          {
            name: 'requiredCount',
            view: boundary.status.requiredCount,
            type: 'u32',
            access: 'read_write'
          },
          {name: 'overflow', view: boundary.status.overflow, type: 'u32', access: 'read_write'},
          {
            name: 'candidateOverflow',
            view: boundary.status.candidateOverflow,
            type: 'u32',
            access: 'read_write'
          }
        ],
        invocationCount: 1,
        declarations: `const BOUNDARY_CAPACITY: u32 = ${boundary.endpoints.length}u;
const NODED_CAPACITY: u32 = ${props.capacity.nodedSegments}u;`,
        body: `let required = scan[scanOffset + NODED_CAPACITY - 1u];
  let candidateIncomplete = nodingCandidateOverflow[nodingCandidateOverflowOffset];
  requiredCount[requiredCountOffset] = required;
  count[countOffset] = min(required, BOUNDARY_CAPACITY);
  overflow[overflowOffset] = select(0u, 1u, required > BOUNDARY_CAPACITY || nodingOverflow[nodingOverflowOffset] != 0u);
  candidateOverflow[candidateOverflowOffset] = candidateIncomplete;`
      })
    );

    const ringCapacity = output.geometry.ringOffsets.length - 1;
    const ringPositions = createTransientView(
      graph,
      `${id}-ring-positions`,
      'float32x2',
      output.geometry.positions.length
    );
    const ringOffsets = createTransientView(
      graph,
      `${id}-ring-offsets`,
      'uint32',
      ringCapacity + 1
    );
    const ringCount = createTransientView(graph, `${id}-ring-count`, 'uint32', 1);
    const ringRequiredCount = createTransientView(graph, `${id}-ring-required-count`, 'uint32', 1);
    const ringOverflow = createTransientView(graph, `${id}-ring-overflow`, 'uint32', 1);
    nodes.push(
      ...new GPUSegmentRingAssembly({
        id: `${id}-assemble`,
        spatialContext: this.spatialContext,
        endpoints: boundary.endpoints,
        count: boundary.status.count,
        groups: boundaryGroups,
        vertexTolerance: props.vertexTolerance,
        interiorSide: 'left',
        normalizeWinding: true,
        cancelOpposingSegments: true,
        splitTouchingRings: true,
        output: {
          positions: ringPositions,
          ringOffsets,
          status: {count: ringCount, requiredCount: ringRequiredCount, overflow: ringOverflow},
          polygons: output.geometry
        }
      }).getCommandNodes(graph)
    );
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-source-ids`,
        operation: OPERATION,
        variant: 'source-ids',
        bindings: [
          {
            name: 'featureOffsets',
            view: output.geometry.featureOffsets,
            type: 'u32',
            access: 'read'
          },
          {name: 'sourceIds', view: output.geometry.sourceIds, type: 'u32', access: 'read_write'}
        ],
        invocationCount: Math.max(1, output.geometry.sourceIds.length),
        declarations: `const RING_CAPACITY: u32 = ${ringCapacity}u;
const SOURCE_CAPACITY: u32 = ${output.geometry.sourceIds.length}u;
const NONE: u32 = ${GPU_TOPOLOGY_NONE}u;`,
        body: `let featureCount = featureOffsets[featureOffsetsOffset + RING_CAPACITY];
  if (index < SOURCE_CAPACITY) {
    sourceIds[sourceIdsOffset + index] = select(NONE, index, index < featureCount);
  }`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-publish-required`,
        operation: OPERATION,
        variant: 'publish-required',
        bindings: [
          {name: 'ringRequiredCount', view: ringRequiredCount, type: 'u32', access: 'read'},
          {
            name: 'requiredCount',
            view: output.status.requiredCount,
            type: 'u32',
            access: 'read_write'
          }
        ],
        invocationCount: 1,
        // Unclamped ring requirement of the assembly (shells plus holes), for sizing capacity.
        body: 'requiredCount[requiredCountOffset] = ringRequiredCount[ringRequiredCountOffset];'
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-publish`,
        operation: OPERATION,
        variant: 'publish',
        bindings: [
          {
            name: 'featureOffsets',
            view: output.geometry.featureOffsets,
            type: 'u32',
            access: 'read'
          },
          {name: 'boundaryOverflow', view: boundary.status.overflow, type: 'u32', access: 'read'},
          {
            name: 'candidateOverflowInput',
            view: boundary.status.candidateOverflow,
            type: 'u32',
            access: 'read'
          },
          {name: 'ringOverflow', view: ringOverflow, type: 'u32', access: 'read'},
          {name: 'count', view: output.status.count, type: 'u32', access: 'read_write'},
          {name: 'overflow', view: output.status.overflow, type: 'u32', access: 'read_write'},
          {
            name: 'candidateOverflow',
            view: output.status.candidateOverflow,
            type: 'u32',
            access: 'read_write'
          }
        ],
        invocationCount: 1,
        declarations: `const RING_CAPACITY: u32 = ${ringCapacity}u;`,
        body: `let featureCount = featureOffsets[featureOffsetsOffset + RING_CAPACITY];
  let incomplete = boundaryOverflow[boundaryOverflowOffset] | ringOverflow[ringOverflowOffset];
  count[countOffset] = featureCount;
  overflow[overflowOffset] = incomplete;
  candidateOverflow[candidateOverflowOffset] = candidateOverflowInput[candidateOverflowInputOffset];`
      })
    );
    return nodes;
  }

  private _createRingBoundsNode<Parameters>(
    graph: GPUCommandGraph<Parameters>,
    side: 'left' | 'right',
    polygons: PackedPolygons,
    bounds: GraphDataView<'float32'>,
    ringCount: number
  ): GPUCommandNode<Parameters> {
    return createWGSLKernelNode<Parameters>(graph, {
      id: `${this.id}-${side}-ring-bounds`,
      operation: OPERATION,
      variant: `${side}-ring-bounds`,
      bindings: [
        {name: 'positions', view: polygons.positions, type: 'f32', access: 'read'},
        {name: 'ringOffsets', view: polygons.ringOffsets, type: 'u32', access: 'read'},
        {name: 'bounds', view: bounds, type: 'f32', access: 'read_write'}
      ],
      invocationCount: Math.max(1, ringCount),
      declarations: `const RING_COUNT: u32 = ${ringCount}u;`,
      body: `if (index >= RING_COUNT) { return; }
  let begin = ringOffsets[ringOffsetsOffset + index];
  let end = ringOffsets[ringOffsetsOffset + index + 1u];
  var lowY = 3.4e38;
  var highY = -3.4e38;
  for (var row = begin; row < end; row++) {
    let y = positions[positionsOffset + row * 2u + 1u];
    lowY = min(lowY, y);
    highY = max(highY, y);
  }
  bounds[boundsOffset + index * 2u] = lowY;
  bounds[boundsOffset + index * 2u + 1u] = highY;`
    });
  }

  private _createNodedPort<Parameters>(graph: GPUCommandGraph<Parameters>): GPUNodedSegmentPort {
    const {id, props} = this;
    const rows = props.capacity.nodedSegments;
    const scalar = (name: string) => createTransientView(graph, `${id}-${name}`, 'uint32', 1);
    return {
      endpoints: createTransientView(graph, `${id}-noded-endpoints`, 'float32x4', rows),
      sourceFeatureIds: createTransientView(graph, `${id}-noded-features`, 'uint32', rows),
      sourceRingIds: createTransientView(graph, `${id}-noded-rings`, 'uint32', rows),
      sourceEdgeIds: createTransientView(graph, `${id}-noded-edges`, 'uint32', rows),
      sourceStartParameters: createTransientView(graph, `${id}-noded-start`, 'float32', rows),
      sourceEndParameters: createTransientView(graph, `${id}-noded-end`, 'float32', rows),
      status: {
        count: scalar('noded-count'),
        requiredCount: scalar('noded-required-count'),
        overflow: scalar('noded-overflow'),
        candidateOverflow: scalar('noded-candidate-overflow')
      }
    };
  }

  private _getOperationExpression(left: string, right: string): string {
    switch (this.props.operation) {
      case 'intersection':
        return `${left} && ${right}`;
      case 'union':
        return `${left} || ${right}`;
      case 'difference':
        return `${left} && !${right}`;
      case 'symmetric-difference':
        return `${left} != ${right}`;
      case 'dissolve':
        return left;
    }
  }

  /**
   * Operand membership of an `ExactProbe` under the fill rule. Every crossing is an exact sign
   * (`utils/exact-geometry-wgsl.ts`): an upward edge counts +1 when the probe is left of it, a
   * downward edge -1 when it is right of it, with the Simulation-of-Simplicity tie rule.
   * `'even-odd'`: inside the shell's parity and outside every hole's parity. The winding rules sum
   * the winding numbers of all rings of one polygon.
   */
  private _getMembershipWGSL(polygonCount: number, fillRule: GPUPolygonOverlayFillRule): string {
    const evenOdd = fillRule === 'even-odd';
    const test =
      fillRule === 'positive'
        ? 'winding > 0'
        : fillRule === 'negative'
          ? 'winding < 0'
          : fillRule === 'nonzero'
            ? 'winding != 0'
            : 'inShell && !inHole';
    return /* wgsl */ `
const POLYGON_COUNT: u32 = ${polygonCount}u;
${EXACT_ORIENTATION_WGSL}
${EXACT_GEOMETRY_WGSL}
fn readOperandVertex(row: u32) -> vec2f {
  return vec2f(positions[positionsOffset + row * 2u], positions[positionsOffset + row * 2u + 1u]);
}
${getExactRingWindingWGSL('getRingWinding', 'readOperandVertex')}
fn insidePolygons(probe: ExactProbe) -> bool {
  for (var polygon = 0u; polygon < POLYGON_COUNT; polygon++) {
    let firstRing = polygonOffsets[polygonOffsetsOffset + polygon];
    let lastRing = polygonOffsets[polygonOffsetsOffset + polygon + 1u];
    var winding = 0;
    var inShell = false;
    var inHole = false;
    for (var ring = firstRing; ring < lastRing; ring++) {
      let begin = ringOffsets[ringOffsetsOffset + ring];
      let end = ringOffsets[ringOffsetsOffset + ring + 1u];
      let lowY = ringBounds[ringBoundsOffset + ring * 2u];
      let highY = ringBounds[ringBoundsOffset + ring * 2u + 1u];
      var ringWinding = 0;
      // A probe outside the ring's closed y extent cannot cross any of its edges.
      if (probe.point.y >= lowY && probe.point.y <= highY) {
        ringWinding = getRingWinding(probe, begin, end).x;
      }
      ${
        evenOdd
          ? `let odd = (ringWinding & 1) != 0;
      if (ring == firstRing) {
        inShell = odd;
        if (!inShell) { break; }
      } else if (odd) {
        inHole = true;
        break;
      }`
          : 'winding += ringWinding;'
      }
    }
    if (${test}) { return true; }
  }
  return false;
}`;
  }
}
