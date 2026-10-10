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
import {validateGraphViewsBelongToGraph} from '../../utils/gpu-contributor-utils';
import type {GPUBoundedResultStatusPort, GPUPolygonGeometryPort} from '../contracts/index';
import {
  GPULineNoding,
  GPU_TOPOLOGY_NONE,
  type GPUNodedSegmentPort
} from '../geometry-topology/index';
import {GPUSegmentRingAssembly} from '../ring-assembly/index';

const OPERATION = 'GPUPolygonOverlay';

/** Boolean set operation applied by {@link GPUPolygonOverlay}. */
export type GPUPolygonOverlayOperation =
  | 'intersection'
  | 'union'
  | 'difference'
  | 'symmetric-difference'
  | 'dissolve';

/** Explicit capacities for the bounded arrangement built by {@link GPUPolygonOverlay}. */
export type GPUPolygonOverlayCapacity = {
  /** Segment-intersection events retained while noding both operands. */
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
  geometry: Omit<GPUPolygonGeometryPort, 'positions' | 'sourceIds'> & {
    positions: GraphDataView<'float32x2'>;
    sourceIds: GraphDataView<'uint32'>;
  };
  sourceIds: GraphDataView<'uint32'>;
  status: GPUBoundedResultStatusPort & {
    requiredCount: GraphDataView<'uint32'>;
    candidateOverflow: GraphDataView<'uint32'>;
  };
  boundary: GPUPolygonOverlayBoundaryPort;
};

type PackedPolygons = Omit<GPUPolygonGeometryPort, 'positions' | 'sourceIds'> & {
  positions: GraphDataView<'float32x2'>;
  sourceIds?: GraphDataView<'uint32'>;
};

/** Properties for {@link GPUPolygonOverlay}. */
export type GPUPolygonOverlayProps = {
  id?: string;
  left: PackedPolygons;
  /** Required except for `dissolve`, which unions every polygon in `left`. */
  right?: PackedPolygons;
  operation: GPUPolygonOverlayOperation;
  capacity: GPUPolygonOverlayCapacity;
  /** Chebyshev vertex identity tolerance for noding and ring assembly. */
  vertexTolerance: number;
  output: GPUPolygonOverlayOutput;
  uncertainCount?: GraphDataView<'uint32'>;
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
 * Output and arrangement capacities are explicit. `candidateOverflow` means noding was incomplete;
 * `overflow` additionally covers boundary, ring or vertex capacity. Boundary provenance remains
 * aligned with the compact selected edge list. Coordinates and side tests use f32; exact-or-
 * uncertain segment predicates are inherited from `GPULineNoding`.
 */
export class GPUPolygonOverlay implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPUPolygonOverlayProps;

  constructor(props: GPUPolygonOverlayProps) {
    this.id = props.id ?? 'polygon-overlay';
    this.props = props;
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
    if (!(props.vertexTolerance > 0) || !Number.isFinite(props.vertexTolerance)) {
      throw new Error(`${id} vertexTolerance must be finite and positive`);
    }
    for (const [name, value] of Object.entries(props.capacity)) {
      if (!Number.isSafeInteger(value) || value < 1) {
        throw new Error(`${id} capacity.${name} must be a positive integer`);
      }
    }
    validatePolygons(id, 'output.geometry', output.geometry);
    validatePackedUint32View(output.sourceIds, `${id} output.sourceIds`);
    if (output.geometry.sourceIds !== output.sourceIds) {
      throw new Error(`${id} output.geometry.sourceIds and output.sourceIds must be the same view`);
    }
    const ringCapacity = output.geometry.ringOffsets.length - 1;
    if (
      output.geometry.polygonOffsets.length !== ringCapacity + 1 ||
      output.geometry.featureOffsets.length !== ringCapacity + 1 ||
      output.sourceIds.length !== ringCapacity
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
      output.sourceIds,
      output.boundary.endpoints,
      output.boundary.operandIds,
      output.boundary.sourceFeatureIds,
      ...Object.values(output.boundary.status),
      ...Object.values(output.status),
      props.uncertainCount
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
        lines: {
          kind: 'lines',
          positions: combinedPositions,
          lineOffsets: combinedOffsets,
          sourceIds: combinedSources
        },
        intersectionCapacity: props.capacity.intersections,
        output: noded,
        uncertainCount: props.uncertainCount,
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
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-left-membership`,
        operation: OPERATION,
        variant: 'left-membership',
        bindings: [
          {name: 'segmentCount', view: noded.status.count, type: 'u32', access: 'read'},
          {name: 'endpoints', view: noded.endpoints, type: 'f32', access: 'read'},
          {name: 'positions', view: left.positions, type: 'f32', access: 'read'},
          {name: 'polygonOffsets', view: left.polygonOffsets, type: 'u32', access: 'read'},
          {name: 'ringOffsets', view: left.ringOffsets, type: 'u32', access: 'read'},
          {name: 'membership', view: leftMembership, type: 'u32', access: 'read_write'}
        ],
        invocationCount: props.capacity.nodedSegments,
        declarations: this._getMembershipWGSL(left.polygonOffsets.length - 1),
        body: `var value = 0u;
  if (index < segmentCount[segmentCountOffset]) {
    let base = endpointsOffset + index * 4u;
    let a = vec2f(endpoints[base], endpoints[base + 1u]);
    let b = vec2f(endpoints[base + 2u], endpoints[base + 3u]);
    let delta = b - a;
    let length = max(length(delta), ${getWGSLFloatLiteral(props.vertexTolerance)});
    let normal = vec2f(-delta.y, delta.x) / length;
    let probeDistance = max(${getWGSLFloatLiteral(props.vertexTolerance * 0.25)}, length * 1e-6);
    let midpoint = 0.5 * (a + b);
    let leftProbe = midpoint + normal * probeDistance;
    let rightProbe = midpoint - normal * probeDistance;
    value = select(0u, 1u, insidePolygons(leftProbe)) |
      select(0u, 2u, insidePolygons(rightProbe));
  }
  membership[membershipOffset + index] = value;`
      })
    );
    if (right && rightMembership) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-right-membership`,
          operation: OPERATION,
          variant: 'right-membership',
          bindings: [
            {name: 'segmentCount', view: noded.status.count, type: 'u32', access: 'read'},
            {name: 'endpoints', view: noded.endpoints, type: 'f32', access: 'read'},
            {name: 'positions', view: right.positions, type: 'f32', access: 'read'},
            {name: 'polygonOffsets', view: right.polygonOffsets, type: 'u32', access: 'read'},
            {name: 'ringOffsets', view: right.ringOffsets, type: 'u32', access: 'read'},
            {name: 'membership', view: rightMembership, type: 'u32', access: 'read_write'}
          ],
          invocationCount: props.capacity.nodedSegments,
          declarations: this._getMembershipWGSL(right.polygonOffsets.length - 1),
          body: `var value = 0u;
  if (index < segmentCount[segmentCountOffset]) {
    let base = endpointsOffset + index * 4u;
    let a = vec2f(endpoints[base], endpoints[base + 1u]);
    let b = vec2f(endpoints[base + 2u], endpoints[base + 3u]);
    let delta = b - a;
    let length = max(length(delta), ${getWGSLFloatLiteral(props.vertexTolerance)});
    let normal = vec2f(-delta.y, delta.x) / length;
    let probeDistance = max(${getWGSLFloatLiteral(props.vertexTolerance * 0.25)}, length * 1e-6);
    let midpoint = 0.5 * (a + b);
    value = select(0u, 1u, insidePolygons(midpoint + normal * probeDistance)) |
      select(0u, 2u, insidePolygons(midpoint - normal * probeDistance));
  }
  membership[membershipOffset + index] = value;`
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
        endpoints: boundary.endpoints,
        count: boundary.status.count,
        groups: boundaryGroups,
        vertexTolerance: props.vertexTolerance,
        interiorSide: 'left',
        normalizeWinding: true,
        geographic: false,
        cancelOpposingSegments: true,
        splitTouchingRings: true,
        output: {
          positions: ringPositions,
          ringOffsets,
          count: ringCount,
          requiredCount: ringRequiredCount,
          overflow: ringOverflow,
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
          {name: 'sourceIds', view: output.sourceIds, type: 'u32', access: 'read_write'}
        ],
        invocationCount: Math.max(1, output.sourceIds.length),
        declarations: `const RING_CAPACITY: u32 = ${ringCapacity}u;
const SOURCE_CAPACITY: u32 = ${output.sourceIds.length}u;
const NONE: u32 = ${GPU_TOPOLOGY_NONE}u;`,
        body: `let featureCount = featureOffsets[featureOffsetsOffset + RING_CAPACITY];
  if (index < SOURCE_CAPACITY) {
    sourceIds[sourceIdsOffset + index] = select(NONE, index, index < featureCount);
  }`
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
          {
            name: 'requiredCount',
            view: output.status.requiredCount,
            type: 'u32',
            access: 'read_write'
          },
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
  requiredCount[requiredCountOffset] = featureCount;
  overflow[overflowOffset] = incomplete;
  candidateOverflow[candidateOverflowOffset] = candidateOverflowInput[candidateOverflowInputOffset];`
      })
    );
    return nodes;
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

  private _getMembershipWGSL(polygonCount: number): string {
    return /* wgsl */ `
const POLYGON_COUNT: u32 = ${polygonCount}u;

fn insidePolygons(point: vec2f) -> bool {
  for (var polygon = 0u; polygon < POLYGON_COUNT; polygon++) {
    let firstRing = polygonOffsets[polygonOffsetsOffset + polygon];
    let lastRing = polygonOffsets[polygonOffsetsOffset + polygon + 1u];
    var inShell = false;
    var inHole = false;
    for (var ring = firstRing; ring < lastRing; ring++) {
      let begin = ringOffsets[ringOffsetsOffset + ring];
      let end = ringOffsets[ringOffsetsOffset + ring + 1u];
      var inside = false;
      if (end > begin + 2u) {
        var previous = end - 1u;
        for (var row = begin; row < end; row++) {
          let a = vec2f(positions[positionsOffset + previous * 2u], positions[positionsOffset + previous * 2u + 1u]);
          let b = vec2f(positions[positionsOffset + row * 2u], positions[positionsOffset + row * 2u + 1u]);
          if ((a.y > point.y) != (b.y > point.y)) {
            let crossing = (b.x - a.x) * (point.y - a.y) / (b.y - a.y) + a.x;
            if (point.x < crossing) { inside = !inside; }
          }
          previous = row;
        }
      }
      if (ring == firstRing) { inShell = inside; } else if (inside) { inHole = true; }
    }
    if (inShell && !inHole) { return true; }
  }
  return false;
}`;
  }
}
