// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  validatePackedUint32View,
  validatePackedView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {createWGSLKernelNode} from '../../utils/wgsl-kernel-nodes';
import {validateGraphViewsBelongToGraph} from '../../utils/gpu-contributor-utils';
import type {GPULineGeometryPort} from '../contracts/index';
import {
  GPUSegmentRingAssembly,
  GPU_SEGMENT_RING_ASSEMBLY_FLAG_CANCELLED,
  GPU_SEGMENT_RING_ASSEMBLY_FLAG_DANGLING,
  GPU_SEGMENT_RING_ASSEMBLY_NONE
} from '../ring-assembly/index';
import {GPULineNoding} from './gpu-line-noding';
import {
  GPU_POLYGONIZE_EDGE_CLASS,
  GPU_TOPOLOGY_NONE,
  type GPUPolygonizeOutput,
  type GPUTopologyPrecisionPolicy
} from './topology-types';

const OPERATION = 'GPUPolygonize';

/** Properties for {@link GPUPolygonize}. */
export type GPUPolygonizeProps = {
  id?: string;
  /** Arbitrary finite planar linework; rings need not be pre-noded or consistently oriented. */
  lines: Omit<GPULineGeometryPort, 'positions' | 'sourceIds'> & {
    positions: GraphDataView<'float32x2'>;
    sourceIds?: GraphDataView<'uint32'>;
  };
  /** Optional polygonization group per input line; omitted lines belong to one shared group. */
  groupIds?: GraphDataView<'uint32'>;
  /** Capacity of the exact intersection event list used by noding. */
  intersectionCapacity: number;
  /** Explicit precision and vertex identity policy. */
  precision: GPUTopologyPrecisionPolicy;
  /** Caller-owned noding, half-edge, polygon and diagnostic outputs. */
  output: GPUPolygonizeOutput;
  uncertainCount?: GraphDataView<'uint32'>;
  spatialSort?: boolean;
  leafCapacity?: number;
};

/**
 * General bounded polygonization of planar linework.
 *
 * The operation first nodes every crossing, touch and collinear overlap, expands every atomic
 * segment into two directed edges, then applies the ring assembler's deterministic left-face
 * traversal. Clockwise unbounded faces and zero-area walks are excluded from the polygon port.
 * Every polygon is one canonical polygon feature and `polygons.sourceIds` carries the source group
 * of its shell. Diagnostics remain aligned with noded source segments.
 */
export class GPUPolygonize implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPUPolygonizeProps;

  constructor(props: GPUPolygonizeProps) {
    this.id = props.id ?? 'polygonize';
    this.props = props;
    const {id} = this;
    const {output, precision} = props;
    if (!(precision.vertexTolerance > 0) || !Number.isFinite(precision.vertexTolerance)) {
      throw new Error(`${id} precision.vertexTolerance must be finite and positive`);
    }
    if (precision.predicates !== 'exact-or-uncertain' || precision.storage !== 'float32') {
      throw new Error(`${id} supports exact-or-uncertain predicates with float32 storage`);
    }
    if (props.groupIds) {
      validatePackedUint32View(props.groupIds, `${id} groupIds`);
      if (props.groupIds.length !== props.lines.lineOffsets.length - 1) {
        throw new Error(`${id} groupIds length must equal the input line count`);
      }
    }
    const segmentCapacity = output.nodedSegments.endpoints.length;
    const directedCapacity = output.directedEdges.endpoints.length;
    if (directedCapacity !== segmentCapacity * 2) {
      throw new Error(`${id} directed edge capacity must be twice the noded segment capacity`);
    }
    for (const [name, view] of [
      ['sourceSegmentIds', output.directedEdges.sourceSegmentIds],
      ['sourceFeatureIds', output.directedEdges.sourceFeatureIds],
      ['twinIds', output.directedEdges.twinIds],
      ['ringIds', output.directedEdges.ringIds],
      ['flags', output.directedEdges.flags]
    ] as const) {
      validatePackedUint32View(view, `${id} output.directedEdges.${name}`);
      if (view.length !== directedCapacity) {
        throw new Error(`${id} directedEdges.${name} length must equal endpoints length`);
      }
    }
    validatePackedView(
      output.directedEdges.endpoints,
      ['float32x4'],
      `${id} output.directedEdges.endpoints`
    );
    validatePackedUint32View(output.diagnostics.edgeClasses, `${id} diagnostics.edgeClasses`);
    if (output.diagnostics.edgeClasses.length !== segmentCapacity) {
      throw new Error(`${id} diagnostics.edgeClasses length must equal noded segment capacity`);
    }
    for (const [name, view] of Object.entries(output.diagnostics)) {
      if (name !== 'edgeClasses') {
        validatePackedUint32View(view, `${id} diagnostics.${name}`);
        if (view.length < 1) {
          throw new Error(`${id} diagnostics.${name} must contain one row`);
        }
      }
    }
    const ringCapacity = output.polygons.ringOffsets.length - 1;
    if (
      ringCapacity < 1 ||
      output.polygons.polygonOffsets.length !== ringCapacity + 1 ||
      output.polygons.featureOffsets.length !== ringCapacity + 1 ||
      output.polygons.sourceIds.length !== ringCapacity
    ) {
      throw new Error(`${id} polygon offset and source capacities must share one ring capacity`);
    }
    if (output.polygons.kind !== 'polygons') {
      throw new Error(`${id} output.polygons.kind must be polygons`);
    }
    if (!output.status.requiredCount || !output.status.candidateOverflow) {
      throw new Error(`${id} output.status requires requiredCount and candidateOverflow`);
    }
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {output} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      props.lines.positions,
      props.lines.lineOffsets,
      props.lines.sourceIds,
      props.groupIds,
      output.nodedSegments.endpoints,
      output.nodedSegments.sourceFeatureIds,
      output.nodedSegments.sourceRingIds,
      output.nodedSegments.sourceEdgeIds,
      output.nodedSegments.sourceStartParameters,
      output.nodedSegments.sourceEndParameters,
      ...Object.values(output.nodedSegments.status),
      output.directedEdges.endpoints,
      output.directedEdges.sourceSegmentIds,
      output.directedEdges.sourceFeatureIds,
      output.directedEdges.twinIds,
      output.directedEdges.ringIds,
      output.directedEdges.flags,
      ...Object.values(output.directedEdges.status),
      output.polygons.positions,
      output.polygons.featureOffsets,
      output.polygons.polygonOffsets,
      output.polygons.ringOffsets,
      output.polygons.sourceIds,
      output.diagnostics.edgeClasses,
      ...Object.values(output.diagnostics).filter(view => view !== output.diagnostics.edgeClasses),
      ...Object.values(output.status),
      props.uncertainCount
    ]);
    const nodes: GPUCommandNode<Parameters>[] = [];
    nodes.push(
      ...new GPULineNoding({
        id: `${id}-noding`,
        lines: props.lines,
        intersectionCapacity: props.intersectionCapacity,
        output: output.nodedSegments,
        uncertainCount: props.uncertainCount,
        spatialSort: props.spatialSort,
        leafCapacity: props.leafCapacity
      }).getCommandNodes(graph)
    );

    const segmentCapacity = output.nodedSegments.endpoints.length;
    const directedCapacity = output.directedEdges.endpoints.length;
    const directedGroups = createTransientView(
      graph,
      `${id}-directed-groups`,
      'uint32',
      directedCapacity
    );
    const groupBindings = props.groupIds
      ? [{name: 'inputGroups', view: props.groupIds, type: 'u32' as const, access: 'read' as const}]
      : [];
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-directed-edge-geometry`,
        operation: OPERATION,
        variant: 'directed-edge-geometry',
        bindings: [
          {name: 'count', view: output.nodedSegments.status.count, type: 'u32', access: 'read'},
          {
            name: 'sourceEndpoints',
            view: output.nodedSegments.endpoints,
            type: 'f32',
            access: 'read'
          },
          {
            name: 'endpoints',
            view: output.directedEdges.endpoints,
            type: 'f32',
            access: 'read_write'
          }
        ],
        invocationCount: directedCapacity,
        body: `let source = index / 2u;
  let isActive = source < count[countOffset];
  let input = sourceEndpointsOffset + source * 4u;
  let outputBase = endpointsOffset + index * 4u;
  if (isActive && (index & 1u) == 0u) {
    endpoints[outputBase] = sourceEndpoints[input]; endpoints[outputBase + 1u] = sourceEndpoints[input + 1u];
    endpoints[outputBase + 2u] = sourceEndpoints[input + 2u]; endpoints[outputBase + 3u] = sourceEndpoints[input + 3u];
  } else if (isActive) {
    endpoints[outputBase] = sourceEndpoints[input + 2u]; endpoints[outputBase + 1u] = sourceEndpoints[input + 3u];
    endpoints[outputBase + 2u] = sourceEndpoints[input]; endpoints[outputBase + 3u] = sourceEndpoints[input + 1u];
  } else {
    endpoints[outputBase] = 0.0; endpoints[outputBase + 1u] = 0.0;
    endpoints[outputBase + 2u] = 0.0; endpoints[outputBase + 3u] = 0.0;
  }`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-directed-edge-provenance`,
        operation: OPERATION,
        variant: 'directed-edge-provenance',
        bindings: [
          {name: 'count', view: output.nodedSegments.status.count, type: 'u32', access: 'read'},
          {
            name: 'sourceFeatures',
            view: output.nodedSegments.sourceFeatureIds,
            type: 'u32',
            access: 'read'
          },
          {
            name: 'sourceRings',
            view: output.nodedSegments.sourceRingIds,
            type: 'u32',
            access: 'read'
          },
          ...groupBindings,
          {
            name: 'sourceSegmentIds',
            view: output.directedEdges.sourceSegmentIds,
            type: 'u32',
            access: 'read_write'
          },
          {
            name: 'sourceFeatureIds',
            view: output.directedEdges.sourceFeatureIds,
            type: 'u32',
            access: 'read_write'
          },
          {name: 'twinIds', view: output.directedEdges.twinIds, type: 'u32', access: 'read_write'},
          {name: 'groups', view: directedGroups, type: 'u32', access: 'read_write'}
        ],
        invocationCount: directedCapacity,
        declarations: `const NONE: u32 = ${GPU_TOPOLOGY_NONE}u;`,
        body: `let source = index / 2u;
  let isActive = source < count[countOffset];
  sourceSegmentIds[sourceSegmentIdsOffset + index] = select(NONE, source, isActive);
  sourceFeatureIds[sourceFeatureIdsOffset + index] = select(NONE, sourceFeatures[sourceFeaturesOffset + source], isActive);
  twinIds[twinIdsOffset + index] = select(NONE, index ^ 1u, isActive);
  let sourceRing = sourceRings[sourceRingsOffset + source];
  groups[groupsOffset + index] = select(NONE, ${props.groupIds ? 'inputGroups[inputGroupsOffset + sourceRing]' : '0u'}, isActive);`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-directed-edge-status`,
        operation: OPERATION,
        variant: 'directed-edge-status',
        bindings: [
          {name: 'count', view: output.nodedSegments.status.count, type: 'u32', access: 'read'},
          {
            name: 'requiredCount',
            view: output.nodedSegments.status.requiredCount,
            type: 'u32',
            access: 'read'
          },
          {
            name: 'candidateOverflow',
            view: output.nodedSegments.status.candidateOverflow,
            type: 'u32',
            access: 'read'
          },
          {
            name: 'directedCount',
            view: output.directedEdges.status.count,
            type: 'u32',
            access: 'read_write'
          },
          {
            name: 'directedRequired',
            view: output.directedEdges.status.requiredCount,
            type: 'u32',
            access: 'read_write'
          },
          {
            name: 'directedOverflow',
            view: output.directedEdges.status.overflow,
            type: 'u32',
            access: 'read_write'
          },
          {
            name: 'directedCandidateOverflow',
            view: output.directedEdges.status.candidateOverflow,
            type: 'u32',
            access: 'read_write'
          }
        ],
        invocationCount: 1,
        body: `directedCount[directedCountOffset] = 2u * count[countOffset];
  directedRequired[directedRequiredOffset] = 2u * requiredCount[requiredCountOffset];
  directedOverflow[directedOverflowOffset] = candidateOverflow[candidateOverflowOffset];
  directedCandidateOverflow[directedCandidateOverflowOffset] = candidateOverflow[candidateOverflowOffset];`
      })
    );

    const ringCapacity = output.polygons.ringOffsets.length - 1;
    const vertexCapacity = output.polygons.positions.length;
    const transient = <Format extends 'uint32' | 'float32' | 'float32x2'>(
      name: string,
      format: Format,
      length: number
    ) => createTransientView(graph, `${id}-${name}`, format, length);
    const ringPositions = transient('ring-positions', 'float32x2', vertexCapacity);
    const ringOffsets = transient('ring-offsets', 'uint32', ringCapacity + 1);
    const ringAreas = transient('ring-areas', 'float32', ringCapacity);
    const ringIsHole = transient('ring-is-hole', 'uint32', ringCapacity);
    const ringShells = transient('ring-shells', 'uint32', ringCapacity);
    const ringCount = transient('ring-count', 'uint32', 1);
    const ringRequiredCount = transient('ring-required-count', 'uint32', 1);
    const ringOverflow = transient('ring-overflow', 'uint32', 1);
    nodes.push(
      ...new GPUSegmentRingAssembly({
        id: `${id}-faces`,
        endpoints: output.directedEdges.endpoints,
        count: output.directedEdges.status.count,
        groups: directedGroups,
        pairedOpposites: true,
        vertexTolerance: props.precision.vertexTolerance,
        interiorSide: 'left',
        normalizeWinding: true,
        geographic: false,
        splitTouchingRings: true,
        output: {
          positions: ringPositions,
          ringOffsets,
          ringAreas,
          ringIsHole,
          ringShells,
          segmentRings: output.directedEdges.ringIds,
          segmentFlags: output.directedEdges.flags,
          count: ringCount,
          requiredCount: ringRequiredCount,
          overflow: ringOverflow,
          polygons: output.polygons
        }
      }).getCommandNodes(graph)
    );

    const diagnosticCounters = transient('diagnostic-counters', 'uint32', 5);
    const edgeClass = GPU_POLYGONIZE_EDGE_CLASS;
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-diagnostic-clear`,
        operation: OPERATION,
        variant: 'diagnostic-clear',
        bindings: [{name: 'counters', view: diagnosticCounters, type: 'u32', access: 'read_write'}],
        invocationCount: 5,
        body: 'counters[countersOffset + index] = 0u;'
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-diagnostics`,
        operation: OPERATION,
        variant: 'diagnostics',
        bindings: [
          {
            name: 'segmentCount',
            view: output.nodedSegments.status.count,
            type: 'u32',
            access: 'read'
          },
          {name: 'ringIds', view: output.directedEdges.ringIds, type: 'u32', access: 'read'},
          {name: 'flags', view: output.directedEdges.flags, type: 'u32', access: 'read'},
          {name: 'ringIsHole', view: ringIsHole, type: 'u32', access: 'read'},
          {name: 'ringShells', view: ringShells, type: 'u32', access: 'read'},
          {name: 'ringAreas', view: ringAreas, type: 'f32', access: 'read'},
          {
            name: 'edgeClasses',
            view: output.diagnostics.edgeClasses,
            type: 'u32',
            access: 'read_write'
          },
          {name: 'counters', view: diagnosticCounters, type: 'atomic<u32>', access: 'read_write'}
        ],
        invocationCount: segmentCapacity,
        declarations: `const NONE: u32 = ${GPU_SEGMENT_RING_ASSEMBLY_NONE}u;
const FLAG_CANCELLED: u32 = ${GPU_SEGMENT_RING_ASSEMBLY_FLAG_CANCELLED}u;
const FLAG_DANGLING: u32 = ${GPU_SEGMENT_RING_ASSEMBLY_FLAG_DANGLING}u;`,
        body: `var classification = 0u;
  if (index < segmentCount[segmentCountOffset]) {
    let first = 2u * index;
    let firstRing = ringIds[ringIdsOffset + first];
    let secondRing = ringIds[ringIdsOffset + first + 1u];
    let firstBounded = firstRing != NONE && abs(ringAreas[ringAreasOffset + firstRing]) > 0.0 && (ringIsHole[ringIsHoleOffset + firstRing] == 0u || ringShells[ringShellsOffset + firstRing] != NONE);
    let secondBounded = secondRing != NONE && abs(ringAreas[ringAreasOffset + secondRing]) > 0.0 && (ringIsHole[ringIsHoleOffset + secondRing] == 0u || ringShells[ringShellsOffset + secondRing] != NONE);
    let used = firstBounded || secondBounded;
    let flagBits = flags[flagsOffset + first] | flags[flagsOffset + first + 1u];
    let dangle = (flagBits & FLAG_DANGLING) != 0u;
    let cancelled = (flagBits & FLAG_CANCELLED) != 0u;
    if (used) {
      classification = ${edgeClass.closedRing}u;
      atomicAdd(&counters[countersOffset], 1u);
    } else {
      classification = ${edgeClass.unused}u;
      atomicAdd(&counters[countersOffset + 4u], 1u);
      if (!cancelled) {
        classification = classification | ${edgeClass.openChain}u;
        atomicAdd(&counters[countersOffset + 1u], 1u);
      }
      if (dangle) {
        classification = classification | ${edgeClass.dangle}u;
        atomicAdd(&counters[countersOffset + 3u], 1u);
      } else if (!cancelled) {
        classification = classification | ${edgeClass.cut}u;
        atomicAdd(&counters[countersOffset + 2u], 1u);
      }
    }
  }
  edgeClasses[edgeClassesOffset + index] = classification;`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-publish-diagnostics`,
        operation: OPERATION,
        variant: 'publish-diagnostics',
        bindings: [
          {name: 'counters', view: diagnosticCounters, type: 'u32', access: 'read'},
          {
            name: 'closedRingCount',
            view: output.diagnostics.closedRingCount,
            type: 'u32',
            access: 'read_write'
          },
          {
            name: 'openChainCount',
            view: output.diagnostics.openChainCount,
            type: 'u32',
            access: 'read_write'
          },
          {
            name: 'cutEdgeCount',
            view: output.diagnostics.cutEdgeCount,
            type: 'u32',
            access: 'read_write'
          },
          {
            name: 'dangleCount',
            view: output.diagnostics.dangleCount,
            type: 'u32',
            access: 'read_write'
          },
          {
            name: 'unusedEdgeCount',
            view: output.diagnostics.unusedEdgeCount,
            type: 'u32',
            access: 'read_write'
          }
        ],
        invocationCount: 1,
        body: `closedRingCount[closedRingCountOffset] = counters[countersOffset];
  openChainCount[openChainCountOffset] = counters[countersOffset + 1u];
  cutEdgeCount[cutEdgeCountOffset] = counters[countersOffset + 2u];
  dangleCount[dangleCountOffset] = counters[countersOffset + 3u];
  unusedEdgeCount[unusedEdgeCountOffset] = counters[countersOffset + 4u];`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-publish-status`,
        operation: OPERATION,
        variant: 'publish-status',
        bindings: [
          {name: 'ringOverflow', view: ringOverflow, type: 'u32', access: 'read'},
          {
            name: 'nodingOverflow',
            view: output.nodedSegments.status.overflow,
            type: 'u32',
            access: 'read'
          },
          {
            name: 'featureOffsets',
            view: output.polygons.featureOffsets,
            type: 'u32',
            access: 'read'
          },
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
        body: `let polygons = featureOffsets[featureOffsetsOffset + RING_CAPACITY];
  let incomplete = ringOverflow[ringOverflowOffset] | nodingOverflow[nodingOverflowOffset];
  count[countOffset] = polygons;
  requiredCount[requiredCountOffset] = polygons;
  overflow[overflowOffset] = incomplete;
  candidateOverflow[candidateOverflowOffset] = incomplete;`
      })
    );
    return nodes;
  }
}
