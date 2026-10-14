// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {
  resolveGPUSpatialContext,
  GPU_PLANAR_OR_GREAT_CIRCLE_SPATIAL_CONTEXT_SUPPORT,
  getGPUGreatCircleRadius
} from '../../analysis-core/contracts/spatial-context';
import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {createWGSLKernelNode, type WGSLKernelBinding} from '../../utils/wgsl-kernel-nodes';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {
  getStorageBindingElementLimit,
  validateCapacity,
  validateMaximumCount,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validatePackedView
} from '../../utils/gpu-contributor-utils';
import {
  getGPUNetworkTopologyViews,
  validateGPUNetworkTopology,
  type GPUNetworkAdjacency
} from '../../analysis-core/network-topology';
import {GPUNearestFeatureJoin} from '../../gpu-spatial-analysis/spatial-join/index';
import type {
  GPUSpatialContext,
  GPUSpatialContextSupport
} from '../../analysis-core/contracts/spatial-context';
import {getNetworkMetricWGSL} from '../network-spatial-context';
import {ACCESSIBILITY_NONE} from './network-accessibility-passes';
import {createNetworkEdgeSourceNodes} from './network-edge-sources';

const SNAPPING_SPATIAL_CONTEXT_SUPPORT: GPUSpatialContextSupport = Object.freeze({
  ...GPU_PLANAR_OR_GREAT_CIRCLE_SPATIAL_CONTEXT_SUPPORT,
  measure: 'distance'
});

const OPERATION = 'GPUNetworkSnapping';

/** Value written to `snappedEdges` and `output.startNodes` when a point snaps to no edge. */
export const GPU_NETWORK_SNAPPING_NONE = 0xffffffff;

/**
 * Largest edge count of {@link GPUNetworkSnapping} with `candidateCapacity` on `device`: the
 * nearest-edge BVH holds 8 bytes of node bounds per node (`2 * edges - 1` nodes) in one storage
 * binding, so 8388608 on a default device (128 MiB bindings). This equals the leaf limit of
 * `GPUNearestFeatureJoin`, whose function can replace the body once it is exported.
 */
export function getGPUNetworkSnappingMaximumCandidateEdgeCount(device: Device): number {
  return getStorageBindingElementLimit(device, 16);
}

/** Value written to fraction, distance, and cost outputs when a point snaps to no edge. */
export const GPU_NETWORK_SNAPPING_NO_VALUE = -1;

/**
 * Which edge endpoints become start nodes of a snapped point:
 * - `'both'`: the edge source (cost `fraction * edgeCost`) and target (`(1 - fraction) * edgeCost`),
 *   for undirected networks that list both directions of each road.
 * - `'forward'`: only the target, for a point that leaves along a directed edge (forward search).
 * - `'reverse'`: only the source, for a point reached along a directed edge (reverse search).
 */
export type GPUNetworkSnappingStartDirection = 'both' | 'forward' | 'reverse';

/** GPU-written views of {@link GPUNetworkSnapping}. */
export type GPUNetworkSnappingOutput = {
  /**
   * Nearest edge row per point, or `GPU_NETWORK_SNAPPING_NONE`. Ties go to the smallest edge row,
   * so a point on a node shared by several edges snaps to the smallest incident edge.
   */
  snappedEdges: GraphDataView<'uint32'>;
  /** Optional fraction along the edge from source (0) to target (1), or -1. */
  snapFractions?: GraphDataView<'float32'>;
  /** Optional distance from the point to the edge in the context's length unit, or -1. */
  snapDistances?: GraphDataView<'float32'>;
  /** Optional cost from the snapped position to the edge source, `fraction * edgeCost`, or -1. */
  sourceCosts?: GraphDataView<'float32'>;
  /** Optional cost from the snapped position to the edge target, `(1 - fraction) * edgeCost`, or -1. */
  targetCosts?: GraphDataView<'float32'>;
  /** Optional snapped position, or the point itself when it snaps to no edge. */
  snappedPositions?: GraphDataView<'float32x2'>;
  /**
   * Optional `2 * pointCount` start nodes: row `2 * i` is the edge source of point `i` and row
   * `2 * i + 1` the edge target, or `GPU_NETWORK_SNAPPING_NONE` when excluded by `startDirection`
   * or unsnapped. Feed it to `GPUNetworkCostMatrix` with `startsPerRow: 2`.
   */
  startNodes?: GraphDataView<'uint32'>;
  /** Optional `2 * pointCount` start costs aligned with `startNodes`, or -1. */
  startCosts?: GraphDataView<'float32'>;
  /** GPU-written status words. */
  status?: {
    /**
     * One-row flag: 1 when the BVH candidate search overflowed, so some points were snapped without
     * all their candidate edges. Required with `candidateCapacity` (library rule 9: a capacity that
     * can drop work reports it); not allowed without it.
     */
    overflow?: GraphDataView<'uint32'>;
  };
};

/**
 * Properties for {@link GPUNetworkSnapping}.
 *
 * Compile-time: point, node and edge counts, `candidateCapacity`, `startDirection`, and which
 * optional views exist. Per-frame: point and node positions, edge lists and costs, and
 * `maximumSnapDistance`.
 */
export type GPUNetworkSnappingProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'network-snapping'`. */
  id?: string;
  /**
   * Coordinates of `positions` and `topology.nodePositions`. Omitted means native planar coordinates.
   * `{coordinateSpace: 'longitude-latitude', metric: 'great-circle'}` takes degrees and measures
   * snap distances, default edge costs and `maximumSnapDistance` in the context's length unit
   * (meters by default, or kilometers). Local planar frames must be azimuthal-equidistant,
   * not Lambert equal-area. The context is passed to `GPUNearestFeatureJoin` on the
   * `candidateCapacity` path.
   */
  spatialContext?: GPUSpatialContext;
  /** Positions to snap (origins, facilities, opportunities), in the coordinates of `spatialContext`. */
  positions: GraphDataView<'float32x2'>;
  /**
   * The network, with node positions in the same coordinates as `positions`. `weights` are optional
   * travel costs per edge and default to the edge length in the context's length unit.
   * Edges with an endpoint out of range are skipped, and edge slots past `offsets[nodeCount]` (a
   * topology shorter than its edge capacity) are not part of the network.
   */
  topology: GPUNetworkAdjacency & {nodePositions: GraphDataView<'float32x2'>};
  /**
   * Optional one-row per-frame maximum snap distance in the context's length unit.
   * Points farther from every edge snap to none.
   * Required with `candidateCapacity`.
   */
  maximumSnapDistance?: GraphDataView<'float32'>;
  /**
   * Compile-time candidate capacity. When given, candidate edges come from `GPUNearestFeatureJoin`
   * (a BVH over edge segments probed with `maximumSnapDistance`); candidates past the capacity are
   * dropped and raise `overflow`. When omitted, each point scans every edge, which is exact and
   * needs no radius but costs `pointCount * edgeCount`. With a capacity the edge count is at most
   * {@link getGPUNetworkSnappingMaximumCandidateEdgeCount} (8388608 on a default device).
   */
  candidateCapacity?: number;
  /** Compile-time. Morton-sort edges before the BVH build, as `GPUNearestFeatureJoin.spatialSort`. */
  spatialSort?: boolean;
  /** Which endpoints `output.startNodes` lists. Defaults to `'both'`. */
  startDirection?: GPUNetworkSnappingStartDirection;
  /** The GPU-written views. */
  output: GPUNetworkSnappingOutput;
};

/**
 * Snaps points onto the nearest edge of a road network, so they can enter a shortest-path search
 * as virtual nodes between the edge endpoints.
 *
 * For every point the contributor reports the nearest edge (smallest edge row on ties), the fraction
 * along it, the snap distance, and the two start costs: `fraction * edgeCost` to the edge source
 * and `(1 - fraction) * edgeCost` to the edge target. The edge cost defaults to the edge length.
 * Points beyond the optional maximum snap distance snap to no edge.
 *
 * With `candidateCapacity`, candidate edges come from `GPUNearestFeatureJoin` over the edge
 * segments; without it, every point scans every edge in a single exact kernel.
 *
 * Distances are planar by default. With a great-circle `spatialContext` positions are longitude
 * and latitude degrees, the closest point of an edge is found in the local equirectangular frame
 * of the point, and snap distances and default edge costs are in the context's length unit. Edges are
 * straight segments between nodes, so they should be road-length. On the `candidateCapacity` path
 * the context is passed to `GPUNearestFeatureJoin`, whose great-circle search ignores the capacity.
 */
export class GPUNetworkSnapping implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUNetworkSnappingProps;
  /** Number of positions to snap. */
  readonly pointCount: number;
  /** Number of network nodes. */
  readonly nodeCount: number;
  /** Number of edges. */
  readonly edgeCount: number;
  /** Resolved start direction. */
  readonly startDirection: GPUNetworkSnappingStartDirection;
  /** Resolved spatial context. */
  readonly spatialContext: GPUSpatialContext;

  constructor(props: GPUNetworkSnappingProps) {
    this.id = props.id ?? 'network-snapping';
    this.props = props;
    this.startDirection = props.startDirection ?? 'both';
    const {id} = this;
    this.spatialContext = resolveGPUSpatialContext(
      id,
      props.spatialContext,
      SNAPPING_SPATIAL_CONTEXT_SUPPORT
    );
    const {output} = props;
    const {nodeCount, edgeCount} = validateGPUNetworkTopology(id, props.topology, {
      nodePositions: true
    });
    validatePackedView(props.positions, ['float32x2'], `${id} positions`);
    if (output.snappedPositions) {
      validatePackedView(output.snappedPositions, ['float32x2'], `${id} output.snappedPositions`);
    }
    for (const [name, view] of [
      ['maximumSnapDistance', props.maximumSnapDistance],
      ['output.snapFractions', output.snapFractions],
      ['output.snapDistances', output.snapDistances],
      ['output.sourceCosts', output.sourceCosts],
      ['output.targetCosts', output.targetCosts],
      ['output.startCosts', output.startCosts]
    ] as const) {
      if (view) {
        validatePackedView(view, ['float32'], `${id} ${name}`);
      }
    }
    for (const [name, view] of [
      ['output.snappedEdges', output.snappedEdges],
      ['output.startNodes', output.startNodes],
      ['output.status.overflow', output.status?.overflow]
    ] as const) {
      if (view) {
        validatePackedView(view, ['uint32'], `${id} ${name}`);
      }
    }
    this.pointCount = props.positions.length;
    this.nodeCount = nodeCount;
    this.edgeCount = edgeCount;
    for (const [name, view] of [
      ['output.snappedEdges', output.snappedEdges],
      ['output.snapFractions', output.snapFractions],
      ['output.snapDistances', output.snapDistances],
      ['output.sourceCosts', output.sourceCosts],
      ['output.targetCosts', output.targetCosts],
      ['output.snappedPositions', output.snappedPositions]
    ] as const) {
      if (view && view.length !== this.pointCount) {
        throw new Error(`${id} ${name} length must equal the position count`);
      }
    }
    for (const [name, view] of [
      ['output.startNodes', output.startNodes],
      ['output.startCosts', output.startCosts]
    ] as const) {
      if (view && view.length !== 2 * this.pointCount) {
        throw new Error(`${id} ${name} length must be twice the position count`);
      }
    }
    if (Boolean(output.startNodes) !== Boolean(output.startCosts)) {
      throw new Error(`${id} output.startNodes and output.startCosts must be given together`);
    }
    for (const [name, view] of [
      ['maximumSnapDistance', props.maximumSnapDistance],
      ['output.status.overflow', output.status?.overflow]
    ] as const) {
      if (view && view.length !== 1) {
        throw new Error(`${id} ${name} must contain exactly one row`);
      }
    }
    if (props.candidateCapacity !== undefined) {
      validateCapacity(id, 'candidateCapacity', props.candidateCapacity);
      if (!props.maximumSnapDistance) {
        throw new Error(`${id} candidateCapacity requires maximumSnapDistance`);
      }
      if (!output.status?.overflow) {
        throw new Error(`${id} candidateCapacity requires output.status.overflow`);
      }
    }
    if (output.status?.overflow && props.candidateCapacity === undefined) {
      throw new Error(`${id} output.status.overflow requires candidateCapacity`);
    }
    const outputs = getOutputs(props);
    validateGraphOutputsDisjointFromInputs(id, outputs, getInputs(props));
    const outputBuffers = outputs.filter(view => view !== undefined).map(view => view.buffer);
    if (new Set(outputBuffers).size !== outputBuffers.length) {
      throw new Error(`${id} outputs must use separate buffers`);
    }
  }

  /**
   * Returns `edge-sources-clear` and `edge-sources` (every edge row is written, rows past
   * `offsets[nodeCount]` with an out-of-range source that every scan skips),
   * the candidate search (`segments` plus the `${id}-join-*` nodes of `GPUNearestFeatureJoin`,
   * then `project`; or one `scan` node), and the optional `costs`, `positions`, and `seeds` nodes.
   */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, pointCount, nodeCount, edgeCount} = this;
    const {offsets, neighbors: edgeTargets, weights: edgeCosts} = props.topology;
    const {nodePositions} = props.topology;
    const {output} = props;
    validateGraphViewsBelongToGraph(id, graph, [...getInputs(props), ...getOutputs(props)]);
    if (props.candidateCapacity !== undefined) {
      validateMaximumCount(
        id,
        'topology edges with candidateCapacity',
        edgeCount,
        getGPUNetworkSnappingMaximumCandidateEdgeCount(graph.device)
      );
    }
    const nodes: GPUCommandNode<Parameters>[] = [];
    const nodeDeclarations = `const NODE_COUNT: u32 = ${nodeCount}u;
const EDGE_COUNT: u32 = ${edgeCount}u;
const NONE: u32 = ${ACCESSIBILITY_NONE}u;`;

    const edgeSources = createTransientView(
      graph,
      `${id}-edge-sources`,
      'uint32',
      Math.max(edgeCount, 1)
    );
    nodes.push(
      ...createNetworkEdgeSourceNodes<Parameters>(graph, {
        id: `${id}-edge-sources`,
        operation: OPERATION,
        offsets,
        edgeSources,
        nodeCount
      })
    );
    const sources = edgeSources;
    const fractions =
      output.snapFractions ?? createTransientView(graph, `${id}-fractions`, 'float32', pointCount);
    const distances =
      output.snapDistances ?? createTransientView(graph, `${id}-distances`, 'float32', pointCount);
    const metricWGSL = getNetworkMetricWGSL(getGPUGreatCircleRadius(this.spatialContext));
    const readNodeWGSL = /* wgsl */ `
fn readNode(node: u32) -> vec2<f32> {
  return vec2<f32>(nodePositions[nodePositionsOffset + node * 2u], nodePositions[nodePositionsOffset + node * 2u + 1u]);
}`;
    const segmentHelpers = /* wgsl */ `${readNodeWGSL}
${metricWGSL}
// Fraction of the closest point along a -> b; 0 for a degenerate edge.
fn getFraction(point: vec2<f32>, start: vec2<f32>, end: vec2<f32>) -> f32 {
  return networkSegmentFraction(point, start, end);
}
fn getDistance(point: vec2<f32>, start: vec2<f32>, end: vec2<f32>, fraction: f32) -> f32 {
  return networkSegmentDistance(point, start, end, fraction);
}`;
    const projectBindings: WGSLKernelBinding[] = [
      {name: 'points', view: props.positions, type: 'f32', access: 'read'},
      {
        name: 'nodePositions',
        view: nodePositions,
        type: 'f32',
        access: 'read'
      },
      {name: 'edgeSources', view: sources, type: 'u32', access: 'read'},
      {
        name: 'edgeTargets',
        view: edgeTargets,
        type: 'u32',
        access: 'read'
      }
    ];
    const projectOutputs: WGSLKernelBinding[] = [
      {
        name: 'snappedEdges',
        view: output.snappedEdges,
        type: 'u32',
        access: 'read_write'
      },
      {name: 'fractions', view: fractions, type: 'f32', access: 'read_write'},
      {name: 'distances', view: distances, type: 'f32', access: 'read_write'}
    ];
    const writeResult = `if (bestEdge == NONE) {
    snappedEdges[snappedEdgesOffset + index] = NONE;
    fractions[fractionsOffset + index] = -1.0;
    distances[distancesOffset + index] = -1.0;
    return;
  }
  snappedEdges[snappedEdgesOffset + index] = bestEdge;
  fractions[fractionsOffset + index] = bestFraction;
  distances[distancesOffset + index] = bestDistance;`;

    if (props.candidateCapacity !== undefined && props.maximumSnapDistance) {
      const segmentStarts = createTransientView(
        graph,
        `${id}-segment-starts`,
        'float32x2',
        edgeCount
      );
      const segmentEnds = createTransientView(graph, `${id}-segment-ends`, 'float32x2', edgeCount);
      const joinRows = createTransientView(graph, `${id}-join-rows`, 'uint32', pointCount);
      if (edgeCount > 0) {
        // Edges with an out-of-range endpoint become segments that never match: far away in
        // planar coordinates, non-finite (skipped by the join) on longitude/latitude, where no
        // position is far from every point.
        nodes.push(
          createWGSLKernelNode<Parameters>(graph, {
            id: `${id}-segments`,
            operation: OPERATION,
            variant: 'segments',
            bindings: [
              {
                name: 'nodePositions',
                view: nodePositions,
                type: 'f32',
                access: 'read'
              },
              {
                name: 'edgeSources',
                view: sources,
                type: 'u32',
                access: 'read'
              },
              {
                name: 'edgeTargets',
                view: edgeTargets,
                type: 'u32',
                access: 'read'
              },
              {
                name: 'segmentStarts',
                view: segmentStarts,
                type: 'f32',
                access: 'read_write'
              },
              {
                name: 'segmentEnds',
                view: segmentEnds,
                type: 'f32',
                access: 'read_write'
              }
            ],
            invocationCount: edgeCount,
            declarations: `${nodeDeclarations}
fn readNode(node: u32) -> vec2<f32> {
  return vec2<f32>(nodePositions[nodePositionsOffset + node * 2u], nodePositions[nodePositionsOffset + node * 2u + 1u]);
}`,
            body: `let sourceNode = edgeSources[edgeSourcesOffset + index];
  let targetNode = edgeTargets[edgeTargetsOffset + index];
  // WGSL const evaluation rejects an infinite value, so the lon/lat sentinel is built from runtime
  // bits (a var), never from a literal bitcast.
  var infinityBits = 0x7f800000u;
  var start = vec2<f32>(${
    this.spatialContext.coordinateSpace === 'longitude-latitude'
      ? 'bitcast<f32>(infinityBits)'
      : '1.0e30'
  });
  var end = start;
  if (sourceNode < NODE_COUNT && targetNode < NODE_COUNT) {
    start = readNode(sourceNode);
    end = readNode(targetNode);
  }
  segmentStarts[segmentStartsOffset + index * 2u] = start.x;
  segmentStarts[segmentStartsOffset + index * 2u + 1u] = start.y;
  segmentEnds[segmentEndsOffset + index * 2u] = end.x;
  segmentEnds[segmentEndsOffset + index * 2u + 1u] = end.y;`
          })
        );
      }
      nodes.push(
        ...new GPUNearestFeatureJoin({
          id: `${id}-join`,
          spatialContext: this.spatialContext,
          queries: {kind: 'points', positions: props.positions},
          features: {
            kind: 'segments',
            starts: segmentStarts,
            ends: segmentEnds
          },
          radius: props.maximumSnapDistance,
          candidateCapacity: props.candidateCapacity,
          spatialSort: props.spatialSort,
          output: {nearestFeatureIds: joinRows, status: {overflow: output.status!.overflow!}}
        }).getCommandNodes(graph)
      );
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-project`,
          operation: OPERATION,
          variant: 'project',
          bindings: [
            ...projectBindings,
            {name: 'joinRows', view: joinRows, type: 'u32', access: 'read'},
            ...projectOutputs
          ],
          invocationCount: pointCount,
          declarations: `${nodeDeclarations}
${segmentHelpers}`,
          body: `let point = vec2<f32>(points[pointsOffset + index * 2u], points[pointsOffset + index * 2u + 1u]);
  var bestEdge = joinRows[joinRowsOffset + index];
  var bestFraction = 0.0;
  var bestDistance = 0.0;
  if (bestEdge < EDGE_COUNT) {
    let sourceNode = edgeSources[edgeSourcesOffset + bestEdge];
    let targetNode = edgeTargets[edgeTargetsOffset + bestEdge];
    let start = readNode(sourceNode);
    let end = readNode(targetNode);
    bestFraction = getFraction(point, start, end);
    bestDistance = getDistance(point, start, end, bestFraction);
  } else {
    bestEdge = NONE;
  }
  ${writeResult}`
        })
      );
    } else {
      const scanBindings = [...projectBindings, ...projectOutputs];
      if (props.maximumSnapDistance) {
        scanBindings.push({
          name: 'maximumSnapDistance',
          view: props.maximumSnapDistance,
          type: 'f32',
          access: 'read'
        });
      }
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-scan`,
          operation: OPERATION,
          variant: 'scan',
          bindings: scanBindings,
          invocationCount: pointCount,
          declarations: `${nodeDeclarations}
${segmentHelpers}`,
          body: `let point = vec2<f32>(points[pointsOffset + index * 2u], points[pointsOffset + index * 2u + 1u]);
  ${props.maximumSnapDistance ? 'let limit = maximumSnapDistance[maximumSnapDistanceOffset];' : ''}
  var bestEdge = NONE;
  var bestFraction = 0.0;
  var bestDistance = 0.0;
  for (var edge = 0u; edge < EDGE_COUNT; edge++) {
    let sourceNode = edgeSources[edgeSourcesOffset + edge];
    let targetNode = edgeTargets[edgeTargetsOffset + edge];
    if (sourceNode >= NODE_COUNT || targetNode >= NODE_COUNT) {
      continue;
    }
    let start = readNode(sourceNode);
    let end = readNode(targetNode);
    let fraction = getFraction(point, start, end);
    let distance = getDistance(point, start, end, fraction);
    // Strict less keeps the smallest edge row on ties; NaN never passes.
    if (${props.maximumSnapDistance ? 'distance <= limit' : 'distance >= 0.0'} && (bestEdge == NONE || distance < bestDistance)) {
      bestEdge = edge;
      bestFraction = fraction;
      bestDistance = distance;
    }
  }
  ${writeResult}`
        })
      );
    }

    const needsCosts = Boolean(output.sourceCosts || output.targetCosts || output.startNodes);
    const sourceCosts =
      output.sourceCosts ??
      (needsCosts ? createTransientView(graph, `${id}-source-costs`, 'float32', pointCount) : null);
    const targetCosts =
      output.targetCosts ??
      (needsCosts ? createTransientView(graph, `${id}-target-costs`, 'float32', pointCount) : null);
    if (sourceCosts && targetCosts) {
      const costBindings: WGSLKernelBinding[] = [
        {
          name: 'snappedEdges',
          view: output.snappedEdges,
          type: 'u32',
          access: 'read'
        },
        {name: 'fractions', view: fractions, type: 'f32', access: 'read'},
        {name: 'edgeSources', view: sources, type: 'u32', access: 'read'},
        {
          name: 'edgeTargets',
          view: edgeTargets,
          type: 'u32',
          access: 'read'
        },
        edgeCosts
          ? {
              name: 'edgeCosts',
              view: edgeCosts,
              type: 'f32',
              access: 'read'
            }
          : {
              name: 'nodePositions',
              view: nodePositions,
              type: 'f32',
              access: 'read'
            },
        {
          name: 'sourceCosts',
          view: sourceCosts,
          type: 'f32',
          access: 'read_write'
        },
        {
          name: 'targetCosts',
          view: targetCosts,
          type: 'f32',
          access: 'read_write'
        }
      ];
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-costs`,
          operation: OPERATION,
          variant: 'costs',
          bindings: costBindings,
          invocationCount: pointCount,
          declarations: `${nodeDeclarations}
${edgeCosts ? '' : `${readNodeWGSL}\n${metricWGSL}`}`,
          body: `let edge = snappedEdges[snappedEdgesOffset + index];
  if (edge >= EDGE_COUNT) {
    sourceCosts[sourceCostsOffset + index] = -1.0;
    targetCosts[targetCostsOffset + index] = -1.0;
    return;
  }
  let fraction = fractions[fractionsOffset + index];
  ${
    edgeCosts
      ? 'let edgeCost = edgeCosts[edgeCostsOffset + edge];'
      : `let edgeCost = networkDistance(
    readNode(edgeSources[edgeSourcesOffset + edge]),
    readNode(edgeTargets[edgeTargetsOffset + edge])
  );`
  }
  sourceCosts[sourceCostsOffset + index] = fraction * edgeCost;
  targetCosts[targetCostsOffset + index] = (1.0 - fraction) * edgeCost;`
        })
      );
    }
    if (output.snappedPositions) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-positions`,
          operation: OPERATION,
          variant: 'positions',
          bindings: [
            {name: 'points', view: props.positions, type: 'f32', access: 'read'},
            {
              name: 'nodePositions',
              view: nodePositions,
              type: 'f32',
              access: 'read'
            },
            {name: 'edgeSources', view: sources, type: 'u32', access: 'read'},
            {
              name: 'edgeTargets',
              view: edgeTargets,
              type: 'u32',
              access: 'read'
            },
            {
              name: 'snappedEdges',
              view: output.snappedEdges,
              type: 'u32',
              access: 'read'
            },
            {name: 'fractions', view: fractions, type: 'f32', access: 'read'},
            {
              name: 'snappedPositions',
              view: output.snappedPositions,
              type: 'f32',
              access: 'read_write'
            }
          ],
          invocationCount: pointCount,
          declarations: `${nodeDeclarations}
${readNodeWGSL}
${metricWGSL}`,
          body: `var position = vec2<f32>(points[pointsOffset + index * 2u], points[pointsOffset + index * 2u + 1u]);
  let edge = snappedEdges[snappedEdgesOffset + index];
  if (edge < EDGE_COUNT) {
    let start = readNode(edgeSources[edgeSourcesOffset + edge]);
    let end = readNode(edgeTargets[edgeTargetsOffset + edge]);
    position = networkInterpolate(start, end, fractions[fractionsOffset + index]);
  }
  snappedPositions[snappedPositionsOffset + index * 2u] = position.x;
  snappedPositions[snappedPositionsOffset + index * 2u + 1u] = position.y;`
        })
      );
    }
    if (output.startNodes && output.startCosts && sourceCosts && targetCosts) {
      const useSource = this.startDirection !== 'forward';
      const useTarget = this.startDirection !== 'reverse';
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-seeds`,
          operation: OPERATION,
          variant: `seeds-${this.startDirection}`,
          bindings: [
            {
              name: 'snappedEdges',
              view: output.snappedEdges,
              type: 'u32',
              access: 'read'
            },
            {name: 'edgeSources', view: sources, type: 'u32', access: 'read'},
            {
              name: 'edgeTargets',
              view: edgeTargets,
              type: 'u32',
              access: 'read'
            },
            {
              name: 'sourceCosts',
              view: sourceCosts,
              type: 'f32',
              access: 'read'
            },
            {
              name: 'targetCosts',
              view: targetCosts,
              type: 'f32',
              access: 'read'
            },
            {
              name: 'seedNodes',
              view: output.startNodes,
              type: 'u32',
              access: 'read_write'
            },
            {
              name: 'seedCosts',
              view: output.startCosts,
              type: 'f32',
              access: 'read_write'
            }
          ],
          invocationCount: pointCount,
          declarations: nodeDeclarations,
          body: `let edge = snappedEdges[snappedEdgesOffset + index];
  let isSnapped = edge < EDGE_COUNT;
  let useSource = isSnapped && ${useSource};
  let useTarget = isSnapped && ${useTarget};
  let sourceNode = select(0u, edgeSources[edgeSourcesOffset + min(edge, EDGE_COUNT - 1u)], isSnapped);
  let targetNode = select(0u, edgeTargets[edgeTargetsOffset + min(edge, EDGE_COUNT - 1u)], isSnapped);
  seedNodes[seedNodesOffset + index * 2u] = select(NONE, sourceNode, useSource);
  seedCosts[seedCostsOffset + index * 2u] = select(-1.0, sourceCosts[sourceCostsOffset + index], useSource);
  seedNodes[seedNodesOffset + index * 2u + 1u] = select(NONE, targetNode, useTarget);
  seedCosts[seedCostsOffset + index * 2u + 1u] = select(-1.0, targetCosts[targetCostsOffset + index], useTarget);`
        })
      );
    }
    return nodes;
  }
}

/** Returns every read-only view of a snapping contributor. */
function getInputs(props: GPUNetworkSnappingProps): (GraphDataView | undefined)[] {
  return [
    props.positions,
    ...getGPUNetworkTopologyViews(props.topology),
    props.maximumSnapDistance
  ];
}

/** Returns every writable view of a snapping contributor. */
function getOutputs(props: GPUNetworkSnappingProps): (GraphDataView | undefined)[] {
  const {output} = props;
  return [
    output.snappedEdges,
    output.snapFractions,
    output.snapDistances,
    output.sourceCosts,
    output.targetCosts,
    output.snappedPositions,
    output.startNodes,
    output.startCosts,
    output.status?.overflow
  ];
}
