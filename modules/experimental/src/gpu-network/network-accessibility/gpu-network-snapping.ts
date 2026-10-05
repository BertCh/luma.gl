// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  validatePackedUint32View,
  validatePackedView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {createWGSLKernelNode, type WGSLKernelBinding} from '../../utils/wgsl-kernel-nodes';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../utils/gpu-contributor-utils';
import {GPUNearestFeatureJoin} from '../../gpu-spatial-analysis/spatial-join/index';
import {ACCESSIBILITY_NONE} from './network-accessibility-passes';

const OPERATION = 'GPUNetworkSnapping';

/** Value written to `snappedEdges` and `seedNodes` when a point snaps to no edge. */
export const GPU_NETWORK_SNAPPING_NONE = 0xffffffff;

/** Value written to fraction, distance, and cost outputs when a point snaps to no edge. */
export const GPU_NETWORK_SNAPPING_NO_VALUE = -1;

/**
 * Which edge endpoints become seeds of a snapped point:
 * - `'both'`: the edge source (cost `fraction * edgeCost`) and target (`(1 - fraction) * edgeCost`),
 *   for undirected networks that list both directions of each road.
 * - `'forward'`: only the target, for a point that leaves along a directed edge (forward search).
 * - `'reverse'`: only the source, for a point reached along a directed edge (reverse search).
 */
export type GPUNetworkSnappingSeedDirection = 'both' | 'forward' | 'reverse';

/**
 * Properties for {@link GPUNetworkSnapping}.
 *
 * Compile-time: point, node and edge counts, `candidateCapacity`, `seedDirection`, and which
 * optional views exist. Per-frame: point and node positions, edge lists and costs, and
 * `maxSnapDistance`.
 */
export type GPUNetworkSnappingProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'network-snapping'`. */
  id?: string;
  /** Planar points to snap (origins, facilities, opportunities). */
  points: GraphDataView<'float32x2'>;
  /** Planar node positions, in the same coordinates as `points`. */
  nodePositions: GraphDataView<'float32x2'>;
  /** Edge source node per edge (COO). Exactly one of `edgeSources` and `offsets` is required. */
  edgeSources?: GraphDataView<'uint32'>;
  /** CSR row offsets with `nodeCount + 1` rows, an alternative to `edgeSources`. */
  offsets?: GraphDataView<'uint32'>;
  /** Edge target node per edge (the CSR `neighbors`). Edges with an endpoint out of range are skipped. */
  edgeTargets: GraphDataView<'uint32'>;
  /** Optional travel cost per edge. Defaults to the planar edge length. */
  edgeCosts?: GraphDataView<'float32'>;
  /**
   * Optional one-row per-frame maximum snap distance. Points farther from every edge snap to none.
   * Required with `candidateCapacity`.
   */
  maxSnapDistance?: GraphDataView<'float32'>;
  /**
   * Compile-time candidate capacity. When given, candidate edges come from `GPUNearestFeatureJoin`
   * (a BVH over edge segments probed with `maxSnapDistance`); candidates past the capacity are
   * dropped and raise `overflow`. When omitted, each point scans every edge, which is exact and
   * needs no radius but costs `pointCount * edgeCount`.
   */
  candidateCapacity?: number;
  /** Compile-time. Morton-sort edges before the BVH build, as `GPUNearestFeatureJoin.spatialSort`. */
  spatialSort?: boolean;
  /** Which endpoints `seedNodes` lists. Defaults to `'both'`. */
  seedDirection?: GPUNetworkSnappingSeedDirection;
  /**
   * Nearest edge row per point, or `GPU_NETWORK_SNAPPING_NONE`. Ties go to the smallest edge row,
   * so a point on a node shared by several edges snaps to the smallest incident edge.
   */
  snappedEdges: GraphDataView<'uint32'>;
  /** Optional fraction along the edge from source (0) to target (1), or -1. */
  snapFractions?: GraphDataView<'float32'>;
  /** Optional planar distance from the point to the edge, or -1. */
  snapDistances?: GraphDataView<'float32'>;
  /** Optional cost from the snapped position to the edge source, `fraction * edgeCost`, or -1. */
  sourceCosts?: GraphDataView<'float32'>;
  /** Optional cost from the snapped position to the edge target, `(1 - fraction) * edgeCost`, or -1. */
  targetCosts?: GraphDataView<'float32'>;
  /** Optional snapped planar position, or the point itself when it snaps to no edge. */
  snappedPositions?: GraphDataView<'float32x2'>;
  /**
   * Optional `2 * pointCount` seed nodes: row `2 * i` is the edge source of point `i` and row
   * `2 * i + 1` the edge target, or `GPU_NETWORK_SNAPPING_NONE` when excluded by `seedDirection`
   * or unsnapped. Feed it to `GPUNetworkCostMatrix` with `seedsPerRow: 2`.
   */
  seedNodes?: GraphDataView<'uint32'>;
  /** Optional `2 * pointCount` seed costs aligned with `seedNodes`, or -1. */
  seedCosts?: GraphDataView<'float32'>;
  /** Optional one-row flag: 1 when the BVH candidate search overflowed. */
  overflow?: GraphDataView<'uint32'>;
};

/**
 * Snaps points onto the nearest edge of a road network, so they can enter a shortest-path search
 * as virtual nodes between the edge endpoints.
 *
 * For every point the contributor reports the nearest edge (smallest edge row on ties), the fraction
 * along it, the planar snap distance, and the two seed costs: `fraction * edgeCost` to the edge
 * source and `(1 - fraction) * edgeCost` to the edge target. The edge cost defaults to the planar
 * edge length. Points beyond the optional maximum snap distance snap to no edge.
 *
 * With `candidateCapacity`, candidate edges come from `GPUNearestFeatureJoin` over the edge
 * segments; without it, every point scans every edge in a single exact kernel.
 */
export class GPUNetworkSnapping implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUNetworkSnappingProps;
  /** Number of points. */
  readonly pointCount: number;
  /** Number of network nodes. */
  readonly nodeCount: number;
  /** Number of edges. */
  readonly edgeCount: number;
  /** Resolved seed direction. */
  readonly seedDirection: GPUNetworkSnappingSeedDirection;

  constructor(props: GPUNetworkSnappingProps) {
    this.id = props.id ?? 'network-snapping';
    this.props = props;
    this.seedDirection = props.seedDirection ?? 'both';
    const {id} = this;
    for (const [name, view] of [
      ['points', props.points],
      ['nodePositions', props.nodePositions],
      ['snappedPositions', props.snappedPositions]
    ] as const) {
      if (view) {
        validatePackedView(view, ['float32x2'], `${id} ${name}`);
      }
    }
    for (const [name, view] of [
      ['edgeSources', props.edgeSources],
      ['offsets', props.offsets],
      ['edgeTargets', props.edgeTargets],
      ['snappedEdges', props.snappedEdges],
      ['seedNodes', props.seedNodes],
      ['overflow', props.overflow]
    ] as const) {
      if (view) {
        validatePackedUint32View(view, `${id} ${name}`);
      }
    }
    for (const [name, view] of [
      ['edgeCosts', props.edgeCosts],
      ['maxSnapDistance', props.maxSnapDistance],
      ['snapFractions', props.snapFractions],
      ['snapDistances', props.snapDistances],
      ['sourceCosts', props.sourceCosts],
      ['targetCosts', props.targetCosts],
      ['seedCosts', props.seedCosts]
    ] as const) {
      if (view) {
        validatePackedView(view, ['float32'], `${id} ${name}`);
      }
    }
    this.pointCount = props.points.length;
    this.nodeCount = props.nodePositions.length;
    this.edgeCount = props.edgeTargets.length;
    if (Boolean(props.edgeSources) === Boolean(props.offsets)) {
      throw new Error(`${id} requires exactly one of edgeSources and offsets`);
    }
    if (props.edgeSources && props.edgeSources.length !== this.edgeCount) {
      throw new Error(`${id} edgeSources length must equal edgeTargets length`);
    }
    if (props.offsets && props.offsets.length !== this.nodeCount + 1) {
      throw new Error(`${id} offsets must contain one more row than nodePositions`);
    }
    if (props.edgeCosts && props.edgeCosts.length !== this.edgeCount) {
      throw new Error(`${id} edgeCosts length must equal edgeTargets length`);
    }
    for (const [name, view] of [
      ['snappedEdges', props.snappedEdges],
      ['snapFractions', props.snapFractions],
      ['snapDistances', props.snapDistances],
      ['sourceCosts', props.sourceCosts],
      ['targetCosts', props.targetCosts],
      ['snappedPositions', props.snappedPositions]
    ] as const) {
      if (view && view.length !== this.pointCount) {
        throw new Error(`${id} ${name} length must equal the point count`);
      }
    }
    for (const [name, view] of [
      ['seedNodes', props.seedNodes],
      ['seedCosts', props.seedCosts]
    ] as const) {
      if (view && view.length !== 2 * this.pointCount) {
        throw new Error(`${id} ${name} length must be twice the point count`);
      }
    }
    if (Boolean(props.seedNodes) !== Boolean(props.seedCosts)) {
      throw new Error(`${id} seedNodes and seedCosts must be given together`);
    }
    for (const [name, view] of [
      ['maxSnapDistance', props.maxSnapDistance],
      ['overflow', props.overflow]
    ] as const) {
      if (view && view.length !== 1) {
        throw new Error(`${id} ${name} must contain exactly one row`);
      }
    }
    if (props.candidateCapacity !== undefined) {
      if (!Number.isSafeInteger(props.candidateCapacity) || props.candidateCapacity < 1) {
        throw new Error(`${id} candidateCapacity must be a positive integer`);
      }
      if (!props.maxSnapDistance) {
        throw new Error(`${id} candidateCapacity requires maxSnapDistance`);
      }
    }
    if (props.overflow && props.candidateCapacity === undefined) {
      throw new Error(`${id} overflow requires candidateCapacity`);
    }
    const outputs = getOutputs(props);
    validateGraphOutputsDisjointFromInputs(id, outputs, getInputs(props));
    const outputBuffers = outputs.filter(view => view !== undefined).map(view => view.buffer);
    if (new Set(outputBuffers).size !== outputBuffers.length) {
      throw new Error(`${id} outputs must use separate buffers`);
    }
  }

  /**
   * Returns optional `edge-sources` (CSR input), the candidate search (`segments` plus the
   * `${id}-join-*` nodes of `GPUNearestFeatureJoin`, then `project`; or one `scan` node), and the
   * optional `costs`, `positions`, and `seeds` nodes.
   */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, pointCount, nodeCount, edgeCount} = this;
    validateGraphViewsBelongToGraph(id, graph, [...getInputs(props), ...getOutputs(props)]);
    const nodes: GPUCommandNode<Parameters>[] = [];
    const nodeDeclarations = `const NODE_COUNT: u32 = ${nodeCount}u;
const EDGE_COUNT: u32 = ${edgeCount}u;
const NONE: u32 = ${ACCESSIBILITY_NONE}u;`;

    let edgeSources = props.edgeSources;
    if (props.offsets) {
      edgeSources = createTransientView(
        graph,
        `${id}-edge-sources`,
        'uint32',
        Math.max(edgeCount, 1)
      );
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-edge-sources`,
          operation: OPERATION,
          variant: 'edge-sources',
          bindings: [
            {
              name: 'offsets',
              view: props.offsets,
              type: 'u32',
              access: 'read'
            },
            {
              name: 'edgeSources',
              view: edgeSources,
              type: 'u32',
              access: 'read_write'
            }
          ],
          invocationCount: nodeCount,
          declarations: nodeDeclarations,
          body: `let end = min(offsets[offsetsOffset + index + 1u], EDGE_COUNT);
  for (var edge = offsets[offsetsOffset + index]; edge < end; edge++) {
    edgeSources[edgeSourcesOffset + edge] = index;
  }`
        })
      );
    }
    const sources = edgeSources!;
    const fractions =
      props.snapFractions ?? createTransientView(graph, `${id}-fractions`, 'float32', pointCount);
    const distances =
      props.snapDistances ?? createTransientView(graph, `${id}-distances`, 'float32', pointCount);
    const segmentHelpers = /* wgsl */ `
fn readNode(node: u32) -> vec2<f32> {
  return vec2<f32>(nodePositions[nodePositionsOffset + node * 2u], nodePositions[nodePositionsOffset + node * 2u + 1u]);
}
// Fraction of the closest point along a -> b; 0 for a degenerate edge.
fn getFraction(point: vec2<f32>, start: vec2<f32>, end: vec2<f32>) -> f32 {
  let segment = end - start;
  let denominator = dot(segment, segment);
  if (denominator > 0.0) {
    return clamp(dot(point - start, segment) / denominator, 0.0, 1.0);
  }
  return 0.0;
}
fn getDistance(point: vec2<f32>, start: vec2<f32>, end: vec2<f32>, fraction: f32) -> f32 {
  let offset = point - (start + fraction * (end - start));
  return sqrt(dot(offset, offset));
}`;
    const projectBindings: WGSLKernelBinding[] = [
      {name: 'points', view: props.points, type: 'f32', access: 'read'},
      {
        name: 'nodePositions',
        view: props.nodePositions,
        type: 'f32',
        access: 'read'
      },
      {name: 'edgeSources', view: sources, type: 'u32', access: 'read'},
      {
        name: 'edgeTargets',
        view: props.edgeTargets,
        type: 'u32',
        access: 'read'
      }
    ];
    const projectOutputs: WGSLKernelBinding[] = [
      {
        name: 'snappedEdges',
        view: props.snappedEdges,
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

    if (props.candidateCapacity !== undefined && props.maxSnapDistance) {
      const segmentStarts = createTransientView(
        graph,
        `${id}-segment-starts`,
        'float32x2',
        edgeCount
      );
      const segmentEnds = createTransientView(graph, `${id}-segment-ends`, 'float32x2', edgeCount);
      const joinRows = createTransientView(graph, `${id}-join-rows`, 'uint32', pointCount);
      if (edgeCount > 0) {
        // Edges with an out-of-range endpoint become far-away segments that never fall inside a
        // finite radius.
        nodes.push(
          createWGSLKernelNode<Parameters>(graph, {
            id: `${id}-segments`,
            operation: OPERATION,
            variant: 'segments',
            bindings: [
              {
                name: 'nodePositions',
                view: props.nodePositions,
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
                view: props.edgeTargets,
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
  var start = vec2<f32>(1.0e30, 1.0e30);
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
          points: props.points,
          features: {
            kind: 'segments',
            starts: segmentStarts,
            ends: segmentEnds
          },
          radius: props.maxSnapDistance,
          candidateCapacity: props.candidateCapacity,
          spatialSort: props.spatialSort,
          nearestFeatureIds: joinRows,
          overflow: props.overflow ?? createTransientView(graph, `${id}-overflow`, 'uint32', 1)
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
      if (props.maxSnapDistance) {
        scanBindings.push({
          name: 'maxSnapDistance',
          view: props.maxSnapDistance,
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
  ${props.maxSnapDistance ? 'let limit = maxSnapDistance[maxSnapDistanceOffset];' : ''}
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
    if (${props.maxSnapDistance ? 'distance <= limit' : 'distance >= 0.0'} && (bestEdge == NONE || distance < bestDistance)) {
      bestEdge = edge;
      bestFraction = fraction;
      bestDistance = distance;
    }
  }
  ${writeResult}`
        })
      );
    }

    const needsCosts = Boolean(props.sourceCosts || props.targetCosts || props.seedNodes);
    const sourceCosts =
      props.sourceCosts ??
      (needsCosts ? createTransientView(graph, `${id}-source-costs`, 'float32', pointCount) : null);
    const targetCosts =
      props.targetCosts ??
      (needsCosts ? createTransientView(graph, `${id}-target-costs`, 'float32', pointCount) : null);
    if (sourceCosts && targetCosts) {
      const costBindings: WGSLKernelBinding[] = [
        {
          name: 'snappedEdges',
          view: props.snappedEdges,
          type: 'u32',
          access: 'read'
        },
        {name: 'fractions', view: fractions, type: 'f32', access: 'read'},
        {name: 'edgeSources', view: sources, type: 'u32', access: 'read'},
        {
          name: 'edgeTargets',
          view: props.edgeTargets,
          type: 'u32',
          access: 'read'
        },
        props.edgeCosts
          ? {
              name: 'edgeCosts',
              view: props.edgeCosts,
              type: 'f32',
              access: 'read'
            }
          : {
              name: 'nodePositions',
              view: props.nodePositions,
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
${
  props.edgeCosts
    ? ''
    : `fn readNode(node: u32) -> vec2<f32> {
  return vec2<f32>(nodePositions[nodePositionsOffset + node * 2u], nodePositions[nodePositionsOffset + node * 2u + 1u]);
}`
}`,
          body: `let edge = snappedEdges[snappedEdgesOffset + index];
  if (edge >= EDGE_COUNT) {
    sourceCosts[sourceCostsOffset + index] = -1.0;
    targetCosts[targetCostsOffset + index] = -1.0;
    return;
  }
  let fraction = fractions[fractionsOffset + index];
  ${
    props.edgeCosts
      ? 'let edgeCost = edgeCosts[edgeCostsOffset + edge];'
      : `let segment = readNode(edgeTargets[edgeTargetsOffset + edge]) - readNode(edgeSources[edgeSourcesOffset + edge]);
  let edgeCost = sqrt(dot(segment, segment));`
  }
  sourceCosts[sourceCostsOffset + index] = fraction * edgeCost;
  targetCosts[targetCostsOffset + index] = (1.0 - fraction) * edgeCost;`
        })
      );
    }
    if (props.snappedPositions) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-positions`,
          operation: OPERATION,
          variant: 'positions',
          bindings: [
            {name: 'points', view: props.points, type: 'f32', access: 'read'},
            {
              name: 'nodePositions',
              view: props.nodePositions,
              type: 'f32',
              access: 'read'
            },
            {name: 'edgeSources', view: sources, type: 'u32', access: 'read'},
            {
              name: 'edgeTargets',
              view: props.edgeTargets,
              type: 'u32',
              access: 'read'
            },
            {
              name: 'snappedEdges',
              view: props.snappedEdges,
              type: 'u32',
              access: 'read'
            },
            {name: 'fractions', view: fractions, type: 'f32', access: 'read'},
            {
              name: 'snappedPositions',
              view: props.snappedPositions,
              type: 'f32',
              access: 'read_write'
            }
          ],
          invocationCount: pointCount,
          declarations: `${nodeDeclarations}
fn readNode(node: u32) -> vec2<f32> {
  return vec2<f32>(nodePositions[nodePositionsOffset + node * 2u], nodePositions[nodePositionsOffset + node * 2u + 1u]);
}`,
          body: `var position = vec2<f32>(points[pointsOffset + index * 2u], points[pointsOffset + index * 2u + 1u]);
  let edge = snappedEdges[snappedEdgesOffset + index];
  if (edge < EDGE_COUNT) {
    let start = readNode(edgeSources[edgeSourcesOffset + edge]);
    let end = readNode(edgeTargets[edgeTargetsOffset + edge]);
    position = start + fractions[fractionsOffset + index] * (end - start);
  }
  snappedPositions[snappedPositionsOffset + index * 2u] = position.x;
  snappedPositions[snappedPositionsOffset + index * 2u + 1u] = position.y;`
        })
      );
    }
    if (props.seedNodes && props.seedCosts && sourceCosts && targetCosts) {
      const useSource = this.seedDirection !== 'forward';
      const useTarget = this.seedDirection !== 'reverse';
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-seeds`,
          operation: OPERATION,
          variant: `seeds-${this.seedDirection}`,
          bindings: [
            {
              name: 'snappedEdges',
              view: props.snappedEdges,
              type: 'u32',
              access: 'read'
            },
            {name: 'edgeSources', view: sources, type: 'u32', access: 'read'},
            {
              name: 'edgeTargets',
              view: props.edgeTargets,
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
              view: props.seedNodes,
              type: 'u32',
              access: 'read_write'
            },
            {
              name: 'seedCosts',
              view: props.seedCosts,
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
    props.points,
    props.nodePositions,
    props.edgeSources,
    props.offsets,
    props.edgeTargets,
    props.edgeCosts,
    props.maxSnapDistance
  ];
}

/** Returns every writable view of a snapping contributor. */
function getOutputs(props: GPUNetworkSnappingProps): (GraphDataView | undefined)[] {
  return [
    props.snappedEdges,
    props.snapFractions,
    props.snapDistances,
    props.sourceCosts,
    props.targetCosts,
    props.snappedPositions,
    props.seedNodes,
    props.seedCosts,
    props.overflow
  ];
}
