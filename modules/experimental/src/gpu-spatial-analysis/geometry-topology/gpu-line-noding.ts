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
import type {Device} from '@luma.gl/core';
import {createWGSLKernelNode, type WGSLKernelBinding} from '../../utils/wgsl-kernel-nodes';
import {
  validateCapacity,
  validateGraphViewsBelongToGraph,
  validateMaximumCount,
  validatePackedView,
  validatePowerOfTwo
} from '../../utils/gpu-contributor-utils';
import {
  GPULineSplit,
  getGPULineSplitMaximumPieceBound,
  GPU_LINE_SPLIT_SELF_CONTACT
} from '../line-split/index';
import {getGPUSegmentIntersectionMaximumVertexCount} from '../segment-intersection/index';
import type {GPULineGeometryPort, GPUPackedGeometryPort} from '../../analysis-core/contracts/index';
import {GPU_TOPOLOGY_NONE, type GPUNodedSegmentPort} from './topology-types';
import {
  resolveGPUSpatialContext,
  type GPUSpatialContext
} from '../../analysis-core/contracts/spatial-context';

const OPERATION = 'GPULineNoding';

/** Properties for {@link GPULineNoding}. */
export type GPULineNodingProps = {
  /** Prefix for generated node and transient IDs. */
  id?: string;
  /**
   * Planar only (default: native planar coordinates). A longitude-latitude context throws at
   * construction: project positions to a planar frame first, for example with
   * `createGPULocalFrame` (`contracts/local-frame`) and `GPULocalFrameProjection`
   * (`geometry-measures`), and pass the frame's context.
   */
  spatialContext?: GPUSpatialContext;
  /** Arbitrary finite planar linework. */
  lines: GPUPackedGeometryPort<GPULineGeometryPort> & {
    /** Optional source feature ID per line, written to `output.sourceFeatureIds`. */
    sourceIds?: GraphDataView<'uint32'>;
  };
  /**
   * Capacity of the exact segment-intersection event list. The line count plus four times this
   * capacity is at most {@link getGPULineSplitMaximumPieceBound} (4194304 on a default
   * device), and `lines.positions` has at most {@link getGPUSegmentIntersectionMaximumVertexCount}
   * rows.
   */
  intersectionCapacity: number;
  /**
   * Caller-owned atomic noded segments and provenance. `output.status.uncertainCount` is the
   * optional uncertain-predicate counter.
   */
  output: GPUNodedSegmentPort;
  spatialSort?: boolean;
  leafCapacity?: number;
};

/**
 * Throws unless a noding of `vertexCount` vertices on `lineCount` lines with `intersectionCapacity`
 * pairs fits the internal line split: the segment table holds at most
 * {@link getGPUSegmentIntersectionMaximumVertexCount} vertices and the piece table at most
 * {@link getGPULineSplitMaximumPieceBound} bounded pieces (`lines + 4 * intersectionCapacity`).
 * Used by every contributor that nodes lines, under its own ID.
 *
 * @internal
 */
export function validateLineNodingBounds(
  id: string,
  device: Device,
  vertexCount: number,
  lineCount: number,
  intersectionCapacity: number
): void {
  validateMaximumCount(
    id,
    'lines.positions rows',
    vertexCount,
    getGPUSegmentIntersectionMaximumVertexCount(device)
  );
  validateMaximumCount(
    id,
    'lines + 4 x intersectionCapacity',
    lineCount + 4 * intersectionCapacity,
    getGPULineSplitMaximumPieceBound(device)
  );
}

/**
 * Nodes arbitrary finite planar linework and emits one atomic row for every segment between
 * consecutive noded vertices. Crossings, touches and both ends of collinear overlaps reuse the
 * exact-or-uncertain predicates of `GPUSegmentIntersection` through `GPULineSplit`.
 *
 * `requiredCount` is exact when `candidateOverflow` is zero. A candidate overflow (the
 * intersection pair list exceeded `intersectionCapacity`) marks both `candidateOverflow` and final
 * `overflow`; callers must grow `intersectionCapacity` and rerun before treating the requirement or
 * geometry as complete. `overflow` alone (with `candidateOverflow` clear and `requiredCount` within
 * capacity) means the input had segment pairs the exact predicates could not certify, or
 * non-finite vertices: no capacity fixes that.
 */
export class GPULineNoding implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPULineNodingProps;
  /** Resolved spatial context (planar only). */
  readonly spatialContext: GPUSpatialContext;

  constructor(props: GPULineNodingProps) {
    this.id = props.id ?? 'line-noding';
    this.spatialContext = resolveGPUSpatialContext(this.id, props.spatialContext);
    this.props = props;
    const {id} = this;
    const {lines, output} = props;
    if (lines.kind !== 'lines') {
      throw new Error(`${id} lines must be linestring geometry`);
    }
    validatePackedView(lines.positions, ['float32x2'], `${id} lines.positions`);
    validatePackedUint32View(lines.lineOffsets, `${id} lines.lineOffsets`);
    if (lines.lineOffsets.length < 2 || lines.positions.length < 1) {
      throw new Error(`${id} lines require at least one line and one vertex`);
    }
    validateCapacity(id, 'intersectionCapacity', props.intersectionCapacity);
    // Delegates are built later, in getCommandNodes: check forwarded options now, under this id.
    if (props.leafCapacity !== undefined) {
      validatePowerOfTwo(id, 'leafCapacity', props.leafCapacity);
    }
    const capacity = output.endpoints.length;
    validatePackedView(output.endpoints, ['float32x4'], `${id} output.endpoints`);
    for (const [name, view] of [
      ['sourceFeatureIds', output.sourceFeatureIds],
      ['sourceRingIds', output.sourceRingIds],
      ['sourceEdgeIds', output.sourceEdgeIds]
    ] as const) {
      validatePackedUint32View(view, `${id} output.${name}`);
      if (view.length !== capacity) {
        throw new Error(`${id} output.${name} length must equal output.endpoints length`);
      }
    }
    for (const [name, view] of [
      ['sourceStartParameters', output.sourceStartParameters],
      ['sourceEndParameters', output.sourceEndParameters]
    ] as const) {
      validatePackedView(view, ['float32'], `${id} output.${name}`);
      if (view.length !== capacity) {
        throw new Error(`${id} output.${name} length must equal output.endpoints length`);
      }
    }
    for (const [name, view] of Object.entries(output.status)) {
      if (view) {
        validatePackedUint32View(view, `${id} output.status.${name}`);
        if (view.length < 1) {
          throw new Error(`${id} output.status.${name} must contain one row`);
        }
      }
    }
    if (!output.status.requiredCount || !output.status.candidateOverflow) {
      throw new Error(`${id} output.status requires requiredCount and candidateOverflow`);
    }
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {lines, output} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      lines.positions,
      lines.lineOffsets,
      lines.sourceIds,
      output.endpoints,
      output.sourceFeatureIds,
      output.sourceRingIds,
      output.sourceEdgeIds,
      output.sourceStartParameters,
      output.sourceEndParameters,
      ...Object.values(output.status)
    ]);
    validateLineNodingBounds(
      id,
      graph.device,
      lines.positions.length,
      lines.lineOffsets.length - 1,
      props.intersectionCapacity
    );
    const pairCapacity = props.intersectionCapacity;
    const lineCount = lines.lineOffsets.length - 1;
    const pieceCapacity = lineCount + pairCapacity * 4;
    const pieceVertexCapacity = lines.positions.length + pairCapacity * 8;
    const transient = <Format extends 'uint32' | 'float32x2'>(
      name: string,
      format: Format,
      length: number
    ) => createTransientView(graph, `${id}-${name}`, format, length);
    const piecePositions = transient('piece-positions', 'float32x2', pieceVertexCapacity);
    const pieceOffsets = transient('piece-offsets', 'uint32', pieceCapacity + 1);
    const pieceSources = transient('piece-sources', 'uint32', pieceCapacity);
    const pieceCount = transient('piece-count', 'uint32', 1);
    const pieceRequiredCount = transient('piece-required-count', 'uint32', 1);
    const pieceVertexCount = transient('piece-vertex-count', 'uint32', 1);
    const pieceRequiredVertexCount = transient('piece-required-vertex-count', 'uint32', 1);
    const pieceSourceEdges = transient('piece-source-edges', 'uint32', pieceVertexCapacity);
    // Incomplete split output: piece capacity, uncertain pairs or non-finite vertices.
    const splitIncomplete = transient('split-incomplete', 'uint32', 1);
    const nodes: GPUCommandNode<Parameters>[] = [];
    nodes.push(
      ...new GPULineSplit({
        id: `${id}-split`,
        lines,
        intersectionCapacity: pairCapacity,
        spatialSort: props.spatialSort,
        leafCapacity: props.leafCapacity,
        output: {
          status: {uncertainCount: output.status.uncertainCount},
          pieces: {
            geometry: {kind: 'lines', positions: piecePositions, lineOffsets: pieceOffsets},
            sourceIds: pieceSources,
            status: {
              count: pieceCount,
              requiredCount: pieceRequiredCount,
              overflow: splitIncomplete,
              candidateOverflow: output.status.candidateOverflow,
              vertexCount: pieceVertexCount,
              requiredVertexCount: pieceRequiredVertexCount
            },
            vertexSourceEdges: pieceSourceEdges
          }
        }
      }).getCommandNodes(graph)
    );

    const segmentFlags = transient('segment-flags', 'uint32', pieceVertexCapacity);
    const segmentScan = transient('segment-scan', 'uint32', pieceVertexCapacity);
    const commonBindings: WGSLKernelBinding[] = [
      {name: 'pieceOffsets', view: pieceOffsets, type: 'u32', access: 'read'},
      {name: 'pieceCount', view: pieceCount, type: 'u32', access: 'read'},
      {name: 'pieceVertexCount', view: pieceVertexCount, type: 'u32', access: 'read'}
    ];
    const pieceLookupWGSL = `
fn pieceOfVertex(vertex: u32) -> u32 {
  let count = pieceCount[pieceCountOffset];
  var low = 0u;
  var high = count;
  while (low + 1u < high) {
    let middle = (low + high) / 2u;
    if (pieceOffsets[pieceOffsetsOffset + middle] <= vertex) { low = middle; } else { high = middle; }
  }
  return low;
}`;
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-segment-flags`,
        operation: OPERATION,
        variant: 'segment-flags',
        bindings: [
          ...commonBindings,
          {name: 'flags', view: segmentFlags, type: 'u32', access: 'read_write'}
        ],
        invocationCount: pieceVertexCapacity,
        declarations: pieceLookupWGSL,
        body: `var valid = false;
  if (index < pieceVertexCount[pieceVertexCountOffset] && pieceCount[pieceCountOffset] > 0u) {
    let piece = pieceOfVertex(index);
    valid = index + 1u < pieceOffsets[pieceOffsetsOffset + piece + 1u];
  }
  flags[flagsOffset + index] = select(0u, 1u, valid);`
      }),
      ...new GPUScan({
        id: `${id}-segment-scan`,
        input: segmentFlags,
        output: segmentScan,
        mode: 'inclusive'
      }).getCommandNodes(graph)
    );

    const outputCapacity = output.endpoints.length;
    const sourceIdBinding: WGSLKernelBinding[] = lines.sourceIds
      ? [{name: 'inputSourceIds', view: lines.sourceIds, type: 'u32', access: 'read'}]
      : [];
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-initialize`,
        operation: OPERATION,
        variant: 'initialize',
        bindings: [
          {name: 'endpoints', view: output.endpoints, type: 'f32', access: 'read_write'},
          {
            name: 'sourceFeatureIds',
            view: output.sourceFeatureIds,
            type: 'u32',
            access: 'read_write'
          },
          {name: 'sourceRingIds', view: output.sourceRingIds, type: 'u32', access: 'read_write'},
          {name: 'sourceEdgeIds', view: output.sourceEdgeIds, type: 'u32', access: 'read_write'},
          {
            name: 'startParameters',
            view: output.sourceStartParameters,
            type: 'f32',
            access: 'read_write'
          },
          {
            name: 'endParameters',
            view: output.sourceEndParameters,
            type: 'f32',
            access: 'read_write'
          }
        ],
        invocationCount: outputCapacity,
        body: `let endpoint = endpointsOffset + index * 4u;
  endpoints[endpoint] = 0.0; endpoints[endpoint + 1u] = 0.0;
  endpoints[endpoint + 2u] = 0.0; endpoints[endpoint + 3u] = 0.0;
  sourceFeatureIds[sourceFeatureIdsOffset + index] = ${GPU_TOPOLOGY_NONE}u;
  sourceRingIds[sourceRingIdsOffset + index] = ${GPU_TOPOLOGY_NONE}u;
  sourceEdgeIds[sourceEdgeIdsOffset + index] = ${GPU_TOPOLOGY_NONE}u;
  startParameters[startParametersOffset + index] = 0.0;
  endParameters[endParametersOffset + index] = 0.0;`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-write-geometry`,
        operation: OPERATION,
        variant: 'write-geometry',
        bindings: [
          {name: 'flags', view: segmentFlags, type: 'u32', access: 'read'},
          {name: 'scan', view: segmentScan, type: 'u32', access: 'read'},
          {name: 'piecePositions', view: piecePositions, type: 'f32', access: 'read'},
          {name: 'endpoints', view: output.endpoints, type: 'f32', access: 'read_write'}
        ],
        invocationCount: pieceVertexCapacity,
        declarations: `const OUTPUT_CAPACITY: u32 = ${outputCapacity}u;
fn piecePoint(vertex: u32) -> vec2f {
  return vec2f(piecePositions[piecePositionsOffset + vertex * 2u], piecePositions[piecePositionsOffset + vertex * 2u + 1u]);
}`,
        body: `if (flags[flagsOffset + index] == 0u) { return; }
  let outputRow = scan[scanOffset + index] - 1u;
  if (outputRow >= OUTPUT_CAPACITY) { return; }
  let a = piecePoint(index);
  let b = piecePoint(index + 1u);
  let endpoint = endpointsOffset + outputRow * 4u;
  endpoints[endpoint] = a.x; endpoints[endpoint + 1u] = a.y;
  endpoints[endpoint + 2u] = b.x; endpoints[endpoint + 3u] = b.y;`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-write-owners`,
        operation: OPERATION,
        variant: 'write-owners',
        bindings: [
          {name: 'pieceOffsets', view: pieceOffsets, type: 'u32', access: 'read'},
          {name: 'pieceCount', view: pieceCount, type: 'u32', access: 'read'},
          {name: 'flags', view: segmentFlags, type: 'u32', access: 'read'},
          {name: 'scan', view: segmentScan, type: 'u32', access: 'read'},
          {name: 'pieceSources', view: pieceSources, type: 'u32', access: 'read'},
          ...sourceIdBinding,
          {
            name: 'sourceFeatureIds',
            view: output.sourceFeatureIds,
            type: 'u32',
            access: 'read_write'
          },
          {name: 'sourceRingIds', view: output.sourceRingIds, type: 'u32', access: 'read_write'}
        ],
        invocationCount: pieceVertexCapacity,
        declarations: `${pieceLookupWGSL}
const OUTPUT_CAPACITY: u32 = ${outputCapacity}u;`,
        body: `if (flags[flagsOffset + index] == 0u) { return; }
  let outputRow = scan[scanOffset + index] - 1u;
  if (outputRow >= OUTPUT_CAPACITY) { return; }
  let piece = pieceOfVertex(index);
  let line = pieceSources[pieceSourcesOffset + piece];
  sourceFeatureIds[sourceFeatureIdsOffset + outputRow] = ${lines.sourceIds ? 'inputSourceIds[inputSourceIdsOffset + line]' : 'line'};
  sourceRingIds[sourceRingIdsOffset + outputRow] = line;`
      }),
      // The edge is the input edge of least midpoint error, ties to the smaller index. The splitter
      // reports the edge each segment lies on; when that edge and its neighbors are regular and it
      // meets no other segment of its line, only the neighbors (and, for closed lines, the far
      // end) can tie or beat it, so they are the only candidates. Otherwise every edge is scanned.
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-write-edge-provenance`,
        operation: OPERATION,
        variant: 'write-edge-provenance',
        bindings: [
          {name: 'flags', view: segmentFlags, type: 'u32', access: 'read'},
          {name: 'scan', view: segmentScan, type: 'u32', access: 'read'},
          {name: 'piecePositions', view: piecePositions, type: 'f32', access: 'read'},
          {name: 'pieceSourceEdges', view: pieceSourceEdges, type: 'u32', access: 'read'},
          {name: 'sourceRingIds', view: output.sourceRingIds, type: 'u32', access: 'read'},
          {name: 'inputPositions', view: lines.positions, type: 'f32', access: 'read'},
          {name: 'inputOffsets', view: lines.lineOffsets, type: 'u32', access: 'read'},
          {name: 'sourceEdgeIds', view: output.sourceEdgeIds, type: 'u32', access: 'read_write'}
        ],
        invocationCount: pieceVertexCapacity,
        declarations: `const OUTPUT_CAPACITY: u32 = ${outputCapacity}u;
const SELF_CONTACT: u32 = ${GPU_LINE_SPLIT_SELF_CONTACT}u;
fn inputPoint(vertex: u32) -> vec2f {
  return vec2f(inputPositions[inputPositionsOffset + vertex * 2u], inputPositions[inputPositionsOffset + vertex * 2u + 1u]);
}
fn piecePoint(vertex: u32) -> vec2f {
  return vec2f(piecePositions[piecePositionsOffset + vertex * 2u], piecePositions[piecePositionsOffset + vertex * 2u + 1u]);
}
// Finite and non-zero length: the intersection treats such an edge's successor as adjacent.
fn isRegularEdge(edge: u32) -> bool {
  let delta = inputPoint(edge + 1u) - inputPoint(edge);
  let lengthSquared = dot(delta, delta);
  return lengthSquared > 0.0 && lengthSquared <= 3.4e38;
}
fn considerEdge(edge: u32, midpoint: vec2f, bestEdge: ptr<function, u32>, bestError: ptr<function, f32>) {
  let p = inputPoint(edge);
  let q = inputPoint(edge + 1u);
  let delta = q - p;
  let lengthSquared = dot(delta, delta);
  if (lengthSquared == 0.0) { return; }
  let parameter = clamp(dot(midpoint - p, delta) / lengthSquared, 0.0, 1.0);
  let difference = midpoint - (p + parameter * delta);
  let error = dot(difference, difference);
  if (error < *bestError || (error == *bestError && edge < *bestEdge)) {
    *bestError = error;
    *bestEdge = edge;
  }
}`,
        body: `if (flags[flagsOffset + index] == 0u) { return; }
  let outputRow = scan[scanOffset + index] - 1u;
  if (outputRow >= OUTPUT_CAPACITY) { return; }
  let line = sourceRingIds[sourceRingIdsOffset + outputRow];
  let midpoint = 0.5 * (piecePoint(index) + piecePoint(index + 1u));
  let firstEdge = inputOffsets[inputOffsetsOffset + line];
  let lastVertex = inputOffsets[inputOffsetsOffset + line + 1u];
  let lastEdge = lastVertex - 2u;
  let source = pieceSourceEdges[pieceSourceEdgesOffset + index];
  let edge = source & ~SELF_CONTACT;
  var bestEdge = ${GPU_TOPOLOGY_NONE}u;
  var bestError = 3.4e38;
  var scanAll = (source & SELF_CONTACT) != 0u || edge < firstEdge || edge > lastEdge;
  if (!scanAll) {
    scanAll = !isRegularEdge(edge) ||
      (edge > firstEdge && !isRegularEdge(edge - 1u)) ||
      (edge < lastEdge && !isRegularEdge(edge + 1u)) ||
      (edge == firstEdge && !isRegularEdge(lastEdge)) ||
      (edge == lastEdge && !isRegularEdge(firstEdge));
  }
  if (scanAll) {
    for (var candidate = firstEdge; candidate + 1u < lastVertex; candidate++) {
      considerEdge(candidate, midpoint, &bestEdge, &bestError);
    }
  } else {
    // The minimum of (error, edge) does not depend on visiting order or repeats.
    considerEdge(firstEdge, midpoint, &bestEdge, &bestError);
    if (edge > firstEdge) { considerEdge(edge - 1u, midpoint, &bestEdge, &bestError); }
    considerEdge(edge, midpoint, &bestEdge, &bestError);
    if (edge < lastEdge) { considerEdge(edge + 1u, midpoint, &bestEdge, &bestError); }
    considerEdge(lastEdge, midpoint, &bestEdge, &bestError);
  }
  sourceEdgeIds[sourceEdgeIdsOffset + outputRow] = bestEdge;`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-write-parameters`,
        operation: OPERATION,
        variant: 'write-parameters',
        bindings: [
          {name: 'flags', view: segmentFlags, type: 'u32', access: 'read'},
          {name: 'scan', view: segmentScan, type: 'u32', access: 'read'},
          {name: 'piecePositions', view: piecePositions, type: 'f32', access: 'read'},
          {name: 'inputPositions', view: lines.positions, type: 'f32', access: 'read'},
          {name: 'sourceEdgeIds', view: output.sourceEdgeIds, type: 'u32', access: 'read'},
          {
            name: 'startParameters',
            view: output.sourceStartParameters,
            type: 'f32',
            access: 'read_write'
          },
          {
            name: 'endParameters',
            view: output.sourceEndParameters,
            type: 'f32',
            access: 'read_write'
          }
        ],
        invocationCount: pieceVertexCapacity,
        declarations: `const OUTPUT_CAPACITY: u32 = ${outputCapacity}u;
fn inputPoint(vertex: u32) -> vec2f {
  return vec2f(inputPositions[inputPositionsOffset + vertex * 2u], inputPositions[inputPositionsOffset + vertex * 2u + 1u]);
}
fn piecePoint(vertex: u32) -> vec2f {
  return vec2f(piecePositions[piecePositionsOffset + vertex * 2u], piecePositions[piecePositionsOffset + vertex * 2u + 1u]);
}`,
        body: `if (flags[flagsOffset + index] == 0u) { return; }
  let outputRow = scan[scanOffset + index] - 1u;
  if (outputRow >= OUTPUT_CAPACITY) { return; }
  let a = piecePoint(index);
  let b = piecePoint(index + 1u);
  let bestEdge = sourceEdgeIds[sourceEdgeIdsOffset + outputRow];
  if (bestEdge != ${GPU_TOPOLOGY_NONE}u) {
    let p = inputPoint(bestEdge);
    let q = inputPoint(bestEdge + 1u);
    let delta = q - p;
    let lengthSquared = dot(delta, delta);
    startParameters[startParametersOffset + outputRow] = clamp(dot(a - p, delta) / lengthSquared, 0.0, 1.0);
    endParameters[endParametersOffset + outputRow] = clamp(dot(b - p, delta) / lengthSquared, 0.0, 1.0);
  }`
      })
    );

    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-status`,
        operation: OPERATION,
        variant: 'status',
        bindings: [
          {name: 'scan', view: segmentScan, type: 'u32', access: 'read'},
          {
            name: 'candidateOverflow',
            view: output.status.candidateOverflow,
            type: 'u32',
            access: 'read'
          },
          {name: 'splitIncomplete', view: splitIncomplete, type: 'u32', access: 'read'},
          {name: 'count', view: output.status.count, type: 'u32', access: 'read_write'},
          {
            name: 'requiredCount',
            view: output.status.requiredCount,
            type: 'u32',
            access: 'read_write'
          },
          {name: 'overflow', view: output.status.overflow, type: 'u32', access: 'read_write'}
        ],
        invocationCount: 1,
        declarations: `const OUTPUT_CAPACITY: u32 = ${outputCapacity}u;`,
        body: `let required = scan[scanOffset + ${pieceVertexCapacity - 1}u];
  let candidateIncomplete = candidateOverflow[candidateOverflowOffset];
  requiredCount[requiredCountOffset] = required;
  count[countOffset] = min(required, OUTPUT_CAPACITY);
  overflow[overflowOffset] = select(0u, 1u,
    required > OUTPUT_CAPACITY || candidateIncomplete != 0u || splitIncomplete[splitIncompleteOffset] != 0u);`
      })
    );
    return nodes;
  }
}
