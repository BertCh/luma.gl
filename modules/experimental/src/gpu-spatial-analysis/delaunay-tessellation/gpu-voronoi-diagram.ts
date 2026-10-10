// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUCompaction,
  validatePackedUint32View,
  validatePackedView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {validateGraphViewsBelongToGraph} from '../../utils/gpu-contributor-utils';
import {createWGSLKernelNode} from '../../utils/wgsl-kernel-nodes';
import {validateGPUStatusPort, type GPUBoundedResultStatusPort} from '../contracts/status';

/** Compact, clipped Voronoi edge segments derived from Delaunay triangle adjacency. */
export type GPUVoronoiDiagramOutput = {
  /** Segment endpoints `(startX, startY, endX, endY)`. */
  segments: GraphDataView<'float32x4'>;
  /** The two Delaunay sites whose bisector owns each segment. */
  siteIds: GraphDataView<'uint32x2'>;
  status: GPUBoundedResultStatusPort & {requiredCount: GraphDataView<'uint32'>};
};

/** Properties for {@link GPUVoronoiDiagram}. */
export type GPUVoronoiDiagramProps = {
  id?: string;
  positions: GraphDataView<'float32x2'>;
  /** Delaunay triples; only the prefix selected by `triangleCount` is read. */
  triangles: GraphDataView<'uint32x3'>;
  triangleCount: GraphDataView<'uint32'>;
  /** One dynamic `[minimumX, minimumY, maximumX, maximumY]` row for hull-ray clipping. */
  clipBounds: GraphDataView<'float32x4'>;
  output: GPUVoronoiDiagramOutput;
};

/** Maximum number of Voronoi segments emitted from a triangle capacity. */
export function getVoronoiMaximumSegmentCount(triangleCapacity: number): number {
  if (!Number.isSafeInteger(triangleCapacity) || triangleCapacity < 0) {
    throw new Error('triangleCapacity must be a non-negative safe integer');
  }
  return 3 * triangleCapacity;
}

/**
 * Derives vector Voronoi edges from accepted Delaunay topology entirely on the GPU.
 *
 * Internal edges connect adjacent triangle circumcenters. Hull edges extend the outward bisector
 * to the dynamic clip bounds. Triangle and local-edge order provide stable compact output order;
 * each internal edge is emitted by the lower triangle row. Degenerate triangles are rejected and
 * counted in `invalidCount` when that canonical status field is supplied.
 */
export class GPUVoronoiDiagram implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPUVoronoiDiagramProps;

  constructor(props: GPUVoronoiDiagramProps) {
    this.id = props.id ?? 'voronoi-diagram';
    this.props = props;
    const {id} = this;
    validatePackedView(props.positions, ['float32x2'], `${id} positions`);
    validatePackedView(props.triangles, ['uint32x3'], `${id} triangles`);
    validatePackedUint32View(props.triangleCount, `${id} triangleCount`);
    if (props.triangleCount.length < 1) {
      throw new Error(`${id} triangleCount must contain one row`);
    }
    validatePackedView(props.clipBounds, ['float32x4'], `${id} clipBounds`);
    if (props.clipBounds.length !== 1) {
      throw new Error(`${id} clipBounds must contain one row`);
    }
    validatePackedView(props.output.segments, ['float32x4'], `${id} output.segments`);
    validatePackedView(props.output.siteIds, ['uint32x2'], `${id} output.siteIds`);
    if (props.output.segments.length !== props.output.siteIds.length) {
      throw new Error(`${id} output segment and site capacities must match`);
    }
    validateGPUStatusPort(`${id} output.status`, props.output.status);
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {output} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      props.positions,
      props.triangles,
      props.triangleCount,
      props.clipBounds,
      output.segments,
      output.siteIds,
      output.status.count,
      output.status.requiredCount,
      output.status.overflow,
      output.status.invalidCount
    ]);
    const triangleCapacity = props.triangles.length;
    const candidateCapacity = 3 * triangleCapacity;
    const centers = createTransientView(graph, `${id}-centers`, 'float32', 3 * triangleCapacity);
    const candidateIds = createTransientView(
      graph,
      `${id}-candidate-ids`,
      'uint32',
      candidateCapacity
    );
    const candidateFlags = createTransientView(
      graph,
      `${id}-candidate-flags`,
      'uint32',
      candidateCapacity
    );
    const compactCandidateIds = createTransientView(
      graph,
      `${id}-compact-candidate-ids`,
      'uint32',
      candidateCapacity
    );
    const invalidCount =
      output.status.invalidCount ?? createTransientView(graph, `${id}-invalid-count`, 'uint32', 1);
    const nodes: GPUCommandNode<Parameters>[] = [
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-initialize`,
        operation: 'GPUVoronoiDiagram',
        variant: 'initialize',
        bindings: [{name: 'invalidCount', view: invalidCount, type: 'u32', access: 'read_write'}],
        invocationCount: 1,
        body: 'invalidCount[invalidCountOffset] = 0u;'
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-circumcenters`,
        operation: 'GPUVoronoiDiagram',
        variant: 'parallel-circumcenters',
        bindings: [
          {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
          {name: 'triangles', view: props.triangles, type: 'u32', access: 'read'},
          {name: 'triangleCountIn', view: props.triangleCount, type: 'u32', access: 'read'},
          {name: 'centers', view: centers, type: 'f32', access: 'read_write'},
          {name: 'invalidCount', view: invalidCount, type: 'atomic<u32>', access: 'read_write'}
        ],
        invocationCount: triangleCapacity,
        declarations: getVoronoiWGSL({
          pointCount: props.positions.length,
          triangleCapacity
        }),
        body: `let triangleCount = min(triangleCountIn[triangleCountInOffset], TRIANGLE_CAPACITY);
  let center = circumcenter(index);
  let base = centersOffset + 3u * index;
  centers[base] = center.x;
  centers[base + 1u] = center.y;
  centers[base + 2u] = select(0.0, center.z, index < triangleCount);
  if (index < triangleCount && center.z == 0.0) { atomicAdd(&invalidCount[invalidCountOffset], 1u); }`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-classify-edges`,
        operation: 'GPUVoronoiDiagram',
        variant: 'parallel-edge-classification',
        bindings: [
          {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
          {name: 'triangles', view: props.triangles, type: 'u32', access: 'read'},
          {name: 'triangleCountIn', view: props.triangleCount, type: 'u32', access: 'read'},
          {name: 'centers', view: centers, type: 'f32', access: 'read'},
          {name: 'candidateIds', view: candidateIds, type: 'u32', access: 'read_write'},
          {name: 'candidateFlags', view: candidateFlags, type: 'u32', access: 'read_write'}
        ],
        invocationCount: candidateCapacity,
        declarations: getVoronoiEdgeWGSL({
          pointCount: props.positions.length,
          triangleCapacity
        }),
        body: `let triangleCount = min(triangleCountIn[triangleCountInOffset], TRIANGLE_CAPACITY);
  let triangle = index / 3u;
  let edge = index % 3u;
  candidateIds[candidateIdsOffset + index] = index;
  var accepted = triangle < triangleCount && centers[centersOffset + 3u * triangle + 2u] != 0.0;
  if (accepted) {
    let a = vertex(triangle, edge);
    let b = vertex(triangle, (edge + 1u) % 3u);
    accepted = a < POINT_COUNT && b < POINT_COUNT;
    if (accepted) {
      let neighbor = findNeighbor(triangle, a, b, triangleCount);
      accepted = neighbor >= triangleCount || (neighbor > triangle && centers[centersOffset + 3u * neighbor + 2u] != 0.0);
    }
  }
  candidateFlags[candidateFlagsOffset + index] = select(0u, 1u, accepted);`
      })
    ];
    nodes.push(
      ...new GPUCompaction({
        id: `${id}-compact-edges`,
        input: candidateIds,
        flags: candidateFlags,
        output: compactCandidateIds,
        count: output.status.requiredCount
      }).getCommandNodes(graph),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-materialize`,
        operation: 'GPUVoronoiDiagram',
        variant: 'parallel-materialization',
        bindings: [
          {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
          {name: 'triangles', view: props.triangles, type: 'u32', access: 'read'},
          {name: 'triangleCountIn', view: props.triangleCount, type: 'u32', access: 'read'},
          {name: 'clipBounds', view: props.clipBounds, type: 'f32', access: 'read'},
          {name: 'centers', view: centers, type: 'f32', access: 'read'},
          {name: 'compactCandidateIds', view: compactCandidateIds, type: 'u32', access: 'read'},
          {name: 'requiredCount', view: output.status.requiredCount, type: 'u32', access: 'read'},
          {name: 'segmentsOut', view: output.segments, type: 'f32', access: 'read_write'}
        ],
        invocationCount: output.segments.length,
        declarations: `${getVoronoiEdgeWGSL({
          pointCount: props.positions.length,
          triangleCapacity
        })}
fn centerAt(triangle: u32) -> vec2<f32> {
  return vec2<f32>(centers[centersOffset + 3u * triangle], centers[centersOffset + 3u * triangle + 1u]);
}
fn hullEndpoint(center: vec2<f32>, a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
  let direction = normalize(vec2<f32>(b.y - a.y, a.x - b.x));
  let bounds = vec4<f32>(clipBounds[clipBoundsOffset], clipBounds[clipBoundsOffset + 1u], clipBounds[clipBoundsOffset + 2u], clipBounds[clipBoundsOffset + 3u]);
  var distance = 3.402823466e+38;
  if (direction.x > 0.0) { distance = min(distance, (bounds.z - center.x) / direction.x); }
  if (direction.x < 0.0) { distance = min(distance, (bounds.x - center.x) / direction.x); }
  if (direction.y > 0.0) { distance = min(distance, (bounds.w - center.y) / direction.y); }
  if (direction.y < 0.0) { distance = min(distance, (bounds.y - center.y) / direction.y); }
  return center + max(distance, 0.0) * direction;
}`,
        body: `if (index >= requiredCount[requiredCountOffset]) { return; }
  let candidate = compactCandidateIds[compactCandidateIdsOffset + index];
  let triangle = candidate / 3u;
  let edge = candidate % 3u;
  let a = vertex(triangle, edge);
  let b = vertex(triangle, (edge + 1u) % 3u);
  let triangleCount = min(triangleCountIn[triangleCountInOffset], TRIANGLE_CAPACITY);
  let neighbor = findNeighbor(triangle, a, b, triangleCount);
  let center = centerAt(triangle);
  var endpoint = hullEndpoint(center, site(a), site(b));
  if (neighbor < triangleCount) { endpoint = centerAt(neighbor); }
  let segmentBase = segmentsOutOffset + 4u * index;
  segmentsOut[segmentBase] = center.x;
  segmentsOut[segmentBase + 1u] = center.y;
  segmentsOut[segmentBase + 2u] = endpoint.x;
  segmentsOut[segmentBase + 3u] = endpoint.y;`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-materialize-site-ids`,
        operation: 'GPUVoronoiDiagram',
        variant: 'parallel-site-id-materialization',
        bindings: [
          {name: 'triangles', view: props.triangles, type: 'u32', access: 'read'},
          {name: 'compactCandidateIds', view: compactCandidateIds, type: 'u32', access: 'read'},
          {name: 'requiredCount', view: output.status.requiredCount, type: 'u32', access: 'read'},
          {name: 'siteIdsOut', view: output.siteIds, type: 'u32', access: 'read_write'}
        ],
        invocationCount: output.siteIds.length,
        declarations: `fn vertex(triangle: u32, corner: u32) -> u32 {
  return triangles[trianglesOffset + 3u * triangle + corner];
}`,
        body: `if (index >= requiredCount[requiredCountOffset]) { return; }
  let candidate = compactCandidateIds[compactCandidateIdsOffset + index];
  let triangle = candidate / 3u;
  let edge = candidate % 3u;
  let a = vertex(triangle, edge);
  let b = vertex(triangle, (edge + 1u) % 3u);
  let siteBase = siteIdsOutOffset + 2u * index;
  siteIdsOut[siteBase] = min(a, b);
  siteIdsOut[siteBase + 1u] = max(a, b);`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-publish-status`,
        operation: 'GPUVoronoiDiagram',
        variant: 'publish-status',
        bindings: [
          {
            name: 'requiredCountIn',
            view: output.status.requiredCount,
            type: 'u32',
            access: 'read'
          },
          {name: 'countOut', view: output.status.count, type: 'u32', access: 'read_write'},
          {name: 'overflowOut', view: output.status.overflow, type: 'u32', access: 'read_write'}
        ],
        invocationCount: 1,
        declarations: `const OUTPUT_CAPACITY: u32 = ${output.segments.length}u;`,
        body: `let requiredCount = requiredCountIn[requiredCountInOffset];
  countOut[countOutOffset] = min(requiredCount, OUTPUT_CAPACITY);
  overflowOut[overflowOutOffset] = select(0u, 1u, requiredCount > OUTPUT_CAPACITY);`
      })
    );
    return nodes;
  }
}

function getVoronoiWGSL(options: {pointCount: number; triangleCapacity: number}): string {
  return /* wgsl */ `
const POINT_COUNT: u32 = ${options.pointCount}u;
const TRIANGLE_CAPACITY: u32 = ${options.triangleCapacity}u;

fn site(row: u32) -> vec2<f32> {
  return vec2<f32>(positions[positionsOffset + 2u * row], positions[positionsOffset + 2u * row + 1u]);
}
fn vertex(triangle: u32, corner: u32) -> u32 {
  return triangles[trianglesOffset + 3u * triangle + corner];
}
fn circumcenter(triangle: u32) -> vec3<f32> {
  let aRow = vertex(triangle, 0u);
  let bRow = vertex(triangle, 1u);
  let cRow = vertex(triangle, 2u);
  if (aRow >= POINT_COUNT || bRow >= POINT_COUNT || cRow >= POINT_COUNT) {
    return vec3<f32>(0.0, 0.0, 0.0);
  }
  let a = site(aRow);
  let b = site(bRow);
  let c = site(cRow);
  let denominator = 2.0 * (a.x * (b.y - c.y) + b.x * (c.y - a.y) + c.x * (a.y - b.y));
  if (abs(denominator) <= 1e-12) { return vec3<f32>(0.0, 0.0, 0.0); }
  let aa = dot(a, a);
  let bb = dot(b, b);
  let cc = dot(c, c);
  return vec3<f32>(
    (aa * (b.y - c.y) + bb * (c.y - a.y) + cc * (a.y - b.y)) / denominator,
    (aa * (c.x - b.x) + bb * (a.x - c.x) + cc * (b.x - a.x)) / denominator,
    1.0
  );
}
`;
}

function getVoronoiEdgeWGSL(options: {pointCount: number; triangleCapacity: number}): string {
  return /* wgsl */ `
const POINT_COUNT: u32 = ${options.pointCount}u;
const TRIANGLE_CAPACITY: u32 = ${options.triangleCapacity}u;
fn site(row: u32) -> vec2<f32> {
  return vec2<f32>(positions[positionsOffset + 2u * row], positions[positionsOffset + 2u * row + 1u]);
}
fn vertex(triangle: u32, corner: u32) -> u32 {
  return triangles[trianglesOffset + 3u * triangle + corner];
}
fn sharesEdge(triangle: u32, a: u32, b: u32) -> bool {
  var containsA = false;
  var containsB = false;
  for (var corner = 0u; corner < 3u; corner++) {
    let value = vertex(triangle, corner);
    containsA = containsA || value == a;
    containsB = containsB || value == b;
  }
  return containsA && containsB;
}
fn findNeighbor(triangle: u32, a: u32, b: u32, triangleCount: u32) -> u32 {
  for (var candidate = 0u; candidate < triangleCount; candidate++) {
    if (candidate != triangle && sharesEdge(candidate, a, b)) { return candidate; }
  }
  return TRIANGLE_CAPACITY;
}`;
}
