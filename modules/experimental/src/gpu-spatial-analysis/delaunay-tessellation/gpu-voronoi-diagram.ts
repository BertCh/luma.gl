// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {resolveGPUSpatialContext, type GPUSpatialContext} from '../contracts/spatial-context';
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
import {createStableLexicographicIndexSortNodes} from '../../utils/stable-lexicographic-index-sort';
import {createWGSLKernelNode} from '../../utils/wgsl-kernel-nodes';
import {validateGPUStatusPort, type GPUBoundedResultStatusPort} from '../contracts/status';
import {EXACT_ORIENTATION_WGSL} from '../segment-intersection/exact-orientation-wgsl';

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
  /**
   * Planar only. Spatial context of `positions`; omitted means native planar coordinates.
   * Longitude/latitude (or any other non-planar) contexts throw at construction: project
   * positions to a planar frame first, for example with `createGPULocalFrame` and
   * `GPULocalFrameProjection`, and pass the frame's context so its `frame` identity is kept.
   */
  spatialContext?: GPUSpatialContext;
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
 * each internal edge is emitted by the lower triangle row. Triangle adjacency comes from one stable
 * sort of undirected edge keys, so the build costs a sort rather than a scan of every triangle per edge. Exactly collinear triangles, and
 * triangles whose circumcenter overflows f32, are rejected and counted in `invalidCount` when that
 * canonical status field is supplied.
 *
 * Planar only: `spatialContext` defaults to native planar coordinates and a longitude/latitude
 * context throws at construction. Project first, for example with `createGPULocalFrame` and
 * `GPULocalFrameProjection`, and pass the frame's context so its `frame` identity is kept.
 */
export class GPUVoronoiDiagram implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPUVoronoiDiagramProps;
  /** Resolved spatial context (planar). */
  readonly spatialContext: GPUSpatialContext;

  constructor(props: GPUVoronoiDiagramProps) {
    this.id = props.id ?? 'voronoi-diagram';
    this.props = props;
    this.spatialContext = resolveGPUSpatialContext(this.id, props.spatialContext);
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
    // Undirected edge key per candidate half-edge; rows with an out-of-range triangle or vertex get
    // the sentinel `(pointCount, pointCount)` and sort last.
    const edgeMinimumKeys = createTransientView(
      graph,
      `${id}-edge-minimum-keys`,
      'uint32',
      candidateCapacity
    );
    const edgeMaximumKeys = createTransientView(
      graph,
      `${id}-edge-maximum-keys`,
      'uint32',
      candidateCapacity
    );
    const keyBits = Math.min(32, Math.max(1, 32 - Math.clz32(props.positions.length)));
    const edgeSort = createStableLexicographicIndexSortNodes<Parameters>(graph, {
      id: `${id}-edge-sort`,
      operation: 'GPUVoronoiDiagram',
      indices: candidateIds,
      keys: [
        {view: edgeMinimumKeys, keyBits},
        {view: edgeMaximumKeys, keyBits}
      ]
    });
    // Neighbor triangle per candidate edge, found once in classification and reused by materialization.
    const neighbors = createTransientView(graph, `${id}-neighbors`, 'uint32', candidateCapacity);
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
        id: `${id}-edge-keys`,
        operation: 'GPUVoronoiDiagram',
        variant: 'parallel-edge-keys',
        bindings: [
          {name: 'triangles', view: props.triangles, type: 'u32', access: 'read'},
          {name: 'triangleCountIn', view: props.triangleCount, type: 'u32', access: 'read'},
          {name: 'candidateIds', view: candidateIds, type: 'u32', access: 'read_write'},
          {name: 'edgeMinimumKeys', view: edgeMinimumKeys, type: 'u32', access: 'read_write'},
          {name: 'edgeMaximumKeys', view: edgeMaximumKeys, type: 'u32', access: 'read_write'}
        ],
        invocationCount: candidateCapacity,
        declarations: `const POINT_COUNT: u32 = ${props.positions.length}u;
const TRIANGLE_CAPACITY: u32 = ${triangleCapacity}u;`,
        body: `let triangleCount = min(triangleCountIn[triangleCountInOffset], TRIANGLE_CAPACITY);
  let triangle = index / 3u;
  let edge = index % 3u;
  candidateIds[candidateIdsOffset + index] = index;
  var minimum = POINT_COUNT;
  var maximum = POINT_COUNT;
  if (triangle < triangleCount) {
    let a = triangles[trianglesOffset + 3u * triangle + edge];
    let b = triangles[trianglesOffset + 3u * triangle + (edge + 1u) % 3u];
    if (a < POINT_COUNT && b < POINT_COUNT) {
      minimum = min(a, b);
      maximum = max(a, b);
    }
  }
  edgeMinimumKeys[edgeMinimumKeysOffset + index] = minimum;
  edgeMaximumKeys[edgeMaximumKeysOffset + index] = maximum;`
      }),
      ...edgeSort.nodes,
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-classify-edges`,
        operation: 'GPUVoronoiDiagram',
        variant: 'sorted-edge-classification',
        bindings: [
          {name: 'sortedCandidates', view: edgeSort.sortedIndices, type: 'u32', access: 'read'},
          {
            name: 'sortedMinimumKeys',
            view: edgeSort.sortedPrimaryKeys,
            type: 'u32',
            access: 'read'
          },
          {name: 'edgeMaximumKeys', view: edgeMaximumKeys, type: 'u32', access: 'read'},
          {name: 'centers', view: centers, type: 'f32', access: 'read'},
          {name: 'neighbors', view: neighbors, type: 'u32', access: 'read_write'},
          {name: 'candidateFlags', view: candidateFlags, type: 'u32', access: 'read_write'}
        ],
        invocationCount: candidateCapacity,
        declarations: `const POINT_COUNT: u32 = ${props.positions.length}u;
const TRIANGLE_CAPACITY: u32 = ${triangleCapacity}u;
const CANDIDATE_COUNT: u32 = ${candidateCapacity}u;
fn sortedTriangle(row: u32) -> u32 {
  return sortedCandidates[sortedCandidatesOffset + row] / 3u;
}
fn sortedMaximumKey(row: u32) -> u32 {
  return edgeMaximumKeys[edgeMaximumKeysOffset + sortedCandidates[sortedCandidatesOffset + row]];
}
fn sortedKeyEquals(row: u32, minimum: u32, maximum: u32) -> bool {
  return sortedMinimumKeys[sortedMinimumKeysOffset + row] == minimum && sortedMaximumKey(row) == maximum;
}
fn isValidCenter(triangle: u32) -> bool {
  return centers[centersOffset + 3u * triangle + 2u] != 0.0;
}`,
        // One thread per sorted half-edge. Equal undirected keys are contiguous and, because the
        // sort is stable over identity ids, ordered by candidate id and hence by triangle. The
        // lowest other triangle in the run is the neighbor the former all-triangle scan returned.
        body: `let candidate = sortedCandidates[sortedCandidatesOffset + index];
  let triangle = candidate / 3u;
  let minimum = sortedMinimumKeys[sortedMinimumKeysOffset + index];
  var neighbor = TRIANGLE_CAPACITY;
  var accepted = false;
  if (minimum < POINT_COUNT) {
    let maximum = sortedMaximumKey(index);
    // Lower bound of this key's run in [0, index].
    var low = 0u;
    var high = index;
    while (low < high) {
      let middle = (low + high) >> 1u;
      let middleMinimum = sortedMinimumKeys[sortedMinimumKeysOffset + middle];
      if (middleMinimum < minimum || (middleMinimum == minimum && sortedMaximumKey(middle) < maximum)) {
        low = middle + 1u;
      } else {
        high = middle;
      }
    }
    // Skip this triangle's own rows (at most three, contiguous) at the head of the run.
    var row = low;
    while (row < CANDIDATE_COUNT && sortedTriangle(row) == triangle && sortedKeyEquals(row, minimum, maximum)) {
      row++;
    }
    if (row < CANDIDATE_COUNT && sortedKeyEquals(row, minimum, maximum)) {
      neighbor = sortedTriangle(row);
    }
    accepted = isValidCenter(triangle) &&
      (neighbor == TRIANGLE_CAPACITY || (neighbor > triangle && isValidCenter(neighbor)));
  }
  neighbors[neighborsOffset + candidate] = neighbor;
  candidateFlags[candidateFlagsOffset + candidate] = select(0u, 1u, accepted);`
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
          {name: 'clipBounds', view: props.clipBounds, type: 'f32', access: 'read'},
          {name: 'centers', view: centers, type: 'f32', access: 'read'},
          {name: 'compactCandidateIds', view: compactCandidateIds, type: 'u32', access: 'read'},
          {name: 'neighbors', view: neighbors, type: 'u32', access: 'read'},
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
  let neighbor = neighbors[neighborsOffset + candidate];
  let center = centerAt(triangle);
  var endpoint = hullEndpoint(center, site(a), site(b));
  if (neighbor < TRIANGLE_CAPACITY) { endpoint = centerAt(neighbor); }
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
${EXACT_ORIENTATION_WGSL}
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
  // Exact orientation rejects only truly collinear triangles, at any coordinate scale. Results 0
  // (collinear) and 2 (non-finite or unrepresentable) both mark the triangle degenerate.
  let orientation = orientSign(a, b, c);
  if (orientation == 0 || orientation == 2) { return vec3<f32>(0.0, 0.0, 0.0); }
  // Solve relative to a: raw-coordinate products cancel catastrophically far from the origin.
  let ab = b - a;
  let ac = c - a;
  let denominator = 2.0 * (ab.x * ac.y - ab.y * ac.x);
  let abLength = dot(ab, ab);
  let acLength = dot(ac, ac);
  let center = a + vec2<f32>(
    (ac.y * abLength - ab.y * acLength) / denominator,
    (ab.x * acLength - ac.x * abLength) / denominator
  );
  // Nearly collinear triangles can overflow; NaN fails both comparisons.
  if (!(abs(center.x) <= ORIENT_FLOAT_MAXIMUM && abs(center.y) <= ORIENT_FLOAT_MAXIMUM)) {
    return vec3<f32>(0.0, 0.0, 0.0);
  }
  return vec3<f32>(center, 1.0);
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
}`;
}
