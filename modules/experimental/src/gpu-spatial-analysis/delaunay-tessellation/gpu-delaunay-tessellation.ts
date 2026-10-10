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

const OPERATION = 'GPUDelaunayTessellation';

/** Bounded Delaunay triangles and their exact completeness status. */
export type GPUDelaunayTessellationOutput = {
  /** Counter-clockwise triples of input point indices. */
  triangles: GraphDataView<'uint32x3'>;
  /** Canonical bounded-result status. `requiredCount` is exact. */
  status: GPUBoundedResultStatusPort & {requiredCount: GraphDataView<'uint32'>};
  /** Number of input rows skipped because an identical earlier coordinate owns the site. */
  duplicateCount?: GraphDataView<'uint32'>;
};

/** Properties for {@link GPUDelaunayTessellation}. */
export type GPUDelaunayTessellationProps = {
  /** Prefix for generated nodes and transients. Defaults to `'delaunay-tessellation'`. */
  id?: string;
  /** Packed finite planar point coordinates. */
  positions: GraphDataView<'float32x2'>;
  /** Caller-owned bounded output. */
  output: GPUDelaunayTessellationOutput;
};

/** Maximum triangle count for `pointCount` distinct planar sites in general position. */
export function getDelaunayMaximumTriangleCount(pointCount: number): number {
  if (!Number.isSafeInteger(pointCount) || pointCount < 0) {
    throw new Error('pointCount must be a non-negative safe integer');
  }
  return Math.max(0, 2 * pointCount - 5);
}

/**
 * Builds a planar Delaunay triangulation on the GPU with deterministic input-order tie breaking.
 *
 * One bounded Bowyer-Watson invocation performs the topology update while applications obtain
 * throughput by placing independent point sets in separate graphs. Exact duplicate coordinates use
 * first-row ownership. Collinear inputs produce zero triangles. Cocircular sites are accepted into
 * the current cavity, making the selected diagonal stable for a fixed input order. The output is
 * capacity bounded, but `status.requiredCount` always reports the exact final triangle count.
 *
 * This contributor deliberately accepts one packed point set: partitioned inputs must select seam
 * ownership before tessellation, because independently triangulated halos do not define a global
 * Delaunay complex.
 */
export class GPUDelaunayTessellation implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPUDelaunayTessellationProps;

  constructor(props: GPUDelaunayTessellationProps) {
    this.id = props.id ?? 'delaunay-tessellation';
    this.props = props;
    const {id} = this;
    const {output} = props;
    validatePackedView(props.positions, ['float32x2'], `${id} positions`);
    validatePackedView(output.triangles, ['uint32x3'], `${id} output.triangles`);
    validateGPUStatusPort(`${id} output.status`, output.status);
    if (!output.status.requiredCount) {
      throw new Error(`${id} output.status.requiredCount is required`);
    }
    if (output.duplicateCount) {
      validatePackedUint32View(output.duplicateCount, `${id} output.duplicateCount`);
      if (output.duplicateCount.length < 1) {
        throw new Error(`${id} output.duplicateCount must contain one row`);
      }
    }
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {output} = props;
    const {status} = output;
    validateGraphViewsBelongToGraph(id, graph, [
      props.positions,
      output.triangles,
      status.count,
      status.requiredCount,
      status.overflow,
      status.invalidCount,
      output.duplicateCount
    ]);
    const pointCount = props.positions.length;
    // Active planar triangulations need at most 2n + 1 triangles including the supertriangle;
    // the extra guard rows turn an unexpected numeric/topology failure into explicit status.
    const scratchTriangleCapacity = Math.max(2 * pointCount + 8, 8);
    const trianglesA = createTransientView(
      graph,
      `${id}-triangles-a`,
      'uint32',
      4 * scratchTriangleCapacity
    );
    const trianglesB = createTransientView(
      graph,
      `${id}-triangles-b`,
      'uint32',
      4 * scratchTriangleCapacity
    );
    const state = createTransientView(graph, `${id}-state`, 'uint32', 2);
    const bounds = createTransientView(graph, `${id}-bounds`, 'float32', 4);
    const duplicateFlags = createTransientView(
      graph,
      `${id}-duplicate-flags`,
      'uint32',
      Math.max(pointCount, 1)
    );
    const badFlags = createTransientView(
      graph,
      `${id}-bad-flags`,
      'uint32',
      scratchTriangleCapacity
    );
    const boundaryFlags = createTransientView(
      graph,
      `${id}-boundary-flags`,
      'uint32',
      3 * scratchTriangleCapacity
    );
    const triangleIds = createTransientView(
      graph,
      `${id}-triangle-ids`,
      'uint32',
      scratchTriangleCapacity
    );
    const resultFlags = createTransientView(
      graph,
      `${id}-result-flags`,
      'uint32',
      scratchTriangleCapacity
    );
    const compactTriangleIds = createTransientView(
      graph,
      `${id}-compact-triangle-ids`,
      'uint32',
      scratchTriangleCapacity
    );
    const invalidCount =
      status.invalidCount ?? createTransientView(graph, `${id}-invalid-count`, 'uint32', 1);
    const duplicateCount =
      output.duplicateCount ?? createTransientView(graph, `${id}-duplicate-count`, 'uint32', 1);
    const sharedDeclarations = getDelaunayCoreWGSL({pointCount, scratchTriangleCapacity});
    const nodes: GPUCommandNode<Parameters>[] = [
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-initialize-status`,
        operation: OPERATION,
        variant: 'initialize-status',
        bindings: [
          {name: 'state', view: state, type: 'u32', access: 'read_write'},
          {name: 'invalidCount', view: invalidCount, type: 'u32', access: 'read_write'},
          {name: 'duplicateCount', view: duplicateCount, type: 'u32', access: 'read_write'}
        ],
        invocationCount: 1,
        body: `state[stateOffset] = 0u;
  state[stateOffset + 1u] = 0u;
  invalidCount[invalidCountOffset] = 0u;
  duplicateCount[duplicateCountOffset] = 0u;`
      })
    ];
    if (pointCount > 0) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-classify-sites`,
          operation: OPERATION,
          variant: 'parallel-site-validation',
          bindings: [
            {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
            {name: 'duplicateFlags', view: duplicateFlags, type: 'u32', access: 'read_write'},
            {name: 'invalidCount', view: invalidCount, type: 'atomic<u32>', access: 'read_write'},
            {
              name: 'duplicateCount',
              view: duplicateCount,
              type: 'atomic<u32>',
              access: 'read_write'
            }
          ],
          invocationCount: pointCount,
          declarations: `const POINT_COUNT: u32 = ${pointCount}u;
fn isFiniteCoordinate(value: f32) -> bool {
  return (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u;
}`,
          body: `let base = positionsOffset + 2u * index;
  let point = vec2<f32>(positions[base], positions[base + 1u]);
  let valid = isFiniteCoordinate(point.x) && isFiniteCoordinate(point.y);
  if (!valid) { atomicAdd(&invalidCount[invalidCountOffset], 1u); }
  var duplicate = false;
  if (valid) {
    for (var earlier = 0u; earlier < index; earlier++) {
      let earlierBase = positionsOffset + 2u * earlier;
      if (positions[earlierBase] == point.x && positions[earlierBase + 1u] == point.y) {
        duplicate = true; break;
      }
    }
  }
  duplicateFlags[duplicateFlagsOffset + index] = select(0u, 1u, duplicate);
  if (duplicate) { atomicAdd(&duplicateCount[duplicateCountOffset], 1u); }`
        })
      );
    }
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-initialize-topology`,
        operation: OPERATION,
        variant: 'initialize-supertriangle',
        bindings: [
          {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
          {name: 'bounds', view: bounds, type: 'f32', access: 'read_write'},
          {name: 'triangles', view: trianglesA, type: 'u32', access: 'read_write'},
          {name: 'state', view: state, type: 'u32', access: 'read_write'},
          {name: 'invalidCount', view: invalidCount, type: 'u32', access: 'read'}
        ],
        invocationCount: 1,
        declarations: `const POINT_COUNT: u32 = ${pointCount}u;`,
        body: `if (POINT_COUNT == 0u) { return; }
  var minimum = vec2<f32>(positions[positionsOffset], positions[positionsOffset + 1u]);
  var maximum = minimum;
  for (var row = 1u; row < POINT_COUNT; row++) {
    let point = vec2<f32>(positions[positionsOffset + 2u * row], positions[positionsOffset + 2u * row + 1u]);
    minimum = min(minimum, point); maximum = max(maximum, point);
  }
  bounds[boundsOffset] = minimum.x; bounds[boundsOffset + 1u] = minimum.y;
  bounds[boundsOffset + 2u] = maximum.x; bounds[boundsOffset + 3u] = maximum.y;
  if (POINT_COUNT >= 3u && invalidCount[invalidCountOffset] == 0u) {
    triangles[trianglesOffset] = POINT_COUNT;
    triangles[trianglesOffset + 1u] = POINT_COUNT + 1u;
    triangles[trianglesOffset + 2u] = POINT_COUNT + 2u;
    triangles[trianglesOffset + 3u] = 0u;
    state[stateOffset] = 1u;
  }`
      })
    );

    let sourceTriangles = trianglesA;
    let destinationTriangles = trianglesB;
    for (let pointRow = 0; pointRow < pointCount; pointRow++) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-site-${pointRow}-cavity`,
          operation: OPERATION,
          variant: 'parallel-cavity-classification',
          bindings: [
            {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
            {name: 'bounds', view: bounds, type: 'f32', access: 'read'},
            {name: 'triangles', view: sourceTriangles, type: 'u32', access: 'read'},
            {name: 'state', view: state, type: 'u32', access: 'read'},
            {name: 'duplicateFlags', view: duplicateFlags, type: 'u32', access: 'read'},
            {name: 'badFlags', view: badFlags, type: 'u32', access: 'read_write'}
          ],
          invocationCount: scratchTriangleCapacity,
          declarations: `${sharedDeclarations}
const INSERTION_ROW: u32 = ${pointRow}u;`,
          body: `var bad = false;
  if (index < state[stateOffset] && state[stateOffset + 1u] == 0u && duplicateFlags[duplicateFlagsOffset + INSERTION_ROW] == 0u) {
    let point = pointAt(INSERTION_ROW);
    let a = pointAt(triangleVertex(index, 0u));
    let b = pointAt(triangleVertex(index, 1u));
    let c = pointAt(triangleVertex(index, 2u));
    bad = inCircumcircle(a, b, c, point);
  }
  badFlags[badFlagsOffset + index] = select(0u, 1u, bad);`
        }),
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-site-${pointRow}-boundary`,
          operation: OPERATION,
          variant: 'parallel-cavity-boundary',
          bindings: [
            {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
            {name: 'bounds', view: bounds, type: 'f32', access: 'read'},
            {name: 'triangles', view: sourceTriangles, type: 'u32', access: 'read'},
            {name: 'state', view: state, type: 'u32', access: 'read'},
            {name: 'badFlags', view: badFlags, type: 'u32', access: 'read'},
            {name: 'boundaryFlags', view: boundaryFlags, type: 'u32', access: 'read_write'}
          ],
          invocationCount: 3 * scratchTriangleCapacity,
          declarations: `${sharedDeclarations}
const INSERTION_ROW: u32 = ${pointRow}u;`,
          body: `let triangle = index / 3u;
  let edge = index % 3u;
  var boundary = triangle < state[stateOffset] && badFlags[badFlagsOffset + triangle] != 0u;
  if (boundary) {
    let a = triangleVertex(triangle, edge);
    let b = triangleVertex(triangle, (edge + 1u) % 3u);
    for (var other = 0u; other < state[stateOffset] && boundary; other++) {
      if (other == triangle || badFlags[badFlagsOffset + other] == 0u) { continue; }
      for (var otherEdge = 0u; otherEdge < 3u; otherEdge++) {
        let otherA = triangleVertex(other, otherEdge);
        let otherB = triangleVertex(other, (otherEdge + 1u) % 3u);
        if ((a == otherA && b == otherB) || (a == otherB && b == otherA)) { boundary = false; }
      }
    }
    boundary = boundary && orient(pointAt(a), pointAt(b), pointAt(INSERTION_ROW)) != 0.0;
  }
  boundaryFlags[boundaryFlagsOffset + index] = select(0u, 1u, boundary);`
        }),
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-site-${pointRow}-rebuild`,
          operation: OPERATION,
          variant: 'stable-topology-rebuild',
          bindings: [
            {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
            {name: 'bounds', view: bounds, type: 'f32', access: 'read'},
            {name: 'triangles', view: sourceTriangles, type: 'u32', access: 'read'},
            {name: 'trianglesOut', view: destinationTriangles, type: 'u32', access: 'read_write'},
            {name: 'state', view: state, type: 'u32', access: 'read_write'},
            {name: 'badFlags', view: badFlags, type: 'u32', access: 'read'},
            {name: 'boundaryFlags', view: boundaryFlags, type: 'u32', access: 'read'},
            {name: 'duplicateFlags', view: duplicateFlags, type: 'u32', access: 'read'}
          ],
          invocationCount: 1,
          declarations: `${sharedDeclarations}
const INSERTION_ROW: u32 = ${pointRow}u;`,
          body: `if (state[stateOffset + 1u] != 0u) { return; }
  let oldCount = state[stateOffset];
  var nextCount = 0u;
  for (var triangle = 0u; triangle < oldCount; triangle++) {
    if (badFlags[badFlagsOffset + triangle] == 0u) {
      let sourceBase = trianglesOffset + 4u * triangle;
      let destinationBase = trianglesOutOffset + 4u * nextCount;
      trianglesOut[destinationBase] = triangles[sourceBase];
      trianglesOut[destinationBase + 1u] = triangles[sourceBase + 1u];
      trianglesOut[destinationBase + 2u] = triangles[sourceBase + 2u];
      trianglesOut[destinationBase + 3u] = 0u;
      nextCount++;
    }
  }
  if (duplicateFlags[duplicateFlagsOffset + INSERTION_ROW] == 0u) {
    for (var candidate = 0u; candidate < 3u * oldCount; candidate++) {
      if (boundaryFlags[boundaryFlagsOffset + candidate] == 0u) { continue; }
      if (nextCount >= SCRATCH_TRIANGLE_CAPACITY) { state[stateOffset + 1u] = 1u; return; }
      let triangle = candidate / 3u;
      let edge = candidate % 3u;
      var a = triangleVertex(triangle, edge);
      var b = triangleVertex(triangle, (edge + 1u) % 3u);
      if (orient(pointAt(a), pointAt(b), pointAt(INSERTION_ROW)) < 0.0) {
        let swap = a; a = b; b = swap;
      }
      let destinationBase = trianglesOutOffset + 4u * nextCount;
      trianglesOut[destinationBase] = a;
      trianglesOut[destinationBase + 1u] = b;
      trianglesOut[destinationBase + 2u] = INSERTION_ROW;
      trianglesOut[destinationBase + 3u] = 0u;
      nextCount++;
    }
  }
  state[stateOffset] = nextCount;`
        })
      );
      [sourceTriangles, destinationTriangles] = [destinationTriangles, sourceTriangles];
    }

    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-publish-internal-failure`,
        operation: OPERATION,
        variant: 'publish-internal-failure',
        bindings: [
          {name: 'state', view: state, type: 'u32', access: 'read'},
          {name: 'invalidCount', view: invalidCount, type: 'u32', access: 'read_write'}
        ],
        invocationCount: 1,
        body: 'invalidCount[invalidCountOffset] += state[stateOffset + 1u];'
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-classify-output`,
        operation: OPERATION,
        variant: 'parallel-output-classification',
        bindings: [
          {name: 'triangles', view: sourceTriangles, type: 'u32', access: 'read'},
          {name: 'state', view: state, type: 'u32', access: 'read'},
          {name: 'triangleIds', view: triangleIds, type: 'u32', access: 'read_write'},
          {name: 'resultFlags', view: resultFlags, type: 'u32', access: 'read_write'}
        ],
        invocationCount: scratchTriangleCapacity,
        declarations: `const POINT_COUNT: u32 = ${pointCount}u;`,
        body: `triangleIds[triangleIdsOffset + index] = index;
  var accepted = index < state[stateOffset] && state[stateOffset + 1u] == 0u;
  if (accepted) {
    let base = trianglesOffset + 4u * index;
    accepted = triangles[base] < POINT_COUNT && triangles[base + 1u] < POINT_COUNT && triangles[base + 2u] < POINT_COUNT;
  }
  resultFlags[resultFlagsOffset + index] = select(0u, 1u, accepted);`
      }),
      ...new GPUCompaction({
        id: `${id}-compact-output`,
        input: triangleIds,
        flags: resultFlags,
        output: compactTriangleIds,
        count: status.requiredCount
      }).getCommandNodes(graph)
    );
    if (output.triangles.length > 0) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-materialize`,
          operation: OPERATION,
          variant: 'parallel-materialization',
          bindings: [
            {name: 'triangles', view: sourceTriangles, type: 'u32', access: 'read'},
            {name: 'compactTriangleIds', view: compactTriangleIds, type: 'u32', access: 'read'},
            {name: 'requiredCount', view: status.requiredCount, type: 'u32', access: 'read'},
            {name: 'trianglesOut', view: output.triangles, type: 'u32', access: 'read_write'}
          ],
          invocationCount: output.triangles.length,
          body: `if (index >= requiredCount[requiredCountOffset]) { return; }
  let triangle = compactTriangleIds[compactTriangleIdsOffset + index];
  let sourceBase = trianglesOffset + 4u * triangle;
  let destinationBase = trianglesOutOffset + 3u * index;
  trianglesOut[destinationBase] = triangles[sourceBase];
  trianglesOut[destinationBase + 1u] = triangles[sourceBase + 1u];
  trianglesOut[destinationBase + 2u] = triangles[sourceBase + 2u];`
        })
      );
    }
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-publish-status`,
        operation: OPERATION,
        variant: 'publish-status',
        bindings: [
          {name: 'requiredCountIn', view: status.requiredCount, type: 'u32', access: 'read'},
          {name: 'countOut', view: status.count, type: 'u32', access: 'read_write'},
          {name: 'overflowOut', view: status.overflow, type: 'u32', access: 'read_write'}
        ],
        invocationCount: 1,
        declarations: `const OUTPUT_CAPACITY: u32 = ${output.triangles.length}u;`,
        body: `let requiredCount = requiredCountIn[requiredCountInOffset];
  countOut[countOutOffset] = min(requiredCount, OUTPUT_CAPACITY);
  overflowOut[overflowOutOffset] = select(0u, 1u, requiredCount > OUTPUT_CAPACITY);`
      })
    );
    return nodes;
  }
}

function getDelaunayCoreWGSL(options: {
  pointCount: number;
  scratchTriangleCapacity: number;
}): string {
  return /* wgsl */ `
const POINT_COUNT: u32 = ${options.pointCount}u;
const SCRATCH_TRIANGLE_CAPACITY: u32 = ${options.scratchTriangleCapacity}u;
fn pointAt(row: u32) -> vec2<f32> {
  if (row < POINT_COUNT) {
    return vec2<f32>(positions[positionsOffset + 2u * row], positions[positionsOffset + 2u * row + 1u]);
  }
  let minimum = vec2<f32>(bounds[boundsOffset], bounds[boundsOffset + 1u]);
  let maximum = vec2<f32>(bounds[boundsOffset + 2u], bounds[boundsOffset + 3u]);
  let center = 0.5 * (minimum + maximum);
  let scale = max(max(maximum.x - minimum.x, maximum.y - minimum.y), 1.0);
  if (row == POINT_COUNT) { return center + vec2<f32>(-32.0 * scale, -2.0 * scale); }
  if (row == POINT_COUNT + 1u) { return center + vec2<f32>(32.0 * scale, -2.0 * scale); }
  return center + vec2<f32>(0.0, 32.0 * scale);
}
fn orient(a: vec2<f32>, b: vec2<f32>, c: vec2<f32>) -> f32 {
  return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
}
fn inCircumcircle(a: vec2<f32>, b: vec2<f32>, c: vec2<f32>, point: vec2<f32>) -> bool {
  let av = a - point;
  let bv = b - point;
  let cv = c - point;
  let determinant = dot(av, av) * (bv.x * cv.y - cv.x * bv.y)
    - dot(bv, bv) * (av.x * cv.y - cv.x * av.y)
    + dot(cv, cv) * (av.x * bv.y - bv.x * av.y);
  let scale = max(max(max(dot(av, av), dot(bv, bv)), dot(cv, cv)), 1.0);
  return determinant >= -1e-6 * scale * scale;
}
fn triangleVertex(triangle: u32, vertex: u32) -> u32 {
  return triangles[trianglesOffset + 4u * triangle + vertex];
}`;
}
