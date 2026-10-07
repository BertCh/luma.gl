// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GraphVectorView,
  validatePackedUint32View,
  validatePackedView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {createWGSLKernelNode, type WGSLKernelBinding} from '../../utils/wgsl-kernel-nodes';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../utils/gpu-contributor-utils';
import {GPUGroupConvexHull} from '../group-geometry/index';

const OPERATION = 'GPUMinimumBounds';
const MAXIMUM_ROWS = 2 ** 24 - 1;
const DEFAULT_MAXIMUM_HULL_VERTICES = 256;
/** Hulls up to this size scan every pair (exact original behavior); larger ones use calipers. */
const CALIPER_MINIMUM_HULL_VERTICES = 16;

/**
 * Caller-owned outputs of {@link GPUMinimumBounds}. Every output is optional and rewritten on every
 * encoding. Rows of a group that is empty, or whose hull exceeded `maximumHullVertices`, hold NaN.
 */
export type GPUMinimumBoundsOutput = {
  /**
   * Corners of the minimum-area rotated rectangle, 4 rows per group, counter-clockwise. The first
   * two corners lie on the side parallel to the hull edge that defined the rectangle.
   */
  rectangleCorners?: GraphDataView<'float32x2'>;
  /**
   * `[width, height, angle, area]` per group of the rotated rectangle. `width` is the extent along
   * the defining hull edge, `height` the extent across it, `angle` the direction of that edge in
   * radians in `(-pi, pi]`.
   */
  rectangleSizes?: GraphDataView<'float32x4'>;
  /** Minimum bounding circle `[centerX, centerY, radius]` per group. */
  circles?: GraphDataView<'float32x3'>;
  /** Longest line (maximum distance pair) `[x0, y0, x1, y1]` per group. */
  longestLines?: GraphDataView<'float32x4'>;
  /** Length of {@link GPUMinimumBoundsOutput.longestLines} per group (the diameter). */
  diameters?: GraphDataView<'float32'>;
  /** Axis-aligned envelope `[minX, minY, maxX, maxY]` per group. */
  bounds?: GraphDataView<'float32x4'>;
  /** True convex hull vertex count per group, before the `maximumHullVertices` cap. */
  hullSizes?: GraphDataView<'uint32'>;
  /**
   * One row, a bit set; bit 1 (`GPU_GROUP_CONVEX_HULL_GROUP_OVERFLOW`) means some hull exceeded
   * `maximumHullVertices` and its group's rows are NaN. `0` when every group was computed.
   */
  overflow: GraphDataView<'uint32'>;
};

/**
 * Properties for {@link GPUMinimumBounds}.
 *
 * Give either geometry (`ringOffsets`, optional `featureRingOffsets`, one group per feature) or
 * explicit `labels` with `groupCount`, not both. All input contents are per-frame; view lengths,
 * `groupCount`, `maximumHullVertices` and the set of requested outputs need a new graph.
 */
export type GPUMinimumBoundsProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'minimum-bounds'`. */
  id?: string;
  /** Planar points or vertices (fewer than `2^24` rows). Non-finite rows are excluded. */
  positions: GraphDataView<'float32x2'>;
  /**
   * GeoArrow-style ring (or line, or point-part) offsets: ring `r` owns
   * `positions[ringOffsets[r] .. ringOffsets[r + 1])`. Vertices outside every ring are excluded.
   */
  ringOffsets?: GraphDataView<'uint32'>;
  /**
   * Optional feature to ring offsets (multi-geometries, holes). Feature `f` owns rings
   * `[featureRingOffsets[f], featureRingOffsets[f + 1])`. Without it every ring is a feature.
   */
  featureRingOffsets?: GraphDataView<'uint32'>;
  /** Alternative to geometry: a group label per position row. Labels `>= groupCount` are excluded. */
  labels?: GraphDataView<'uint32'>;
  /** Number of groups when `labels` is given. */
  groupCount?: number;
  /**
   * Largest convex hull handled per group (at least 3, default 256). Larger hulls are flagged in
   * `overflow` and produce NaN. Hulls above 16 vertices use rotating calipers for the rectangle and
   * longest line (linear per group); smaller hulls scan all pairs.
   */
  maximumHullVertices?: number;
  /** Caller-owned outputs. */
  output: GPUMinimumBoundsOutput;
};

/**
 * Minimum bounding shapes of every feature (or label group): the minimum-area rotated rectangle
 * (`oriented_envelope`, `minimum_rotated_rectangle`), the minimum bounding circle
 * (`minimum_bounding_circle`, `minimum_bounding_radius`), the longest line / diameter (Sedona
 * `ST_LongestLine`, `ST_MaxDistance`) and the axis-aligned envelope.
 *
 * ## Semantics
 *
 * - The exact lattice convex hull of every group comes from {@link GPUGroupConvexHull} (vertex to
 *   feature labels are derived from the offsets), so every shape depends only on hull vertices.
 *   Rectangle, circle, line and envelope are computed by one thread per group over its hull in
 *   group-relative coordinates, so large coordinate offsets do not cost float32 precision.
 * - Rectangle: for every hull edge the bounding rectangle aligned with it is measured (rotating
 *   calipers: the three extreme vertices only move forward as the edge turns, so a hull costs O(h)
 *   instead of O(h^2); hulls of at most 16 vertices scan every vertex), and the
 *   smallest area wins. An area within `1e-6` relative of the best so far does not replace it, so
 *   equal areas (squares) resolve to the lowest edge index, counting hull edges counter-clockwise
 *   from the smallest `(x, y)` vertex. Rectangles agree with GEOS in area, not in corner order.
 *   A single point gives four equal corners; collinear points a zero-height rectangle.
 * - Circle: Welzl's algorithm on the hull vertices in a fixed hash-shuffled order (deterministic,
 *   expected linear), then the radius is recomputed as the largest hull distance so the circle
 *   always covers the hull. Degenerate groups give radius 0.
 * - Longest line: the pair of hull vertices with the largest distance; ties go to the lowest
 *   vertex pair in hull order. Large hulls test only the antipodal candidates found by a rotating
 *   caliper (a farthest pair is always antipodal), O(h) instead of O(h^2).
 * - Envelope: from the hull vertices, so it equals the point extent.
 *
 * Output is fixed size per group. Hulls over `maximumHullVertices` are flagged in `overflow`.
 *
 * The contributor never compiles, encodes, submits or reads back.
 */
export class GPUMinimumBounds implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUMinimumBoundsProps;
  /** Number of groups (features, or `groupCount`). */
  readonly groupCount: number;
  /** Resolved hull vertex cap. */
  readonly maximumHullVertices: number;

  constructor(props: GPUMinimumBoundsProps) {
    this.id = props.id ?? 'minimum-bounds';
    this.props = props;
    const {id} = this;
    const {output} = props;
    for (const [name, view] of [
      ['positions', props.positions],
      ['ringOffsets', props.ringOffsets],
      ['featureRingOffsets', props.featureRingOffsets],
      ['labels', props.labels]
    ] as const) {
      if (view && (view as unknown) instanceof GraphVectorView) {
        throw new Error(`${id} ${name} must be a single packed view, not a chunked vector`);
      }
    }
    validatePackedView(props.positions, ['float32x2'], `${id} positions`);
    const rows = props.positions.length;
    if (rows < 1) {
      throw new Error(`${id} needs at least one row`);
    }
    if (rows > MAXIMUM_ROWS) {
      throw new Error(`${id} supports fewer than 2^24 rows`);
    }
    if (props.labels && props.ringOffsets) {
      throw new Error(`${id} takes either labels or ringOffsets, not both`);
    }
    if (props.labels) {
      validatePackedUint32View(props.labels, `${id} labels`);
      if (props.labels.length !== rows) {
        throw new Error(`${id} labels length must equal positions length`);
      }
      if (props.featureRingOffsets) {
        throw new Error(`${id} featureRingOffsets needs ringOffsets`);
      }
      const groups = props.groupCount;
      if (
        groups === undefined ||
        !Number.isInteger(groups) ||
        groups < 1 ||
        groups > MAXIMUM_ROWS
      ) {
        throw new Error(`${id} groupCount must be a positive integer below 2^24 with labels`);
      }
      this.groupCount = groups;
    } else if (props.ringOffsets) {
      validatePackedUint32View(props.ringOffsets, `${id} ringOffsets`);
      if (props.ringOffsets.length < 2) {
        throw new Error(`${id} ringOffsets must contain at least two rows`);
      }
      if (props.featureRingOffsets) {
        validatePackedUint32View(props.featureRingOffsets, `${id} featureRingOffsets`);
        if (props.featureRingOffsets.length < 2) {
          throw new Error(`${id} featureRingOffsets must contain at least two rows`);
        }
      }
      this.groupCount = props.featureRingOffsets
        ? props.featureRingOffsets.length - 1
        : props.ringOffsets.length - 1;
    } else {
      throw new Error(`${id} needs ringOffsets or labels`);
    }
    this.maximumHullVertices = props.maximumHullVertices ?? DEFAULT_MAXIMUM_HULL_VERTICES;
    if (!Number.isInteger(this.maximumHullVertices) || this.maximumHullVertices < 3) {
      throw new Error(`${id} maximumHullVertices must be an integer of at least 3`);
    }
    const groups = this.groupCount;
    const checks = [
      ['rectangleCorners', output.rectangleCorners, ['float32x2'], groups * 4],
      ['rectangleSizes', output.rectangleSizes, ['float32x4'], groups],
      ['circles', output.circles, ['float32x3'], groups],
      ['longestLines', output.longestLines, ['float32x4'], groups],
      ['diameters', output.diameters, ['float32'], groups],
      ['bounds', output.bounds, ['float32x4'], groups],
      ['hullSizes', output.hullSizes, ['uint32'], groups],
      ['overflow', output.overflow, ['uint32'], 1]
    ] as const;
    for (const [name, view, formats, length] of checks) {
      if (!view) {
        continue;
      }
      validatePackedView(view, formats, `${id} output.${name}`);
      if (view.length < length) {
        throw new Error(`${id} output.${name} must hold at least ${length} rows`);
      }
    }
    if (
      !output.rectangleCorners &&
      !output.rectangleSizes &&
      !output.circles &&
      !output.longestLines &&
      !output.diameters &&
      !output.bounds
    ) {
      throw new Error(`${id} needs at least one shape output`);
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      checks.map(([, view]) => view),
      [props.positions, props.ringOffsets, props.featureRingOffsets, props.labels]
    );
  }

  /** Returns label, hull and one node per requested shape in order. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id, groupCount} = this;
    const {output, positions} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      positions,
      props.ringOffsets,
      props.featureRingOffsets,
      props.labels,
      ...Object.values(output)
    ]);
    const rows = positions.length;
    const u32 = (name: string, length: number) =>
      createTransientView(graph, `${id}-${name}`, 'uint32', length);
    const read = (name: string, view: GraphDataView, type: 'u32' | 'f32'): WGSLKernelBinding => ({
      name,
      view,
      type,
      access: 'read'
    });
    const write = (name: string, view: GraphDataView, type: 'u32' | 'f32' = 'f32') =>
      ({name, view, type, access: 'read_write'}) as WGSLKernelBinding;
    const kernel = (
      step: string,
      invocationCount: number,
      bindings: WGSLKernelBinding[],
      body: string,
      declarations = ''
    ) =>
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-${step}`,
        operation: OPERATION,
        variant: step,
        bindings,
        invocationCount,
        declarations,
        body
      });
    const nodes: GPUCommandNode<Parameters>[] = [];

    // 1. Vertex to group labels.
    let labels = props.labels;
    if (!labels) {
      const ringOffsets = props.ringOffsets!;
      const featureRingOffsets = props.featureRingOffsets;
      const ringCount = ringOffsets.length - 1;
      const featureCount = featureRingOffsets ? featureRingOffsets.length - 1 : ringCount;
      labels = createTransientView(graph, `${id}-labels`, 'uint32', rows);
      const bindings = [read('ringOffsets', ringOffsets, 'u32')];
      if (featureRingOffsets) {
        bindings.push(read('featureRingOffsets', featureRingOffsets, 'u32'));
      }
      bindings.push(write('labels', labels, 'u32'));
      nodes.push(
        kernel(
          'labels',
          rows,
          bindings,
          `var label = 0xffffffffu;
  if (index >= ringOffsets[ringOffsetsOffset] && index < ringOffsets[ringOffsetsOffset + ${ringCount}u]) {
    var low = 0u;
    var high = ${ringCount - 1}u;
    while (low < high) {
      let middle = (low + high + 1u) >> 1u;
      if (ringOffsets[ringOffsetsOffset + middle] <= index) {
        low = middle;
      } else {
        high = middle - 1u;
      }
    }
    label = low;
    ${
      featureRingOffsets
        ? `if (low >= featureRingOffsets[featureRingOffsetsOffset] && low < featureRingOffsets[featureRingOffsetsOffset + ${featureCount}u]) {
      var featureLow = 0u;
      var featureHigh = ${featureCount - 1}u;
      while (featureLow < featureHigh) {
        let middle = (featureLow + featureHigh + 1u) >> 1u;
        if (featureRingOffsets[featureRingOffsetsOffset + middle] <= low) {
          featureLow = middle;
        } else {
          featureHigh = middle - 1u;
        }
      }
      label = featureLow;
    } else {
      label = 0xffffffffu;
    }`
        : ''
    }
  }
  labels[labelsOffset + index] = label;`
        )
      );
    }

    // 2. Per-group convex hulls.
    const hullIndices = u32('hull-indices', rows);
    const hullPositions = createTransientView(graph, `${id}-hull-positions`, 'float32x2', rows);
    const hullOffsets = u32('hull-offsets', groupCount + 1);
    const hullCounts = u32('hull-counts', groupCount);
    nodes.push(
      ...new GPUGroupConvexHull({
        id: `${id}-hull`,
        positions,
        labels,
        groupCount,
        maximumVerticesPerGroup: this.maximumHullVertices,
        totalCapacity: rows,
        output: {
          vertexIndices: hullIndices,
          vertexPositions: hullPositions,
          offsets: hullOffsets,
          counts: hullCounts,
          sizes: output.hullSizes,
          overflow: output.overflow
        }
      }).getCommandNodes(graph)
    );

    // 3. Shapes, one thread per group over its hull in anchor-relative coordinates.
    const hullBindings = () => [
      read('offsets', hullOffsets, 'u32'),
      read('counts', hullCounts, 'u32'),
      read('hull', hullPositions, 'f32')
    ];
    const prelude = `let count = counts[countsOffset + index];
  let begin = offsets[offsetsOffset + index];
  let nan = bitcast<f32>(0x7fc00000u | (index & 0u));
  var anchor = vec2<f32>(0.0, 0.0);
  if (count > 0u) {
    anchor = vec2<f32>(hull[hullOffset + begin * 2u], hull[hullOffset + begin * 2u + 1u]);
  }`;
    const declarations = `const CALIPER_MINIMUM: u32 = ${CALIPER_MINIMUM_HULL_VERTICES + 1}u;
fn loadPoint(slot: u32, anchor: vec2<f32>) -> vec2<f32> {
  return vec2<f32>(hull[hullOffset + slot * 2u], hull[hullOffset + slot * 2u + 1u]) - anchor;
}
// Rotating-caliper step: the hull is strictly convex, so a linear function along it rises to one
// maximum and falls again. Starting on the rising side, walk forward while one of the next two
// vertices is higher (two, to ride over float noise at the plateau of a perpendicular edge).
fn advanceMaximum(start: u32, begin: u32, count: u32, anchor: vec2<f32>, axis: vec2<f32>) -> u32 {
  var slot = start;
  var value = dot(loadPoint(begin + slot, anchor), axis);
  for (var step = 0u; step < count; step++) {
    var moved = false;
    for (var lookahead = 1u; lookahead <= 2u; lookahead++) {
      let candidate = (slot + lookahead) % count;
      let candidateValue = dot(loadPoint(begin + candidate, anchor), axis);
      if (candidateValue > value) {
        slot = candidate;
        value = candidateValue;
        moved = true;
        break;
      }
    }
    if (!moved) { break; }
  }
  return slot;
}
fn findMaximum(begin: u32, count: u32, anchor: vec2<f32>, axis: vec2<f32>) -> u32 {
  var best = 0u;
  var bestValue = dot(loadPoint(begin, anchor), axis);
  for (var slot = 1u; slot < count; slot++) {
    let value = dot(loadPoint(begin + slot, anchor), axis);
    if (value > bestValue) {
      best = slot;
      bestValue = value;
    }
  }
  return best;
}`;
    if (output.rectangleCorners || output.rectangleSizes) {
      const bindings = hullBindings();
      if (output.rectangleCorners) {
        bindings.push(write('corners', output.rectangleCorners));
      }
      if (output.rectangleSizes) {
        bindings.push(write('sizes', output.rectangleSizes));
      }
      nodes.push(
        kernel(
          'rectangle',
          groupCount,
          bindings,
          `${prelude}
  var bestArea = 0.0;
  var bestEdge = 0xffffffffu;
  var bestDirection = vec2<f32>(1.0, 0.0);
  var bestRange = vec4<f32>(0.0);
  var rightSlot = 0u;
  var topSlot = 0u;
  var leftSlot = 0u;
  for (var edge = 0u; edge < count; edge++) {
    let from0 = loadPoint(begin + edge, anchor);
    let to0 = loadPoint(begin + (edge + 1u) % count, anchor);
    let delta = to0 - from0;
    let edgeLength = sqrt(dot(delta, delta));
    var direction = vec2<f32>(1.0, 0.0);
    if (edgeLength > 0.0) {
      direction = delta / edgeLength;
    }
    let normal = vec2<f32>(-direction.y, direction.x);
    var range = vec4<f32>(3.0e38, 3.0e38, -3.0e38, -3.0e38);
    if (count < CALIPER_MINIMUM) {
      for (var vertex = 0u; vertex < count; vertex++) {
        let point = loadPoint(begin + vertex, anchor);
        let along = dot(point, direction);
        let across = dot(point, normal);
        range = vec4<f32>(min(range.x, along), min(range.y, across), max(range.z, along), max(range.w, across));
      }
    } else {
      // Calipers: the edge's own side is the minimum across; the other three extremes turn
      // counter-clockwise with the edge, so each pointer only moves forward.
      if (edge == 0u) {
        rightSlot = findMaximum(begin, count, anchor, direction);
        topSlot = findMaximum(begin, count, anchor, normal);
        leftSlot = findMaximum(begin, count, anchor, -direction);
      } else {
        rightSlot = advanceMaximum(rightSlot, begin, count, anchor, direction);
        topSlot = advanceMaximum(topSlot, begin, count, anchor, normal);
        leftSlot = advanceMaximum(leftSlot, begin, count, anchor, -direction);
      }
      range = vec4<f32>(
        dot(loadPoint(begin + leftSlot, anchor), direction),
        min(dot(from0, normal), dot(to0, normal)),
        dot(loadPoint(begin + rightSlot, anchor), direction),
        dot(loadPoint(begin + topSlot, anchor), normal)
      );
    }
    let area = (range.z - range.x) * (range.w - range.y);
    if (bestEdge == 0xffffffffu || area < bestArea * (1.0 - 1.0e-6)) {
      bestArea = area;
      bestEdge = edge;
      bestDirection = direction;
      bestRange = range;
    }
  }
  let normal = vec2<f32>(-bestDirection.y, bestDirection.x);
  let width = bestRange.z - bestRange.x;
  let height = bestRange.w - bestRange.y;
  var angle = 0.0;
  if (bestDirection.x == 0.0) {
    angle = select(-1.5707964, 1.5707964, bestDirection.y > 0.0);
  } else {
    angle = atan2(bestDirection.y, bestDirection.x);
  }
  ${
    output.rectangleCorners
      ? `for (var corner = 0u; corner < 4u; corner++) {
    let alongValue = select(bestRange.x, bestRange.z, corner == 1u || corner == 2u);
    let acrossValue = select(bestRange.y, bestRange.w, corner >= 2u);
    let point = anchor + bestDirection * alongValue + normal * acrossValue;
    corners[cornersOffset + (index * 4u + corner) * 2u] = select(nan, point.x, count > 0u);
    corners[cornersOffset + (index * 4u + corner) * 2u + 1u] = select(nan, point.y, count > 0u);
  }`
      : ''
  }
  ${
    output.rectangleSizes
      ? `let isValid = count > 0u;
  sizes[sizesOffset + index * 4u] = select(nan, width, isValid);
  sizes[sizesOffset + index * 4u + 1u] = select(nan, height, isValid);
  sizes[sizesOffset + index * 4u + 2u] = select(nan, angle, isValid);
  sizes[sizesOffset + index * 4u + 3u] = select(nan, width * height, isValid);`
      : ''
  }`,
          declarations
        )
      );
    }

    if (output.circles) {
      const order = u32('circle-order', rows);
      nodes.push(
        kernel(
          'circle',
          groupCount,
          [...hullBindings(), write('order', order, 'u32'), write('circles', output.circles)],
          `${prelude}
  if (count == 0u) {
    circles[circlesOffset + index * 3u] = nan;
    circles[circlesOffset + index * 3u + 1u] = nan;
    circles[circlesOffset + index * 3u + 2u] = nan;
    return;
  }
  for (var slot = 0u; slot < count; slot++) {
    order[orderOffset + begin + slot] = slot;
  }
  // Deterministic Fisher-Yates shuffle keyed by group and position.
  for (var slot = count - 1u; slot > 0u; slot--) {
    let pick = hashPair(index, slot) % (slot + 1u);
    let held = order[orderOffset + begin + slot];
    order[orderOffset + begin + slot] = order[orderOffset + begin + pick];
    order[orderOffset + begin + pick] = held;
  }
  var center = loadPoint(begin + order[orderOffset + begin], anchor);
  var radiusSquared = 0.0;
  for (var i = 1u; i < count; i++) {
    let pointI = loadPoint(begin + order[orderOffset + begin + i], anchor);
    if (isOutside(pointI, center, radiusSquared)) {
      center = pointI;
      radiusSquared = 0.0;
      for (var j = 0u; j < i; j++) {
        let pointJ = loadPoint(begin + order[orderOffset + begin + j], anchor);
        if (isOutside(pointJ, center, radiusSquared)) {
          center = 0.5 * (pointI + pointJ);
          radiusSquared = dot(pointI - center, pointI - center);
          for (var k = 0u; k < j; k++) {
            let pointK = loadPoint(begin + order[orderOffset + begin + k], anchor);
            if (isOutside(pointK, center, radiusSquared)) {
              center = getCircumcenter(pointI, pointJ, pointK);
              radiusSquared = max(
                dot(pointI - center, pointI - center),
                max(dot(pointJ - center, pointJ - center), dot(pointK - center, pointK - center))
              );
            }
          }
        }
      }
    }
  }
  // Recompute the radius from the hull so the circle covers every vertex.
  var coverSquared = 0.0;
  for (var slot = 0u; slot < count; slot++) {
    let delta = loadPoint(begin + slot, anchor) - center;
    coverSquared = max(coverSquared, dot(delta, delta));
  }
  circles[circlesOffset + index * 3u] = anchor.x + center.x;
  circles[circlesOffset + index * 3u + 1u] = anchor.y + center.y;
  circles[circlesOffset + index * 3u + 2u] = sqrt(coverSquared);`,
          `${declarations}
fn hashPair(a: u32, b: u32) -> u32 {
  var state = a * 747796405u + b * 2891336453u + 2891336453u;
  let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}
fn isOutside(point: vec2<f32>, center: vec2<f32>, radiusSquared: f32) -> bool {
  let delta = point - center;
  return dot(delta, delta) > radiusSquared * (1.0 + 4.0e-6);
}
fn getDiametralCenter(a: vec2<f32>, b: vec2<f32>, c: vec2<f32>) -> vec2<f32> {
  let ab = dot(a - b, a - b);
  let ac = dot(a - c, a - c);
  let bc = dot(b - c, b - c);
  if (ab >= ac && ab >= bc) {
    return 0.5 * (a + b);
  }
  if (ac >= bc) {
    return 0.5 * (a + c);
  }
  return 0.5 * (b + c);
}
fn getCircumcenter(a: vec2<f32>, b: vec2<f32>, c: vec2<f32>) -> vec2<f32> {
  let ab = b - a;
  let ac = c - a;
  let determinant = 2.0 * (ab.x * ac.y - ab.y * ac.x);
  let lengths = sqrt(dot(ab, ab) * dot(ac, ac));
  if (abs(determinant) <= 2.0e-7 * lengths) {
    return getDiametralCenter(a, b, c);
  }
  let abSquared = dot(ab, ab);
  let acSquared = dot(ac, ac);
  return a + vec2<f32>(
    (ac.y * abSquared - ab.y * acSquared) / determinant,
    (ab.x * acSquared - ac.x * abSquared) / determinant
  );
}`
        )
      );
    }

    if (output.longestLines || output.diameters) {
      const bindings = hullBindings();
      if (output.longestLines) {
        bindings.push(write('lines', output.longestLines));
      }
      if (output.diameters) {
        bindings.push(write('diameters', output.diameters));
      }
      nodes.push(
        kernel(
          'longest-line',
          groupCount,
          bindings,
          `${prelude}
  bestSquared = -1.0;
  bestFrom = 0u;
  bestTo = 0u;
  if (count < CALIPER_MINIMUM) {
    for (var i = 0u; i < count; i++) {
      for (var j = i; j < count; j++) {
        considerPair(begin, anchor, i, j);
      }
    }
  } else {
    // Rotating calipers: a farthest pair has parallel supporting lines, so it is found among the
    // vertices opposite each edge. Pairs are compared exactly as the full scan does.
    var opposite = 0u;
    for (var edge = 0u; edge < count; edge++) {
      let next = (edge + 1u) % count;
      let edgeStart = loadPoint(begin + edge, anchor);
      let edgeVector = loadPoint(begin + next, anchor) - edgeStart;
      let normal = vec2<f32>(-edgeVector.y, edgeVector.x);
      if (edge == 0u) {
        opposite = findMaximum(begin, count, anchor, normal);
      } else {
        opposite = advanceMaximum(opposite, begin, count, anchor, normal);
      }
      for (var offset = count - 1u; offset <= count + 1u; offset++) {
        let candidate = (opposite + offset) % count;
        considerPair(begin, anchor, edge, candidate);
        considerPair(begin, anchor, next, candidate);
      }
    }
  }
  let isValid = count > 0u;
  ${
    output.longestLines
      ? `let startPoint = loadPoint(begin + bestFrom, anchor) + anchor;
  let endPoint = loadPoint(begin + bestTo, anchor) + anchor;
  lines[linesOffset + index * 4u] = select(nan, startPoint.x, isValid);
  lines[linesOffset + index * 4u + 1u] = select(nan, startPoint.y, isValid);
  lines[linesOffset + index * 4u + 2u] = select(nan, endPoint.x, isValid);
  lines[linesOffset + index * 4u + 3u] = select(nan, endPoint.y, isValid);`
      : ''
  }
  ${
    output.diameters
      ? 'diameters[diametersOffset + index] = select(nan, sqrt(max(bestSquared, 0.0)), isValid);'
      : ''
  }`,
          `${declarations}
var<private> bestSquared: f32;
var<private> bestFrom: u32;
var<private> bestTo: u32;
// Keeps the farthest pair; equal distances go to the lowest (first, second) pair in hull order.
fn considerPair(begin: u32, anchor: vec2<f32>, one: u32, other: u32) {
  let first = min(one, other);
  let second = max(one, other);
  let delta = loadPoint(begin + second, anchor) - loadPoint(begin + first, anchor);
  let distanceSquared = dot(delta, delta);
  if (distanceSquared > bestSquared ||
      (distanceSquared == bestSquared && (first < bestFrom || (first == bestFrom && second < bestTo)))) {
    bestSquared = distanceSquared;
    bestFrom = first;
    bestTo = second;
  }
}`
        )
      );
    }

    if (output.bounds) {
      nodes.push(
        kernel(
          'bounds',
          groupCount,
          [...hullBindings(), write('bounds', output.bounds)],
          `${prelude}
  var low = vec2<f32>(3.0e38, 3.0e38);
  var high = vec2<f32>(-3.0e38, -3.0e38);
  for (var slot = 0u; slot < count; slot++) {
    let point = loadPoint(begin + slot, anchor);
    low = min(low, point);
    high = max(high, point);
  }
  let isValid = count > 0u;
  bounds[boundsOffset + index * 4u] = select(nan, anchor.x + low.x, isValid);
  bounds[boundsOffset + index * 4u + 1u] = select(nan, anchor.y + low.y, isValid);
  bounds[boundsOffset + index * 4u + 2u] = select(nan, anchor.x + high.x, isValid);
  bounds[boundsOffset + index * 4u + 3u] = select(nan, anchor.y + high.y, isValid);`,
          declarations
        )
      );
    }
    return nodes;
  }
}
