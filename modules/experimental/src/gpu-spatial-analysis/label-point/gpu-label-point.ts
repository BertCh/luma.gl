// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GraphVectorView,
  validatePackedUint32View,
  validatePackedView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../utils/gpu-contributor-utils';
import {createWGSLKernelNode, type WGSLKernelBinding} from '../../utils/wgsl-kernel-nodes';

const OPERATION = 'GPULabelPoint';

/** Default samples per axis of the first search grid. */
export const GPU_LABEL_POINT_DEFAULT_INITIAL_GRID_SIZE = 16;
/** Default samples per axis of each refinement grid. */
export const GPU_LABEL_POINT_DEFAULT_REFINEMENT_GRID_SIZE = 8;
/** Default number of refinement rounds. */
export const GPU_LABEL_POINT_DEFAULT_REFINEMENT_ROUNDS = 6;
/** Default vertex count above which a feature is searched by a whole workgroup. */
export const GPU_LABEL_POINT_DEFAULT_SMALL_FEATURE_VERTEX_LIMIT = 32;
/** Lanes of the per-feature workgroup used for large features. */
const WORKGROUP_SIZE = 64;

/** Caller-owned outputs of {@link GPULabelPoint}. */
export type GPULabelPointOutput = {
  /** One label point per feature (longitude/latitude or planar), NaN for features with no vertices. */
  points: GraphDataView<'float32x2'>;
  /**
   * Optional signed distance from each label point to the nearest boundary in position units:
   * positive inside, so it is the radius of the largest circle found around the label ("explain"
   * column for label size). 0 for degenerate features, NaN for empty ones.
   */
  distances?: GraphDataView<'float32'>;
  /**
   * Optional per-feature flag, `1` when no strictly interior point was found (zero-area or
   * degenerate polygons, or numerically collapsed slivers) and the point lies on the boundary,
   * else `0`.
   */
  degenerate?: GraphDataView<'uint32'>;
};

/**
 * Properties for {@link GPULabelPoint}.
 *
 * Per-frame (no recompile): the contents of every input buffer. Compile-time: view lengths, the
 * grid sizes, `refinementRounds`, and which outputs are present.
 */
export type GPULabelPointProps = {
  /** Prefix for generated node IDs. Defaults to `'label-point'`. */
  id?: string;
  /** Packed vertex positions in a planar coordinate system (project longitude/latitude first). */
  positions: GraphDataView<'float32x2'>;
  /** `ringCount + 1` monotonic vertex offsets, as in `GPUGeometryMeasures`. Rings close implicitly. */
  ringOffsets: GraphDataView<'uint32'>;
  /**
   * Optional `featureCount + 1` monotonic ring offsets. Rings of one feature combine by the
   * even-odd rule (holes and multi-polygon parts work in either winding). When omitted every ring
   * is its own feature.
   */
  featureRingOffsets?: GraphDataView<'uint32'>;
  /** Samples per axis of the first grid over the feature's bounds. Default 16. */
  initialGridSize?: number;
  /** Samples per axis of every refinement grid. Default 8. */
  refinementGridSize?: number;
  /** Refinement rounds. Default 6. */
  refinementRounds?: number;
  /**
   * Number of best first-grid cells that may be refined, from 1 to 16. Cells whose distance bound
   * cannot beat the best result so far are skipped, so typical shapes refine one or two. Default 4.
   */
  refinementCandidates?: number;
  /**
   * Features with at most this many vertices are searched by one thread each, which is cheapest
   * for many small features. Larger features get one 64-lane workgroup each, so their latency
   * stays proportional to `vertices / 64`. Default 32, the measured crossover.
   */
  smallFeatureVertexLimit?: number;
  /** Output columns. */
  output: GPULabelPointOutput;
};

/**
 * Pole of inaccessibility (the point of a polygon farthest from its boundary) for many polygons at
 * once, the label anchor of Mapbox `polylabel` and `turf pointOnFeature`-style labeling.
 *
 * Instead of polylabel's priority queue the search uses a fixed schedule that suits one GPU
 * invocation per feature: three horizontal scanlines seed guaranteed-interior candidates (the
 * midpoint of the leftmost interior interval), then an `initialGridSize^2` grid over the bounds
 * is scored by signed distance to every boundary segment, then `refinementRounds` rounds score a
 * `refinementGridSize^2` grid over the cell around each of the best `refinementCandidates` first-grid
 * cells whose distance bound `d + cellRadius` can still beat the best result. The window shrinks by
 * `2 / refinementGridSize` each round. Ties keep the first candidate, so the result is
 * deterministic.
 *
 * **Guarantees.** The seeds are interior in exact arithmetic for any non-degenerate polygon, and a
 * candidate only replaces the best by a strictly larger signed distance, so the label point is
 * inside the polygon (outside holes) whenever the polygon has interior; `degenerate` flags the
 * rest. **Precision.** The final grid spacing is
 * `2 * halfExtent * (2 / initialGridSize) * (2 / refinementGridSize)^refinementRounds /
 * refinementGridSize` (about `1e-5` of the bounds with the defaults), but the search is greedy:
 * it refines only the best coarse cells, so long thin or multi-lobed shapes can return the best
 * point of a lobe that is a bit smaller than the true pole. Raise `initialGridSize` or
 * `refinementCandidates` for such shapes.
 * f32 arithmetic in coordinates relative to the bounds center.
 *
 * Cost per feature is `segments * (initialGridSize^2 + candidates * refinementRounds *
 * refinementGridSize^2)` distance evaluations with `candidates` between 1 and
 * `refinementCandidates` (about 640 evaluations per segment per refined candidate at the
 * defaults). Features with at most `smallFeatureVertexLimit` vertices run them sequentially in one
 * invocation. Larger features run in one 64-lane workgroup: lanes score different grid cells (each
 * lane walks all segments in the same order as the serial path) and cooperate on the bounds,
 * scanline seeds and seed distances, so latency is about `segments * (grid cells / 64)`. Both
 * paths pick the same candidates with the same tie rule and give identical results.
 */
export class GPULabelPoint implements GPUCommandNodeProducer {
  /** Prefix for every node ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPULabelPointProps;
  /** Number of features. */
  readonly featureCount: number;
  /** Resolved initial grid size. */
  readonly initialGridSize: number;
  /** Resolved refinement grid size. */
  readonly refinementGridSize: number;
  /** Resolved refinement rounds. */
  readonly refinementRounds: number;
  /** Resolved number of refined candidates. */
  readonly refinementCandidates: number;
  /** Resolved vertex count above which a feature is searched by a whole workgroup. */
  readonly smallFeatureVertexLimit: number;

  constructor(props: GPULabelPointProps) {
    this.id = props.id ?? 'label-point';
    this.props = props;
    this.initialGridSize = props.initialGridSize ?? GPU_LABEL_POINT_DEFAULT_INITIAL_GRID_SIZE;
    this.refinementGridSize =
      props.refinementGridSize ?? GPU_LABEL_POINT_DEFAULT_REFINEMENT_GRID_SIZE;
    this.refinementRounds = props.refinementRounds ?? GPU_LABEL_POINT_DEFAULT_REFINEMENT_ROUNDS;
    this.refinementCandidates = props.refinementCandidates ?? 4;
    this.smallFeatureVertexLimit =
      props.smallFeatureVertexLimit ?? GPU_LABEL_POINT_DEFAULT_SMALL_FEATURE_VERTEX_LIMIT;
    const {id} = this;
    for (const [name, view] of Object.entries({
      positions: props.positions,
      ringOffsets: props.ringOffsets,
      featureRingOffsets: props.featureRingOffsets,
      ...props.output
    })) {
      if ((view as unknown) instanceof GraphVectorView) {
        throw new Error(`${id} ${name} must be a single packed view, not a chunked vector`);
      }
    }
    if (!Number.isInteger(this.initialGridSize) || this.initialGridSize < 2) {
      throw new Error(`${id} initialGridSize must be an integer of at least 2`);
    }
    if (!Number.isInteger(this.refinementGridSize) || this.refinementGridSize < 3) {
      throw new Error(`${id} refinementGridSize must be an integer of at least 3`);
    }
    if (!Number.isInteger(this.refinementRounds) || this.refinementRounds < 0) {
      throw new Error(`${id} refinementRounds must be a non-negative integer`);
    }
    if (
      !Number.isInteger(this.refinementCandidates) ||
      this.refinementCandidates < 1 ||
      this.refinementCandidates > 16
    ) {
      throw new Error(`${id} refinementCandidates must be an integer from 1 to 16`);
    }
    if (!Number.isInteger(this.smallFeatureVertexLimit) || this.smallFeatureVertexLimit < 0) {
      throw new Error(`${id} smallFeatureVertexLimit must be a non-negative integer`);
    }
    validatePackedView(props.positions, ['float32x2'], `${id} positions`);
    if (props.positions.length < 1) {
      throw new Error(`${id} needs at least one position`);
    }
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
    this.featureCount = props.featureRingOffsets
      ? props.featureRingOffsets.length - 1
      : props.ringOffsets.length - 1;
    const {points, distances, degenerate} = props.output;
    validatePackedView(points, ['float32x2'], `${id} output.points`);
    if (points.length !== this.featureCount) {
      throw new Error(`${id} output.points must hold ${this.featureCount} rows`);
    }
    if (distances) {
      validatePackedView(distances, ['float32'], `${id} output.distances`);
      if (distances.length !== this.featureCount) {
        throw new Error(`${id} output.distances must hold ${this.featureCount} rows`);
      }
    }
    if (degenerate) {
      validatePackedUint32View(degenerate, `${id} output.degenerate`);
      if (degenerate.length !== this.featureCount) {
        throw new Error(`${id} output.degenerate must hold ${this.featureCount} rows`);
      }
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      [points, distances, degenerate],
      [props.positions, props.ringOffsets, props.featureRingOffsets]
    );
  }

  /** Returns the single search node. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id} = this;
    const {points, distances, degenerate} = props.output;
    validateGraphViewsBelongToGraph(id, graph, [
      props.positions,
      props.ringOffsets,
      props.featureRingOffsets,
      points,
      distances,
      degenerate
    ]);
    const bindings: WGSLKernelBinding[] = [
      {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
      {name: 'ringOffsets', view: props.ringOffsets, type: 'u32', access: 'read'}
    ];
    if (props.featureRingOffsets) {
      bindings.push({
        name: 'featureRingOffsets',
        view: props.featureRingOffsets,
        type: 'u32',
        access: 'read'
      });
    }
    bindings.push({name: 'pointsOut', view: points, type: 'f32', access: 'read_write'});
    if (distances) {
      bindings.push({name: 'distancesOut', view: distances, type: 'f32', access: 'read_write'});
    }
    if (degenerate) {
      bindings.push({name: 'degenerateOut', view: degenerate, type: 'u32', access: 'read_write'});
    }
    const declarations = `const ROW_COUNT: u32 = ${props.positions.length}u;
const RING_COUNT: u32 = ${props.ringOffsets.length - 1}u;
const FEATURE_COUNT: u32 = ${this.featureCount}u;
const INITIAL_GRID_SIZE: u32 = ${this.initialGridSize}u;
const REFINEMENT_GRID_SIZE: u32 = ${this.refinementGridSize}u;
const REFINEMENT_ROUNDS: u32 = ${this.refinementRounds}u;
const CANDIDATE_COUNT: u32 = ${this.refinementCandidates}u;
const SMALL_FEATURE_VERTEX_LIMIT: u32 = ${this.smallFeatureVertexLimit}u;
const WORKGROUP_SIZE: u32 = ${WORKGROUP_SIZE}u;
const LARGE: f32 = 3.0e38;
${LABEL_POINT_WGSL}
fn getRingStart(feature: u32) -> u32 {
  ${props.featureRingOffsets ? 'return min(featureRingOffsets[featureRingOffsetsOffset + feature], RING_COUNT);' : 'return feature;'}
}
fn getRingEnd(feature: u32) -> u32 {
  ${props.featureRingOffsets ? 'return min(featureRingOffsets[featureRingOffsetsOffset + feature + 1u], RING_COUNT);' : 'return feature + 1u;'}
}`;
    const writeOutputs = (feature: string, emptyCondition: string) => `
  pointsOut[pointsOutOffset + 2u * ${feature}] = result.x;
  pointsOut[pointsOutOffset + 2u * ${feature} + 1u] = result.y;
  ${distances ? `distancesOut[distancesOutOffset + ${feature}] = select(max(bestDistance, 0.0), nan, ${emptyCondition});` : ''}
  ${degenerate ? `degenerateOut[degenerateOutOffset + ${feature}] = isDegenerate;` : ''}`;
    return [
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-search`,
        operation: OPERATION,
        variant: 'small',
        bindings,
        invocationCount: this.featureCount,
        declarations,
        body: /* wgsl */ `let nan = bitcast<f32>(0x7fc00000u | (index & 0u));
  let ringStart = getRingStart(index);
  let ringEnd = max(getRingEnd(index), ringStart);
  let rowStart = getRingRowStart(ringStart);
  let rowEnd = max(getRingRowStart(ringEnd), rowStart);
  if (rowEnd - rowStart > SMALL_FEATURE_VERTEX_LIMIT) {
    return;
  }
  var result = vec2<f32>(nan);
  var bestDistance = nan;
  var isDegenerate = 1u;
  if (rowEnd > rowStart) {
    var minimum = getPosition(rowStart);
    var maximum = minimum;
    for (var row = rowStart + 1u; row < rowEnd; row++) {
      let p = getPosition(row);
      minimum = min(minimum, p);
      maximum = max(maximum, p);
    }
    let origin = 0.5 * (minimum + maximum);
    let size = maximum - minimum;
    var halfSize = 0.5 * max(size.x, size.y);
    // Seeds: first vertex (on the boundary, the fallback), then scanline midpoints.
    var best = vec3<f32>(getPosition(rowStart) - origin, 0.0);
    var bestIsSet = false;
    for (var seedIndex = 0u; seedIndex < 3u; seedIndex++) {
      let fraction = select(select(0.0, -0.2, seedIndex == 1u), 0.2, seedIndex == 2u);
      let seed = getScanlineSeed(ringStart, ringEnd, origin, fraction * size.y);
      if (seed.z > 0.0) {
        let candidate = vec3<f32>(seed.xy, getSignedDistance(seed.xy, ringStart, ringEnd, origin));
        if (!bestIsSet || candidate.z > best.z) {
          best = candidate;
          bestIsSet = true;
        }
      }
    }
    if (!bestIsSet) {
      best.z = getSignedDistance(best.xy, ringStart, ringEnd, origin);
    }
    if (halfSize > 0.0) {
      var top = array<vec3<f32>, CANDIDATE_COUNT>();
      for (var slot = 0u; slot < CANDIDATE_COUNT; slot++) {
        top[slot] = vec3<f32>(0.0, 0.0, -LARGE);
      }
      scoreGrid(vec2<f32>(0.0), halfSize, INITIAL_GRID_SIZE, ringStart, ringEnd, origin, &top);
      let cellHalfSize = halfSize / f32(INITIAL_GRID_SIZE);
      for (var slot = 0u; slot < CANDIDATE_COUNT; slot++) {
        var local = top[slot];
        // Candidates are sorted by distance, so once the cell bound cannot beat the best, stop.
        if (local.z + 1.4142135 * cellHalfSize <= best.z) {
          break;
        }
        if (local.z > best.z) {
          best = local;
        }
        var windowHalfSize = 2.0 * cellHalfSize;
        for (var round = 0u; round < REFINEMENT_ROUNDS; round++) {
          local = searchWindow(local.xy, windowHalfSize, REFINEMENT_GRID_SIZE, ringStart, ringEnd, origin, local);
          windowHalfSize = 2.0 * windowHalfSize / f32(REFINEMENT_GRID_SIZE);
        }
        if (local.z > best.z) {
          best = local;
        }
      }
    }
    result = origin + best.xy;
    bestDistance = best.z;
    isDegenerate = select(0u, 1u, best.z <= 0.0);
  }${writeOutputs('index', 'rowEnd <= rowStart')}`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-search-workgroup`,
        operation: OPERATION,
        variant: 'workgroup',
        bindings,
        invocationCount: this.featureCount * WORKGROUP_SIZE,
        workgroupSize: WORKGROUP_SIZE,
        guardIndex: false,
        declarations,
        // One workgroup per feature. Every barrier sits in workgroup-uniform control flow: the
        // feature comes from the workgroup ID, and data-dependent decisions are read back with
        // workgroupUniformLoad.
        body: /* wgsl */ `let feature = workgroupIndex;
  let lane = localInvocationIndex;
  if (feature >= FEATURE_COUNT) {
    return;
  }
  let nan = bitcast<f32>(0x7fc00000u | (feature & 0u));
  let ringStart = getRingStart(feature);
  let ringEnd = max(getRingEnd(feature), ringStart);
  let rowStart = getRingRowStart(ringStart);
  let rowEnd = max(getRingRowStart(ringEnd), rowStart);
  if (lane == 0u) {
    sharedUint[0] = select(0u, 1u, rowEnd - rowStart > SMALL_FEATURE_VERTEX_LIMIT);
  }
  if (workgroupUniformLoad(&sharedUint[0]) == 0u) {
    return;
  }
  var localMinimum = vec2<f32>(LARGE);
  var localNegativeMaximum = vec2<f32>(LARGE);
  for (var row = rowStart + lane; row < rowEnd; row += WORKGROUP_SIZE) {
    let p = getPosition(row);
    localMinimum = min(localMinimum, p);
    localNegativeMaximum = min(localNegativeMaximum, -p);
  }
  let minimum = vec2<f32>(reduceMinFloat(lane, localMinimum.x), reduceMinFloat(lane, localMinimum.y));
  let maximum = -vec2<f32>(reduceMinFloat(lane, localNegativeMaximum.x), reduceMinFloat(lane, localNegativeMaximum.y));
  let origin = 0.5 * (minimum + maximum);
  let size = maximum - minimum;
  let halfSize = 0.5 * max(size.x, size.y);
  var best = vec3<f32>(getPosition(rowStart) - origin, 0.0);
  var bestIsSet = false;
  for (var seedIndex = 0u; seedIndex < 3u; seedIndex++) {
    let fraction = select(select(0.0, -0.2, seedIndex == 1u), 0.2, seedIndex == 2u);
    let seed = getScanlineSeedCooperative(lane, ringStart, ringEnd, origin, fraction * size.y);
    if (seed.z > 0.0) {
      let candidate = vec3<f32>(seed.xy, getSignedDistanceCooperative(lane, seed.xy, ringStart, ringEnd, origin));
      if (!bestIsSet || candidate.z > best.z) {
        best = candidate;
        bestIsSet = true;
      }
    }
  }
  if (!bestIsSet) {
    best.z = getSignedDistanceCooperative(lane, best.xy, ringStart, ringEnd, origin);
  }
  if (halfSize > 0.0) {
    var topDistance = array<f32, CANDIDATE_COUNT>();
    var topCell = array<u32, CANDIDATE_COUNT>();
    for (var slot = 0u; slot < CANDIDATE_COUNT; slot++) {
      topDistance[slot] = -LARGE;
      topCell[slot] = NO_CELL;
    }
    // Each lane scores the first-grid cells lane, lane + 64, ... in order and keeps its best few.
    let initialCellSize = 2.0 * halfSize / f32(INITIAL_GRID_SIZE);
    for (var cellIndex = lane; cellIndex < INITIAL_GRID_SIZE * INITIAL_GRID_SIZE; cellIndex += WORKGROUP_SIZE) {
      let p = getGridPoint(vec2<f32>(0.0), halfSize, initialCellSize, INITIAL_GRID_SIZE, cellIndex);
      var entryDistance = getSignedDistance(p, ringStart, ringEnd, origin);
      var entryCell = cellIndex;
      for (var slot = 0u; slot < CANDIDATE_COUNT; slot++) {
        if (entryDistance > topDistance[slot]) {
          let displacedDistance = topDistance[slot];
          let displacedCell = topCell[slot];
          topDistance[slot] = entryDistance;
          topCell[slot] = entryCell;
          entryDistance = displacedDistance;
          entryCell = displacedCell;
        }
      }
    }
    let cellHalfSize = halfSize / f32(INITIAL_GRID_SIZE);
    for (var slot = 0u; slot < CANDIDATE_COUNT; slot++) {
      // Next best first-grid cell over all lanes: highest distance, earliest cell on ties.
      let winner = reduceArgMax(lane, topDistance[0], topCell[0]);
      if (winner.cell != NO_CELL && lane == winner.cell % WORKGROUP_SIZE) {
        for (var shift = 0u; shift + 1u < CANDIDATE_COUNT; shift++) {
          topDistance[shift] = topDistance[shift + 1u];
          topCell[shift] = topCell[shift + 1u];
        }
        topDistance[CANDIDATE_COUNT - 1u] = -LARGE;
        topCell[CANDIDATE_COUNT - 1u] = NO_CELL;
      }
      var local = vec3<f32>(0.0, 0.0, -LARGE);
      if (winner.cell != NO_CELL) {
        local = vec3<f32>(getGridPoint(vec2<f32>(0.0), halfSize, initialCellSize, INITIAL_GRID_SIZE, winner.cell), winner.distance);
      }
      if (local.z + 1.4142135 * cellHalfSize <= best.z) {
        break;
      }
      if (local.z > best.z) {
        best = local;
      }
      var windowHalfSize = 2.0 * cellHalfSize;
      for (var round = 0u; round < REFINEMENT_ROUNDS; round++) {
        local = searchWindowCooperative(lane, local.xy, windowHalfSize, REFINEMENT_GRID_SIZE, ringStart, ringEnd, origin, local);
        windowHalfSize = 2.0 * windowHalfSize / f32(REFINEMENT_GRID_SIZE);
      }
      if (local.z > best.z) {
        best = local;
      }
    }
  }
  if (lane == 0u) {
    let result = origin + best.xy;
    let bestDistance = best.z;
    let isDegenerate = select(0u, 1u, best.z <= 0.0);${writeOutputs('feature', 'false')}
  }`
      })
    ];
  }
}

/**
 * WGSL helpers over the bound `positions`, `ringOffsets` and the constants `ROW_COUNT`,
 * `RING_COUNT`, `LARGE`: ring accessors, signed distance, scanline seeds and grid search.
 * Candidates are `vec3(x, y, signedDistance)` in coordinates relative to the bounds center.
 *
 * @internal
 */
const LABEL_POINT_WGSL = /* wgsl */ `
fn getPosition(row: u32) -> vec2<f32> {
  return vec2<f32>(positions[positionsOffset + 2u * row], positions[positionsOffset + 2u * row + 1u]);
}

fn getRingRowStart(ring: u32) -> u32 {
  return min(ringOffsets[ringOffsetsOffset + ring], ROW_COUNT);
}

fn getRingRowEnd(ring: u32) -> u32 {
  return min(ringOffsets[ringOffsetsOffset + ring + 1u], ROW_COUNT);
}

fn getSegmentDistanceSquared(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>) -> f32 {
  let edge = b - a;
  let lengthSquared = dot(edge, edge);
  var t = 0.0;
  if (lengthSquared > 0.0) {
    t = clamp(dot(p - a, edge) / lengthSquared, 0.0, 1.0);
  }
  let offset = p - (a + t * edge);
  return dot(offset, offset);
}

// Signed distance to the boundary of all rings: positive inside under the even-odd rule.
fn getSignedDistance(p: vec2<f32>, ringStart: u32, ringEnd: u32, origin: vec2<f32>) -> f32 {
  var minimumSquared = LARGE;
  var inside = false;
  for (var ring = ringStart; ring < ringEnd; ring++) {
    let rowStart = getRingRowStart(ring);
    let rowEnd = getRingRowEnd(ring);
    if (rowEnd > rowStart) {
      var previous = getPosition(rowEnd - 1u) - origin;
      for (var row = rowStart; row < rowEnd; row++) {
        let current = getPosition(row) - origin;
        minimumSquared = min(minimumSquared, getSegmentDistanceSquared(p, previous, current));
        if ((current.y > p.y) != (previous.y > p.y)) {
          let crossingX = previous.x + (p.y - previous.y) * (current.x - previous.x) / (current.y - previous.y);
          if (p.x < crossingX) {
            inside = !inside;
          }
        }
        previous = current;
      }
    }
  }
  let distance = sqrt(minimumSquared);
  return select(-distance, distance, inside);
}

// Midpoint of the leftmost interior interval on the horizontal line y = lineY (relative to the
// bounds center). Returns (x, y, 1) or z = 0 when there is none.
fn getScanlineSeed(ringStart: u32, ringEnd: u32, origin: vec2<f32>, lineY: f32) -> vec3<f32> {
  var firstX = LARGE;
  var firstCount = 0u;
  for (var scanPass = 0u; scanPass < 2u; scanPass++) {
    var secondX = LARGE;
    for (var ring = ringStart; ring < ringEnd; ring++) {
      let rowStart = getRingRowStart(ring);
      let rowEnd = getRingRowEnd(ring);
      if (rowEnd > rowStart) {
        var previous = getPosition(rowEnd - 1u) - origin;
        for (var row = rowStart; row < rowEnd; row++) {
          let current = getPosition(row) - origin;
          if ((current.y > lineY) != (previous.y > lineY)) {
            let crossingX = previous.x + (lineY - previous.y) * (current.x - previous.x) / (current.y - previous.y);
            if (scanPass == 0u) {
              if (crossingX < firstX) {
                firstX = crossingX;
                firstCount = 1u;
              } else if (crossingX == firstX) {
                firstCount += 1u;
              }
            } else if (crossingX > firstX) {
              secondX = min(secondX, crossingX);
            }
          }
          previous = current;
        }
      }
    }
    if (scanPass == 1u) {
      if (firstCount % 2u == 1u && secondX < LARGE) {
        return vec3<f32>(0.5 * (firstX + secondX), lineY, 1.0);
      }
    }
  }
  return vec3<f32>(0.0);
}

// Scores an n x n grid like searchWindow but keeps the best CANDIDATE_COUNT cells, sorted by
// signed distance descending (ties keep the earlier cell).
fn scoreGrid(
  center: vec2<f32>,
  halfSize: f32,
  gridSize: u32,
  ringStart: u32,
  ringEnd: u32,
  origin: vec2<f32>,
  top: ptr<function, array<vec3<f32>, CANDIDATE_COUNT>>
) {
  let cell = 2.0 * halfSize / f32(gridSize);
  for (var j = 0u; j < gridSize; j++) {
    for (var i = 0u; i < gridSize; i++) {
      let p = center - vec2<f32>(halfSize) + cell * (vec2<f32>(f32(i), f32(j)) + vec2<f32>(0.5));
      var entry = vec3<f32>(p, getSignedDistance(p, ringStart, ringEnd, origin));
      for (var slot = 0u; slot < CANDIDATE_COUNT; slot++) {
        if (entry.z > (*top)[slot].z) {
          let displaced = (*top)[slot];
          (*top)[slot] = entry;
          entry = displaced;
        }
      }
    }
  }
}

// Scores an n x n grid of cell centers over [center - halfSize, center + halfSize]^2 and returns
// the better of the grid winner and \`best\` (ties keep the earlier candidate).
fn searchWindow(
  center: vec2<f32>,
  halfSize: f32,
  gridSize: u32,
  ringStart: u32,
  ringEnd: u32,
  origin: vec2<f32>,
  bestIn: vec3<f32>
) -> vec3<f32> {
  var best = bestIn;
  let cell = 2.0 * halfSize / f32(gridSize);
  for (var j = 0u; j < gridSize; j++) {
    for (var i = 0u; i < gridSize; i++) {
      let p = center - vec2<f32>(halfSize) + cell * (vec2<f32>(f32(i), f32(j)) + vec2<f32>(0.5));
      let distance = getSignedDistance(p, ringStart, ringEnd, origin);
      if (distance > best.z) {
        best = vec3<f32>(p, distance);
      }
    }
  }
  return best;
}

// Workgroup-cooperative variants used by the one-workgroup-per-feature kernel. Reductions end in
// workgroupUniformLoad so their results are workgroup-uniform, which keeps later barriers legal.
const NO_CELL: u32 = 0xffffffffu;
var<workgroup> sharedUint: array<u32, 1>;
var<workgroup> reduceFloat: array<f32, WORKGROUP_SIZE>;
var<workgroup> reduceUint: array<u32, WORKGROUP_SIZE>;

struct ArgMax {
  distance: f32,
  cell: u32
}

fn reduceMinFloat(lane: u32, value: f32) -> f32 {
  reduceFloat[lane] = value;
  workgroupBarrier();
  for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride / 2u) {
    if (lane < stride) {
      reduceFloat[lane] = min(reduceFloat[lane], reduceFloat[lane + stride]);
    }
    workgroupBarrier();
  }
  return workgroupUniformLoad(&reduceFloat[0]);
}

fn reduceSumUint(lane: u32, value: u32) -> u32 {
  reduceUint[lane] = value;
  workgroupBarrier();
  for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride / 2u) {
    if (lane < stride) {
      reduceUint[lane] = reduceUint[lane] + reduceUint[lane + stride];
    }
    workgroupBarrier();
  }
  return workgroupUniformLoad(&reduceUint[0]);
}

// Highest distance over the workgroup, the lowest cell index among equal distances.
fn reduceArgMax(lane: u32, distance: f32, cell: u32) -> ArgMax {
  reduceFloat[lane] = distance;
  reduceUint[lane] = cell;
  workgroupBarrier();
  for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride / 2u) {
    if (lane < stride) {
      let otherDistance = reduceFloat[lane + stride];
      let otherCell = reduceUint[lane + stride];
      if (otherDistance > reduceFloat[lane] || (otherDistance == reduceFloat[lane] && otherCell < reduceUint[lane])) {
        reduceFloat[lane] = otherDistance;
        reduceUint[lane] = otherCell;
      }
    }
    workgroupBarrier();
  }
  let winnerDistance = workgroupUniformLoad(&reduceFloat[0]);
  let winnerCell = workgroupUniformLoad(&reduceUint[0]);
  return ArgMax(winnerDistance, winnerCell);
}

fn getGridPoint(center: vec2<f32>, halfSize: f32, cell: f32, gridSize: u32, cellIndex: u32) -> vec2<f32> {
  let i = cellIndex % gridSize;
  let j = cellIndex / gridSize;
  return center - vec2<f32>(halfSize) + cell * (vec2<f32>(f32(i), f32(j)) + vec2<f32>(0.5));
}

// Same value as getSignedDistance: lanes stride over the segments of every ring.
fn getSignedDistanceCooperative(lane: u32, p: vec2<f32>, ringStart: u32, ringEnd: u32, origin: vec2<f32>) -> f32 {
  var minimumSquared = LARGE;
  var crossings = 0u;
  for (var ring = ringStart; ring < ringEnd; ring++) {
    let ringRowStart = getRingRowStart(ring);
    let ringRowEnd = getRingRowEnd(ring);
    for (var row = ringRowStart + lane; row < ringRowEnd; row += WORKGROUP_SIZE) {
      let current = getPosition(row) - origin;
      let previous = getPosition(select(row - 1u, ringRowEnd - 1u, row == ringRowStart)) - origin;
      minimumSquared = min(minimumSquared, getSegmentDistanceSquared(p, previous, current));
      if ((current.y > p.y) != (previous.y > p.y)) {
        let crossingX = previous.x + (p.y - previous.y) * (current.x - previous.x) / (current.y - previous.y);
        if (p.x < crossingX) {
          crossings += 1u;
        }
      }
    }
  }
  let distance = sqrt(reduceMinFloat(lane, minimumSquared));
  return select(-distance, distance, (reduceSumUint(lane, crossings) & 1u) == 1u);
}

// Same value as getScanlineSeed. Pass 0 finds the leftmost crossing, pass 1 counts the crossings
// sharing it, pass 2 finds the next crossing to its right.
fn getScanlineSeedCooperative(lane: u32, ringStart: u32, ringEnd: u32, origin: vec2<f32>, lineY: f32) -> vec3<f32> {
  var firstX = LARGE;
  var firstCount = 0u;
  for (var scanPass = 0u; scanPass < 3u; scanPass++) {
    var localX = LARGE;
    var localCount = 0u;
    for (var ring = ringStart; ring < ringEnd; ring++) {
      let ringRowStart = getRingRowStart(ring);
      let ringRowEnd = getRingRowEnd(ring);
      for (var row = ringRowStart + lane; row < ringRowEnd; row += WORKGROUP_SIZE) {
        let current = getPosition(row) - origin;
        let previous = getPosition(select(row - 1u, ringRowEnd - 1u, row == ringRowStart)) - origin;
        if ((current.y > lineY) != (previous.y > lineY)) {
          let crossingX = previous.x + (lineY - previous.y) * (current.x - previous.x) / (current.y - previous.y);
          if (scanPass == 0u) {
            localX = min(localX, crossingX);
          } else if (scanPass == 1u) {
            if (crossingX == firstX) {
              localCount += 1u;
            }
          } else if (crossingX > firstX) {
            localX = min(localX, crossingX);
          }
        }
      }
    }
    if (scanPass == 0u) {
      firstX = reduceMinFloat(lane, localX);
    } else if (scanPass == 1u) {
      firstCount = reduceSumUint(lane, localCount);
    } else {
      let secondX = reduceMinFloat(lane, localX);
      if (firstCount % 2u == 1u && secondX < LARGE) {
        return vec3<f32>(0.5 * (firstX + secondX), lineY, 1.0);
      }
    }
  }
  return vec3<f32>(0.0);
}

// Same result as searchWindow: every lane scores cells lane, lane + 64, ... and the best cell
// (earliest on ties) replaces bestIn only by a strictly larger distance.
fn searchWindowCooperative(
  lane: u32,
  center: vec2<f32>,
  halfSize: f32,
  gridSize: u32,
  ringStart: u32,
  ringEnd: u32,
  origin: vec2<f32>,
  bestIn: vec3<f32>
) -> vec3<f32> {
  let cell = 2.0 * halfSize / f32(gridSize);
  var laneDistance = -LARGE;
  var laneCell = NO_CELL;
  for (var cellIndex = lane; cellIndex < gridSize * gridSize; cellIndex += WORKGROUP_SIZE) {
    let distance = getSignedDistance(getGridPoint(center, halfSize, cell, gridSize, cellIndex), ringStart, ringEnd, origin);
    if (distance > laneDistance) {
      laneDistance = distance;
      laneCell = cellIndex;
    }
  }
  let winner = reduceArgMax(lane, laneDistance, laneCell);
  if (winner.cell != NO_CELL && winner.distance > bestIn.z) {
    return vec3<f32>(getGridPoint(center, halfSize, cell, gridSize, winner.cell), winner.distance);
  }
  return bestIn;
}
`;
