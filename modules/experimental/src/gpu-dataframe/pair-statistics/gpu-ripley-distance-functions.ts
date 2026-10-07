// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  validatePackedView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {
  createFillNode,
  createWGSLKernelNode,
  type WGSLKernelBinding
} from '../../utils/wgsl-kernel-nodes';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../utils/gpu-contributor-utils';
import {getPairHistogramNodes, validatePairHistogramShape} from './pair-histogram';
import {
  getPairStatisticsInputNodes,
  getPairStatisticsSharedWGSL,
  PAIR_STATISTICS_FLOAT_WGSL,
  validatePairStatisticsInputs
} from './pair-statistics-grid';
import {GPU_RIPLEY_DISTANCE_PARAMETER_LENGTH} from './ripley-distance-parameters';

const OPERATION = 'GPURipleyDistanceFunctions';

/** Largest `radiusCount`. */
const MAXIMUM_RADIUS_COUNT = 256;

/** Largest number of reference locations of F: counts stay exact `u32` sums. */
const MAXIMUM_REFERENCE_COUNT = 2 ** 24;

/**
 * Accumulator channels per radius of one function: 0 raw, 1 border numerator, 2 border subtract,
 * 3 border denominator, 4 uncensored observations (Kaplan-Meier events), 5 censored observations,
 * 6 Hanisch weights of uncensored observations (fixed point), 7 Hanisch weight of observations
 * with no neighbor within the maximum distance (slot 0 only).
 */
const CHANNEL_COUNT = 8;

/**
 * Properties for {@link GPURipleyDistanceFunctions}.
 *
 * Per-frame (no rebuild or recompile): the contents of `positions`, `mask` and `parameters`
 * (window bounds, maximum distance, edge-correction mode: none, border, Kaplan-Meier, Hanisch). Compile-time: the row count,
 * `gridSize`, `referenceGrid`, `radiusCount` and which optional outputs are present.
 */
export type GPURipleyDistanceFunctionsProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'ripley-distance-functions'`. */
  id?: string;
  /** Packed planar points, one row per event. At least one and fewer than 2^24 rows. */
  positions: GraphDataView<'float32x2'>;
  /**
   * Per-frame parameters: packed float32 view of at least `GPU_RIPLEY_DISTANCE_PARAMETER_LENGTH`
   * elements written with `getGPURipleyDistanceParameterValues`.
   */
  parameters: GraphDataView<'float32'>;
  /**
   * Maximum `[columns, rows]` of the neighbor-search cell lattice. Compile-time. Results never
   * depend on it, only speed.
   */
  gridSize: readonly [number, number];
  /**
   * `[columns, rows]` of the regular lattice of reference locations used for F: the window is cut
   * into this many equal cells and each center is one location. Compile-time; the product must be
   * in `[1, 2^24]`. A finer lattice reduces the sampling noise of F, which is exact in the limit.
   */
  referenceGrid: readonly [number, number];
  /**
   * Number of radii. Compile-time, an integer in `[1, 256]`. Radius `b` is
   * `maximumDistance * (b + 1) / radiusCount`.
   */
  radiusCount: number;
  /** Optional row selection: nonzero includes the row. */
  mask?: GraphDataView<'uint32'>;
  /**
   * Caller-owned nearest-neighbor distance function `G(r)`, `radiusCount` rows: the fraction of
   * events whose nearest other event is within `r`. Quiet NaN where undefined.
   */
  g?: GraphDataView<'float32'>;
  /**
   * Caller-owned empty-space function `F(r)`, `radiusCount` rows: the fraction of reference
   * locations whose nearest event is within `r`. Quiet NaN where undefined.
   */
  f?: GraphDataView<'float32'>;
  /**
   * Caller-owned `J(r) = (1 - G(r)) / (1 - F(r))`, `radiusCount` rows. Below 1 means clustering,
   * above 1 regularity, 1 under complete spatial randomness. NaN where `G`, `F` or `1 - F` are not
   * usable.
   */
  j?: GraphDataView<'float32'>;
  /** Optional caller-owned radii, `radiusCount` rows. */
  radii?: GraphDataView<'float32'>;
};

/**
 * Ripley's distance functions of a planar point pattern in a rectangular window: the
 * nearest-neighbor function G, the empty-space function F and their ratio J
 * (spatstat `Gest`, `Fest`, `Jest`, here without kernel smoothing).
 *
 * Definition, which the GPU result matches within the bounds below:
 * - The window is the per-frame bounds rectangle. Included events have a nonzero mask (when
 *   given) and a finite position inside the window; `n` is their count. `d_i` is the distance
 *   from event `i` to its nearest other included event, searched through the 3x3 cell
 *   neighborhood, which covers every event within `maximumDistance`; an event with none within
 *   `maximumDistance` counts in no numerator.
 * - Radius `r_b = maximumDistance * (b + 1) / radiusCount`.
 * - `'none'`: `G(r) = #{i: d_i <= r} / n` and `F(r) = #{u: e_u <= r} / m`, where `e_u` is the
 *   distance from reference location `u` to its nearest event and `m` is the reference count.
 * - `'border'` (reduced sample, spatstat `correction = "rs"`): with `b_i` the distance from a
 *   point to the window boundary, `G(r) = #{i: d_i <= r, b_i >= r} / #{i: b_i >= r}`, and the same
 *   for F over the reference locations. It is NaN when no point is farther than `r` from the
 *   boundary. `'border'` is conservative: it throws away points near the edge.
 * - `'kaplan-meier'` (spatstat `"km"`): every point is an observation `o = min(d, b)`, uncensored
 *   when `d <= b`. With events `D_s` and censorings `C_s` binned on the radius grid and
 *   `N_s = #{o in slot >= s}`, `G(r_s) = 1 - prod_{t <= s} (1 - D_t / N_t)`; the same for F over
 *   the reference locations. Points with no neighbor within `maximumDistance` are censored at
 *   their border distance (or never, when it exceeds `maximumDistance`), so it is exact.
 * - `'hanisch'` (spatstat `"han"` for G, the Chiu-Stoyan weighting `"cs"` for F): uncensored points
 *   (`d <= b`) are weighted by `1 / |W eroded by d|` (rectangle area `(w - 2d) (h - 2d)`), and
 *   `G(r) = sum_{d <= r} weight / sum weight`. Weights are accumulated in 14-bit fixed point
 *   (relative error about 1e-4) and capped at 1024 times the window area ratio. A point with no
 *   neighbor within `maximumDistance` has an unknown `d`: if its border distance is at least
 *   `maximumDistance` it enters the denominator with `d = maximumDistance` (a lower bound on its
 *   weight, so G is biased slightly high), otherwise it is exactly censored. The estimate is
 *   exact when every point has a neighbor within `maximumDistance`.
 * - `J(r) = (1 - G(r)) / (1 - F(r))`, NaN where either function is NaN or `F(r) >= 1`.
 * - Undefined results (`n < 2` for G, `n < 1` for F, zero reference count, an empty border set)
 *   are quiet NaN.
 *
 * Determinism and precision: G sums exact 64-bit integers in workgroup-local accumulators, F sums
 * exact integers with global `atomicAdd`s, and both finish with the same prefix sums in a fixed
 * order, so results are bitwise reproducible and match an oracle to f32 rounding of the final
 * divisions, except for distances that land within f32 rounding of a radius.
 *
 * Cost: G is one pass over the 3x3 neighbor cells of every event (as `GPURipley`); F is one such
 * pass per reference location, with global atomics on `radiusCount` bins.
 */
export class GPURipleyDistanceFunctions implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPURipleyDistanceFunctionsProps;

  constructor(props: GPURipleyDistanceFunctionsProps) {
    this.id = props.id ?? 'ripley-distance-functions';
    this.props = props;
    const id = this.id;
    validatePairStatisticsInputs(id, {
      ...props,
      parameterLength: GPU_RIPLEY_DISTANCE_PARAMETER_LENGTH
    });
    const {radiusCount, referenceGrid} = props;
    if (!Number.isInteger(radiusCount) || radiusCount < 1 || radiusCount > MAXIMUM_RADIUS_COUNT) {
      throw new Error(`${id} radiusCount must be an integer in [1, ${MAXIMUM_RADIUS_COUNT}]`);
    }
    validatePairHistogramShape(id, radiusCount, CHANNEL_COUNT);
    if (
      referenceGrid.length !== 2 ||
      !Number.isInteger(referenceGrid[0]) ||
      !Number.isInteger(referenceGrid[1]) ||
      referenceGrid[0] < 1 ||
      referenceGrid[1] < 1 ||
      referenceGrid[0] * referenceGrid[1] > MAXIMUM_REFERENCE_COUNT
    ) {
      throw new Error(
        `${id} referenceGrid must be two positive integers with a product of at most ${MAXIMUM_REFERENCE_COUNT}`
      );
    }
    if (!props.g && !props.f && !props.j) {
      throw new Error(`${id} needs at least one of g, f or j`);
    }
    for (const [name, view] of [
      ['g', props.g],
      ['f', props.f],
      ['j', props.j],
      ['radii', props.radii]
    ] as const) {
      if (view) {
        validatePackedView(view, ['float32'], `${id} ${name}`);
        if (view.length < radiusCount) {
          throw new Error(`${id} ${name} must hold radiusCount rows`);
        }
      }
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      [props.g, props.f, props.j, props.radii],
      [props.positions, props.parameters, props.mask]
    );
  }

  /** Returns the G histogram, F reference, and finishing nodes in dependency order. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {positions, parameters, gridSize, radiusCount, referenceGrid} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      positions,
      parameters,
      props.mask,
      props.g,
      props.f,
      props.j,
      props.radii
    ]);
    const referenceCount = referenceGrid[0] * referenceGrid[1];
    const inputs = getPairStatisticsInputNodes<Parameters>(graph, {
      id,
      operation: OPERATION,
      positions,
      parameters,
      gridSize,
      mask: props.mask
    });
    const nodes = inputs.nodes;
    const constantsWGSL = /* wgsl */ `
const RADIUS_COUNT: u32 = ${radiusCount}u;
const PI: f32 = 3.14159265358979;
`;
    // Needs the Lattice type, so only the kernels that include the shared pair-statistics WGSL.
    const observationWGSL = /* wgsl */ `${constantsWGSL}
const HANISCH_SCALE: f32 = 16384.0;
const HANISCH_CAP: f32 = 1024.0;

// Fixed-point Hanisch weight |W| / |W eroded by distance| of one observation; at most 2^24.
fn getHanischAmount(lattice: Lattice, distance: f32) -> u32 {
  let width = lattice.maximumX - lattice.minimumX;
  let height = lattice.maximumY - lattice.minimumY;
  let eroded = max(width - 2.0 * distance, 0.0) * max(height - 2.0 * distance, 0.0);
  let ratio = select(HANISCH_CAP, min(width * height / eroded, HANISCH_CAP), eroded > 0.0);
  return u32(ratio * HANISCH_SCALE + 0.5);
}

// Slot of a censoring distance; slots at or above RADIUS_COUNT lie beyond every radius.
fn getCensorSlot(distance: f32, step: f32) -> u32 {
  return min(u32(max(ceil(distance / step) - 1.0, 0.0)), RADIUS_COUNT);
}
`;
    // G: one focus per event, nearest other event within the maximum distance.
    const histogram = getPairHistogramNodes<Parameters>(graph, {
      id: `${id}-g`,
      operation: OPERATION,
      positions,
      parameters,
      gridSize,
      sortedRows: inputs.sortedRows,
      cellOffsets: inputs.cellOffsets,
      slotCount: radiusCount,
      channelCount: CHANNEL_COUNT,
      pairOrder: 'ordered',
      declarations: observationWGSL,
      focusPrologue: `var nearestDistance = lattice.maximumDistance;
    var hasNeighbor = false;
    let borderDistance = min(min(x - lattice.minimumX, lattice.maximumX - x), min(y - lattice.minimumY, lattice.maximumY - y));
    let borderCount = min(u32(floor(borderDistance / lattice.maximumDistance * f32(RADIUS_COUNT))), RADIUS_COUNT);`,
      pairAction: `if (!hasNeighbor || pairDistance < nearestDistance) {
              nearestDistance = pairDistance;
            }
            hasNeighbor = true;`,
      focusEpilogue: `accumulate(borderCount, 3u, 1u);
    var annulus = 0u;
    if (hasNeighbor) {
      let scaled = nearestDistance / lattice.maximumDistance * f32(RADIUS_COUNT);
      annulus = min(u32(max(ceil(scaled) - 1.0, 0.0)), RADIUS_COUNT - 1u);
      accumulate(annulus, 0u, 1u);
      if (annulus < borderCount) {
        accumulate(annulus, 1u, 1u);
        accumulate(borderCount, 2u, 1u);
      }
    }
    // Kaplan-Meier and Hanisch: observed when the neighbor is no farther than the border.
    if (hasNeighbor && nearestDistance <= borderDistance) {
      accumulate(annulus, 4u, 1u);
      accumulate(annulus, 6u, getHanischAmount(lattice, nearestDistance));
    } else {
      accumulate(getCensorSlot(borderDistance, lattice.maximumDistance / f32(RADIUS_COUNT)), 5u, 1u);
      if (!hasNeighbor && borderDistance >= lattice.maximumDistance) {
        accumulate(0u, 7u, getHanischAmount(lattice, lattice.maximumDistance));
      }
    }`
    });
    nodes.push(...histogram.nodes);

    // F: one thread per reference location, difference arrays over radius slots 0..radiusCount.
    const referenceAccumulators = createTransientView(
      graph,
      `${id}-f-accumulators`,
      'uint32',
      (radiusCount + 1) * CHANNEL_COUNT * 2
    );
    nodes.push(
      createFillNode<Parameters>(graph, {
        id: `${id}-f-clear`,
        operation: OPERATION,
        view: referenceAccumulators,
        type: 'u32',
        value: '0u'
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-f-reference`,
        operation: OPERATION,
        variant: 'f-reference',
        bindings: [
          {name: 'parameters', view: parameters, type: 'f32', access: 'read'},
          // Cell-ordered `[x bits, y bits, row]` triples built for the G histogram.
          {name: 'sortedPoints', view: histogram.sortedPoints, type: 'u32', access: 'read'},
          {name: 'cellOffsets', view: inputs.cellOffsets, type: 'u32', access: 'read'},
          {
            name: 'referenceAccumulators',
            view: referenceAccumulators,
            type: 'atomic<u32>',
            access: 'read_write'
          }
        ],
        invocationCount: referenceCount,
        declarations: `${getPairStatisticsSharedWGSL(gridSize)}
${observationWGSL}
const REFERENCE_COLUMNS: u32 = ${referenceGrid[0]}u;
const REFERENCE_ROWS: u32 = ${referenceGrid[1]}u;
const CHANNELS: u32 = ${CHANNEL_COUNT}u;

// Exact unsigned 64-bit sums: low and high words per accumulator, carry from the atomicAdd result.
fn addReference(slot: u32, channel: u32, amount: u32) {
  let word = (referenceAccumulatorsOffset + slot * CHANNELS + channel) * 2u;
  let previous = atomicAdd(&referenceAccumulators[word], amount);
  if (previous > 0xffffffffu - amount) {
    atomicAdd(&referenceAccumulators[word + 1u], 1u);
  }
}`,
        body: `let lattice = readLattice();
  if (lattice.valid) {
    let referenceColumn = index % REFERENCE_COLUMNS;
    let referenceRow = index / REFERENCE_COLUMNS;
    let x = lattice.minimumX + (f32(referenceColumn) + 0.5) * (lattice.maximumX - lattice.minimumX) / f32(REFERENCE_COLUMNS);
    let y = lattice.minimumY + (f32(referenceRow) + 0.5) * (lattice.maximumY - lattice.minimumY) / f32(REFERENCE_ROWS);
    let borderDistance = min(min(x - lattice.minimumX, lattice.maximumX - x), min(y - lattice.minimumY, lattice.maximumY - y));
    let borderCount = min(u32(floor(borderDistance / lattice.maximumDistance * f32(RADIUS_COUNT))), RADIUS_COUNT);
    var nearestSquared = lattice.radiusSquared;
    var hasEvent = false;
    let column = getCellColumn(lattice, x);
    let row = getCellRow(lattice, y);
    let firstColumn = max(column, 1u) - 1u;
    let lastColumn = min(column + 1u, lattice.columns - 1u);
    let firstRow = max(row, 1u) - 1u;
    let lastRow = min(row + 1u, lattice.rows - 1u);
    for (var cellRow = firstRow; cellRow <= lastRow; cellRow++) {
      let rowBase = cellRow * lattice.columns;
      let begin = cellOffsets[cellOffsetsOffset + rowBase + firstColumn];
      let end = cellOffsets[cellOffsetsOffset + rowBase + lastColumn + 1u];
      for (var candidateSlot = begin; candidateSlot < end; candidateSlot++) {
        let deltaX = bitcast<f32>(sortedPoints[sortedPointsOffset + candidateSlot * 3u]) - x;
        let deltaY = bitcast<f32>(sortedPoints[sortedPointsOffset + candidateSlot * 3u + 1u]) - y;
        let distanceSquared = deltaX * deltaX + deltaY * deltaY;
        if (distanceSquared <= lattice.radiusSquared && (!hasEvent || distanceSquared < nearestSquared)) {
          nearestSquared = distanceSquared;
          hasEvent = true;
        }
      }
    }
    addReference(borderCount, 3u, 1u);
    let nearestDistance = sqrt(nearestSquared);
    var annulus = 0u;
    if (hasEvent) {
      let scaled = nearestDistance / lattice.maximumDistance * f32(RADIUS_COUNT);
      annulus = min(u32(max(ceil(scaled) - 1.0, 0.0)), RADIUS_COUNT - 1u);
      addReference(annulus, 0u, 1u);
      if (annulus < borderCount) {
        addReference(annulus, 1u, 1u);
        addReference(borderCount, 2u, 1u);
      }
    }
    if (hasEvent && nearestDistance <= borderDistance) {
      addReference(annulus, 4u, 1u);
      addReference(annulus, 6u, getHanischAmount(lattice, nearestDistance));
    } else {
      addReference(getCensorSlot(borderDistance, lattice.maximumDistance / f32(RADIUS_COUNT)), 5u, 1u);
      if (!hasEvent && borderDistance >= lattice.maximumDistance) {
        addReference(0u, 7u, getHanischAmount(lattice, lattice.maximumDistance));
      }
    }
  }`
      })
    );

    const finishBindings: WGSLKernelBinding[] = [
      {name: 'accumulators', view: histogram.accumulators, type: 'u32', access: 'read'},
      {name: 'referenceAccumulators', view: referenceAccumulators, type: 'u32', access: 'read'},
      {name: 'cellOffsets', view: inputs.cellOffsets, type: 'u32', access: 'read'},
      {name: 'parameters', view: parameters, type: 'f32', access: 'read'}
    ];
    for (const [name, view] of [
      ['g', props.g],
      ['f', props.f],
      ['j', props.j],
      ['radii', props.radii]
    ] as const) {
      if (view) {
        finishBindings.push({name, view, type: 'f32', access: 'read_write'});
      }
    }
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-finish`,
        operation: OPERATION,
        variant: 'finish',
        bindings: finishBindings,
        invocationCount: radiusCount,
        declarations: /* wgsl */ `${PAIR_STATISTICS_FLOAT_WGSL}
${constantsWGSL}
const CELL_COUNT: u32 = ${gridSize[0] * gridSize[1]}u;
const REFERENCE_COUNT: f32 = ${referenceCount}.0;
const CHANNELS: u32 = ${CHANNEL_COUNT}u;

// Event counts stay below 2^24, so the low word of each 64-bit accumulator is exact.
fn readEvents(slot: u32, channel: u32) -> u32 {
  return accumulators[accumulatorsOffset + (slot * CHANNELS + channel) * 2u];
}

fn readReferences(slot: u32, channel: u32) -> u32 {
  return referenceAccumulators[referenceAccumulatorsOffset + (slot * CHANNELS + channel) * 2u];
}

// Full 64-bit sum as f32, for the fixed-point Hanisch weights.
fn readWeight(reference: bool, slot: u32, channel: u32) -> f32 {
  let word = (slot * CHANNELS + channel) * 2u;
  if (reference) {
    return f32(referenceAccumulators[referenceAccumulatorsOffset + word + 1u]) * 4294967296.0 +
      f32(referenceAccumulators[referenceAccumulatorsOffset + word]);
  }
  return f32(accumulators[accumulatorsOffset + word + 1u]) * 4294967296.0 +
    f32(accumulators[accumulatorsOffset + word]);
}

fn readObserved(reference: bool, slot: u32, channel: u32) -> f32 {
  return select(f32(readEvents(slot, channel)), f32(readReferences(slot, channel)), reference);
}

// Kaplan-Meier estimate of the distribution function at radius slot last, with hazards
// binned on the radius grid (events before censoring within a slot), as spatstat does.
fn getKaplanMeier(reference: bool, total: f32, last: u32) -> f32 {
  var survival = 1.0;
  var atRisk = total;
  for (var slot = 0u; slot <= last; slot++) {
    let observed = readObserved(reference, slot, 4u);
    if (atRisk >= 1.0 && observed > 0.0) {
      survival = survival * (1.0 - observed / atRisk);
    }
    atRisk = atRisk - observed - readObserved(reference, slot, 5u);
  }
  return 1.0 - survival;
}

// Hanisch estimate: weighted observed fraction up to last over the total observed weight.
fn getHanisch(reference: bool, last: u32) -> f32 {
  var numerator = 0.0;
  var denominator = readWeight(reference, 0u, 7u);
  for (var slot = 0u; slot < RADIUS_COUNT; slot++) {
    let weight = readWeight(reference, slot, 6u);
    denominator = denominator + weight;
    if (slot <= last) {
      numerator = numerator + weight;
    }
  }
  return select(getQuietNaN(last), numerator / denominator, denominator > 0.0);
}

fn getEventPrefix(channel: u32, last: u32) -> u32 {
  var sum = 0u;
  for (var slot = 0u; slot <= last; slot++) {
    sum += readEvents(slot, channel);
  }
  return sum;
}

fn getReferencePrefix(channel: u32, last: u32) -> u32 {
  var sum = 0u;
  for (var slot = 0u; slot <= last; slot++) {
    sum += readReferences(slot, channel);
  }
  return sum;
}`,
        body: `let nan = getQuietNaN(index);
  let n = f32(cellOffsets[cellOffsetsOffset + CELL_COUNT]);
  let mode = u32(parameters[parametersOffset + 5u] + 0.5);
  let border = mode == 1u;
  let maximumDistance = parameters[parametersOffset + 4u];
  let valid = isFiniteFloat(parameters[parametersOffset]) && isFiniteFloat(maximumDistance) && maximumDistance > 0.0;
  var gValue = nan;
  var fValue = nan;
  if (valid && n >= 2.0) {
    if (border) {
      let denominator = n - f32(getEventPrefix(3u, index));
      if (denominator >= 1.0) {
        gValue = f32(getEventPrefix(1u, index) - getEventPrefix(2u, index)) / denominator;
      }
    } else {
      gValue = f32(getEventPrefix(0u, index)) / n;
    }
    if (mode == 2u) {
      gValue = getKaplanMeier(false, n, index);
    } else if (mode == 3u) {
      gValue = getHanisch(false, index);
    }
  }
  if (valid && n >= 1.0) {
    if (border) {
      let denominator = REFERENCE_COUNT - f32(getReferencePrefix(3u, index));
      if (denominator >= 1.0) {
        fValue = f32(getReferencePrefix(1u, index) - getReferencePrefix(2u, index)) / denominator;
      }
    } else {
      fValue = f32(getReferencePrefix(0u, index)) / REFERENCE_COUNT;
    }
    if (mode == 2u) {
      fValue = getKaplanMeier(true, REFERENCE_COUNT, index);
    } else if (mode == 3u) {
      fValue = getHanisch(true, index);
    }
  }
  ${props.g ? 'g[gOffset + index] = gValue;' : ''}
  ${props.f ? 'f[fOffset + index] = fValue;' : ''}
  ${
    props.j
      ? `let survival = 1.0 - fValue;
  j[jOffset + index] = select(nan, (1.0 - gValue) / survival, isFiniteFloat(gValue) && isFiniteFloat(fValue) && survival > 0.0);`
      : ''
  }
  ${props.radii ? 'radii[radiiOffset + index] = maximumDistance * f32(index + 1u) / f32(RADIUS_COUNT);' : ''}`
      })
    );
    return nodes;
  }
}
