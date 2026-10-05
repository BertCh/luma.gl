// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  validatePackedUint32View,
  validatePackedView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {createWGSLKernelNode, type WGSLKernelBinding} from '../../utils/wgsl-kernel-nodes';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../utils/gpu-contributor-utils';
import {getPairHistogramNodes, validatePairHistogramShape} from './pair-histogram';
import {
  getPairStatisticsInputNodes,
  PAIR_STATISTICS_FLOAT_WGSL,
  validatePairStatisticsInputs
} from './pair-statistics-grid';
import {GPU_RIPLEY_PARAMETER_LENGTH} from './ripley-parameters';

const OPERATION = 'GPURipley';

/** Largest `radiusCount`. */
const MAXIMUM_RADIUS_COUNT = 256;

/** Accumulator channels per radius: border plus, border minus, border focus, raw pairs, isotropic. */
const CHANNEL_COUNT = 5;

/** Fixed-point scale of one isotropic weight, `2^17`: weights up to the cap stay below `2^24`. */
const ISOTROPIC_SCALE = 131072;

/** Largest isotropic edge-correction weight (smallest circle fraction is `1 / cap`). */
export const GPU_RIPLEY_WEIGHT_CAP = 100;

/**
 * Properties for {@link GPURipley}.
 *
 * Per-frame (no rebuild or recompile): the contents of `positions`, `mask` and `parameters`
 * (window bounds, maximum distance, edge-correction mode). Compile-time: the row count,
 * `gridSize`, `radiusCount` and which optional outputs are present.
 */
export type GPURipleyProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'ripley'`. */
  id?: string;
  /** Packed planar points, one row per event. At least one and fewer than 2^24 rows. */
  positions: GraphDataView<'float32x2'>;
  /**
   * Per-frame parameters: packed float32 view of at least `GPU_RIPLEY_PARAMETER_LENGTH` elements
   * written with `getGPURipleyParameterValues`. Invalid bounds or a non-positive maximum distance
   * exclude every row.
   */
  parameters: GraphDataView<'float32'>;
  /**
   * Maximum `[columns, rows]` of the neighbor-search cell lattice. Compile-time. Results never
   * depend on it, only speed.
   */
  gridSize: readonly [number, number];
  /**
   * Number of radii. Compile-time, an integer in `[1, 256]`. Radius `b` is
   * `maximumDistance * (b + 1) / radiusCount`.
   */
  radiusCount: number;
  /** Optional row selection: nonzero includes the row. */
  mask?: GraphDataView<'uint32'>;
  /** Caller-owned Ripley's K per radius, `radiusCount` rows. Quiet NaN where undefined. */
  k: GraphDataView<'float32'>;
  /** Optional caller-owned Besag's L, `sqrt(K / pi)`, per radius. */
  l?: GraphDataView<'float32'>;
  /** Optional caller-owned `L(r) - r` per radius (zero under complete spatial randomness). */
  lMinusR?: GraphDataView<'float32'>;
  /**
   * Optional caller-owned pair-correlation estimate per annulus,
   * `(K(r_b) - K(r_b-1)) / (pi (r_b^2 - r_b-1^2))` with `r_-1 = 0` and `K(0) = 0`.
   */
  pairCorrelation?: GraphDataView<'float32'>;
  /**
   * Optional caller-owned uncorrected count of ordered pairs `(i, j)` whose distance falls in
   * annulus `b`, `radiusCount` rows. Saturates at `2^32 - 1`.
   */
  pairCounts?: GraphDataView<'uint32'>;
  /** Optional caller-owned radii, `radiusCount` rows. */
  radii?: GraphDataView<'float32'>;
};

/**
 * Ripley's K, Besag's L and the annulus pair-correlation function of a planar point pattern in a
 * rectangular window (spatstat `Kest`, `Lest`, `pcf` for rectangle windows), from one
 * deterministic pass over every ordered pair within the per-frame maximum distance.
 *
 * Definition, which the GPU result matches within the bounds below:
 * - The window is the per-frame bounds rectangle with area `A`. Included rows have a nonzero mask
 *   (when given) and a finite position inside the window. `n` is the included row count.
 * - Each ordered pair `(i, j)`, `i != j`, with `d_ij <= maximumDistance` (f32 `sqrt(dx^2 + dy^2)`)
 *   falls in annulus `a = clamp(ceil(d / maximumDistance * radiusCount) - 1, 0, radiusCount - 1)`,
 *   so it counts toward `K(r_b)` exactly for `b >= a`.
 * - `'none'`: `K(r) = A / (n (n - 1)) * #{(i, j): d_ij <= r}`.
 * - `'border'` (reduced sample, spatstat `Kborder`): with `b_i` the distance from `i` to the
 *   window boundary, `K(r) = sum_i 1{b_i >= r} #{j != i: d_ij <= r} / (lambda * sum_i 1{b_i >= r})`
 *   with `lambda = (n - 1) / A`. This is the conditional-on-n density that makes the estimator
 *   equal to `'none'` when no focus point is near the boundary. spatstat's `Kest` takes
 *   `lambda = n / A` for the border estimator as far as I recall, which differs by the constant
 *   factor `(n - 1) / n`. `K(r)` is NaN when no point is farther than `r` from the boundary.
 *   Implemented with per-focus difference arrays over radius indices in integer channels, with
 *   the prefix sums in the finishing kernel.
 * - `'isotropic'` (Ripley 1977): each pair is weighted `e_ij = 1 / f` where `f` is the fraction of
 *   the circle centered at `i` with radius `d_ij` that lies inside the window, computed exactly
 *   for a rectangle from the arcs cut off by each side, minus the overlap of arcs of adjacent sides
 *   (the corner is inside the circle). `f` is floored at `1 / GPU_RIPLEY_WEIGHT_CAP` (cap 100,
 *   which I believe matches spatstat's `edge.Ripley` guard but have not verified), so
 *   `e_ij <= 100`. `K(r) = A / (n (n - 1)) * sum_{d_ij <= r} e_ij`. Use `maximumDistance` of at
 *   most half the shorter window side for this estimator to be meaningful.
 * - `L(r) = sqrt(K(r) / pi)`, `L(r) - r`, and the pair correlation per annulus from differences
 *   of `K` (a noisy, unsmoothed estimator; spatstat's `pcf` smooths with a kernel).
 * - Undefined results (`n < 2`, an empty border set, zero window area) are quiet NaN.
 *
 * Determinism and precision: all pair counts are exact 64-bit integer sums (so `pairCounts` is
 * exact, and `'none'` and `'border'` give `K` exactly up to f32 rounding of the final division).
 * Isotropic weights are accumulated as `round(e * 2^17)`, an error of at most `2^-18` per pair
 * plus the f32 error of the weight itself. Results are bitwise reproducible.
 *
 * Monte Carlo CSR envelopes and cross-type K are not implemented.
 */
export class GPURipley implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPURipleyProps;

  constructor(props: GPURipleyProps) {
    this.id = props.id ?? 'ripley';
    this.props = props;
    const id = this.id;
    validatePairStatisticsInputs(id, {...props, parameterLength: GPU_RIPLEY_PARAMETER_LENGTH});
    const {radiusCount} = props;
    if (!Number.isInteger(radiusCount) || radiusCount < 1 || radiusCount > MAXIMUM_RADIUS_COUNT) {
      throw new Error(`${id} radiusCount must be an integer in [1, ${MAXIMUM_RADIUS_COUNT}]`);
    }
    validatePairHistogramShape(id, radiusCount, CHANNEL_COUNT);
    for (const [name, view] of [
      ['k', props.k],
      ['l', props.l],
      ['lMinusR', props.lMinusR],
      ['pairCorrelation', props.pairCorrelation],
      ['radii', props.radii]
    ] as const) {
      if (view) {
        validatePackedView(view, ['float32'], `${id} ${name}`);
        if (view.length < radiusCount) {
          throw new Error(`${id} ${name} must hold radiusCount rows`);
        }
      }
    }
    if (props.pairCounts) {
      validatePackedUint32View(props.pairCounts, `${id} pairCounts`);
      if (props.pairCounts.length < radiusCount) {
        throw new Error(`${id} pairCounts must hold radiusCount rows`);
      }
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      [props.k, props.l, props.lMinusR, props.pairCorrelation, props.pairCounts, props.radii],
      [props.positions, props.parameters, props.mask]
    );
  }

  /** Returns the Ripley nodes in dependency order. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {positions, parameters, gridSize, radiusCount} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      positions,
      parameters,
      props.mask,
      props.k,
      props.l,
      props.lMinusR,
      props.pairCorrelation,
      props.pairCounts,
      props.radii
    ]);
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
const ISOTROPIC_SCALE: f32 = ${ISOTROPIC_SCALE}.0;
const WEIGHT_CAP: f32 = ${GPU_RIPLEY_WEIGHT_CAP}.0;
const PI: f32 = 3.14159265358979;
const HALF_PI: f32 = 1.57079632679490;
`;
    const histogram = getPairHistogramNodes<Parameters>(graph, {
      id: `${id}-histogram`,
      operation: OPERATION,
      positions,
      parameters,
      gridSize,
      sortedRows: inputs.sortedRows,
      cellOffsets: inputs.cellOffsets,
      slotCount: radiusCount,
      channelCount: CHANNEL_COUNT,
      pairOrder: 'ordered',
      declarations: /* wgsl */ `${constantsWGSL}
// Angle of the arc of a circle of radius d (centered at distance h from a side line) outside it.
fn getSideAngle(h: f32, d: f32) -> f32 {
  if (d > h) {
    return atan2(sqrt((d - h) * (d + h)), h);
  }
  return 0.0;
}

fn getArcOverlap(first: f32, second: f32) -> f32 {
  return max(0.0, first + second - HALF_PI);
}

// Ripley (1977) weight 1 / (fraction of the circle inside the rectangle), capped.
fn getIsotropicWeight(left: f32, bottom: f32, right: f32, top: f32, d: f32) -> f32 {
  let a = getSideAngle(left, d);
  let b = getSideAngle(bottom, d);
  let c = getSideAngle(right, d);
  let e = getSideAngle(top, d);
  let overlap = getArcOverlap(a, b) + getArcOverlap(b, c) + getArcOverlap(c, e) + getArcOverlap(e, a);
  let inside = 1.0 - (a + b + c + e) / PI + overlap / (2.0 * PI);
  return min(1.0 / max(inside, 1.0 / WEIGHT_CAP), WEIGHT_CAP);
}`,
      focusPrologue: `let mode = u32(readParameter(5u) + 0.5);
    let left = x - lattice.minimumX;
    let bottom = y - lattice.minimumY;
    let right = lattice.maximumX - x;
    let top = lattice.maximumY - y;
    var borderCount = 0u;
    if (mode == 1u) {
      let borderDistance = min(min(left, right), min(bottom, top));
      borderCount = min(u32(floor(borderDistance / lattice.maximumDistance * f32(RADIUS_COUNT))), RADIUS_COUNT);
    }`,
      pairAction: `let scaled = pairDistance / lattice.maximumDistance * f32(RADIUS_COUNT);
            let annulus = min(u32(max(ceil(scaled) - 1.0, 0.0)), RADIUS_COUNT - 1u);
            accumulate(annulus, 3u, 1u);
            if (mode == 1u) {
              if (annulus < borderCount) {
                accumulate(annulus, 0u, 1u);
                accumulate(borderCount, 1u, 1u);
              }
            } else if (mode == 2u) {
              accumulate(annulus, 4u, quantizePairAmount(getIsotropicWeight(left, bottom, right, top, pairDistance) * ISOTROPIC_SCALE));
            }`,
      focusEpilogue: `if (mode == 1u) {
      accumulate(borderCount, 2u, 1u);
    }`
    });
    nodes.push(...histogram.nodes);

    const finishBindings: WGSLKernelBinding[] = [
      {name: 'accumulators', view: histogram.accumulators, type: 'u32', access: 'read'},
      {name: 'cellOffsets', view: inputs.cellOffsets, type: 'u32', access: 'read'},
      {name: 'parameters', view: parameters, type: 'f32', access: 'read'},
      {name: 'k', view: props.k, type: 'f32', access: 'read_write'}
    ];
    if (props.pairCounts) {
      finishBindings.push({
        name: 'pairCounts',
        view: props.pairCounts,
        type: 'u32',
        access: 'read_write'
      });
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

fn readAccumulator(slot: u32, channel: u32) -> vec2<u32> {
  let word = (slot * ${CHANNEL_COUNT}u + channel) * 2u;
  return vec2<u32>(accumulators[accumulatorsOffset + word], accumulators[accumulatorsOffset + word + 1u]);
}

fn add64(first: vec2<u32>, second: vec2<u32>) -> vec2<u32> {
  let low = first.x + second.x;
  return vec2<u32>(low, first.y + second.y + select(0u, 1u, low < first.x));
}

fn subtract64(first: vec2<u32>, second: vec2<u32>) -> vec2<u32> {
  return vec2<u32>(first.x - second.x, first.y - second.y - select(0u, 1u, first.x < second.x));
}

fn toFloat64(value: vec2<u32>) -> f32 {
  return f32(value.y) * 4294967296.0 + f32(value.x);
}

// Exact 64-bit prefix sum of one channel over slots [0, last].
fn getPrefix(channel: u32, last: u32) -> vec2<u32> {
  var sum = vec2<u32>(0u, 0u);
  for (var slot = 0u; slot <= last; slot++) {
    sum = add64(sum, readAccumulator(slot, channel));
  }
  return sum;
}`,
        body: `let n = f32(cellOffsets[cellOffsetsOffset + CELL_COUNT]);
  let area = (parameters[parametersOffset + 2u] - parameters[parametersOffset]) *
    (parameters[parametersOffset + 3u] - parameters[parametersOffset + 1u]);
  let mode = u32(parameters[parametersOffset + 5u] + 0.5);
  var kValue = getQuietNaN(index);
  if (n >= 2.0 && isFiniteFloat(area) && area > 0.0) {
    if (mode == 1u) {
      let numerator = subtract64(getPrefix(0u, index), getPrefix(1u, index));
      let borderCount = n - toFloat64(getPrefix(2u, index));
      if (borderCount >= 1.0) {
        kValue = area * toFloat64(numerator) / ((n - 1.0) * borderCount);
      }
    } else if (mode == 2u) {
      kValue = area / (n * (n - 1.0)) * (toFloat64(getPrefix(4u, index)) / ISOTROPIC_SCALE);
    } else {
      kValue = area / (n * (n - 1.0)) * toFloat64(getPrefix(3u, index));
    }
  }
  k[kOffset + index] = kValue;
  ${
    props.pairCounts
      ? `let annulusCount = readAccumulator(index, 3u);
  pairCounts[pairCountsOffset + index] = select(annulusCount.x, 0xffffffffu, annulusCount.y != 0u);`
      : ''
  }`
      })
    );

    const deriveBindings: WGSLKernelBinding[] = [
      {name: 'parameters', view: parameters, type: 'f32', access: 'read'},
      {name: 'k', view: props.k, type: 'f32', access: 'read'}
    ];
    for (const [name, view] of [
      ['l', props.l],
      ['lMinusR', props.lMinusR],
      ['pairCorrelation', props.pairCorrelation],
      ['radii', props.radii]
    ] as const) {
      if (view) {
        deriveBindings.push({name, view, type: 'f32', access: 'read_write'});
      }
    }
    if (deriveBindings.length > 2) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-derive`,
          operation: OPERATION,
          variant: 'derive',
          bindings: deriveBindings,
          invocationCount: radiusCount,
          declarations: /* wgsl */ `${PAIR_STATISTICS_FLOAT_WGSL}
${constantsWGSL}`,
          body: `let nan = getQuietNaN(index);
  let maximumDistance = parameters[parametersOffset + 4u];
  let radius = maximumDistance * f32(index + 1u) / f32(RADIUS_COUNT);
  let kValue = k[kOffset + index];
  let finite = isFiniteFloat(kValue);
  let lValue = select(nan, sqrt(max(kValue, 0.0) / PI), finite);
  ${props.l ? 'l[lOffset + index] = lValue;' : ''}
  ${props.lMinusR ? 'lMinusR[lMinusROffset + index] = select(nan, lValue - radius, finite);' : ''}
  ${props.radii ? 'radii[radiiOffset + index] = radius;' : ''}
  ${
    props.pairCorrelation
      ? `var previousK = 0.0;
  var previousFinite = true;
  if (index > 0u) {
    previousK = k[kOffset + index - 1u];
    previousFinite = isFiniteFloat(previousK);
  }
  let previousRadius = maximumDistance * f32(index) / f32(RADIUS_COUNT);
  let annulusArea = PI * (radius - previousRadius) * (radius + previousRadius);
  pairCorrelation[pairCorrelationOffset + index] =
    select(nan, (kValue - previousK) / annulusArea, finite && previousFinite);`
      : ''
  }`
        })
      );
    }
    return nodes;
  }
}
