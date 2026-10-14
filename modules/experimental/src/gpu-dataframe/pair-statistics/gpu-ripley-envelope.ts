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
  resolveGPUSpatialContext,
  type GPUSpatialContext
} from '../../gpu-spatial-analysis/contracts/spatial-context';
import {PHILOX_WGSL} from '../../utils/philox-random';
import {createFillNode, createWGSLKernelNode} from '../../utils/wgsl-kernel-nodes';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../utils/gpu-contributor-utils';
import {GPURipley} from './gpu-ripley';
import {
  getPairStatisticsSharedWGSL,
  PAIR_STATISTICS_FLOAT_WGSL,
  validatePairStatisticsInputs
} from './pair-statistics-grid';
import {
  GPU_RIPLEY_ENVELOPE_STATISTICS_LENGTH,
  GPU_RIPLEY_PARAMETER_LENGTH,
  GPU_RIPLEY_SEED_SLOT
} from './ripley-parameters';

const OPERATION = 'GPURipleyEnvelope';

/** Largest `radiusCount`. */
const MAXIMUM_RADIUS_COUNT = 256;

/**
 * Largest `simulationCount`. Every simulation adds one complete `GPURipley` pass (about 25 nodes:
 * cell index, sort, histogram, finish), so the graph holds roughly `25 * simulationCount` nodes.
 */
export const GPU_RIPLEY_ENVELOPE_MAXIMUM_SIMULATION_COUNT = 199;

/** Second Philox key word, the ASCII bytes `RIPF`, so seeds do not collide with other streams. */
const PHILOX_KEY_TAG = 0x52495046;

/** Scale of the statistic an envelope is reported in. */
export type GPURipleyEnvelopeStatistic = 'k' | 'l' | 'l-minus-r';

const STATISTIC_CODES: Record<GPURipleyEnvelopeStatistic, number> = {
  k: 0,
  l: 1,
  'l-minus-r': 2
};

/**
 * Properties for {@link GPURipleyEnvelope}.
 *
 * Per-frame (no rebuild or recompile): the contents of `positions`, `mask`, `parameters` (window
 * bounds, maximum distance, edge correction and the Monte Carlo `seed`) and `observedK`.
 * Compile-time: the row count, `gridSize`, `radiusCount`, `simulationCount`, `rank`, `statistic`
 * and which outputs are present.
 */
export type GPURipleyEnvelopeProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'ripley-envelope'`. */
  id?: string;
  /**
   * Coordinate and metric descriptor of `positions`. Omitted means native planar coordinates; a
   * `longitude-latitude` context throws (project to a planar frame first).
   */
  spatialContext?: GPUSpatialContext;
  /**
   * The observed packed planar points (the same view given to the observed {@link GPURipley}).
   * The row count is the capacity of every simulated pattern.
   */
  positions: GraphDataView<'float32x2'>;
  /**
   * Per-frame parameters: the same 8-element layout and view as the observed `GPURipley`, written
   * with `getGPURipleyParameterValues` including its `seed`. Window, maximum distance and edge
   * correction apply to every simulation exactly as to the observed pattern.
   */
  parameters: GraphDataView<'float32'>;
  /** Maximum `[columns, rows]` of the neighbor-search lattice. Compile-time; speed only. */
  gridSize: readonly [number, number];
  /** Number of radii; must equal the observed `GPURipley` `radiusCount`. An integer in `[1, 256]`. */
  radiusCount: number;
  /**
   * Number of CSR simulations `s`, an integer in `[2 * rank, 199]`. Compile-time. The Monte Carlo
   * significance level of the pointwise and global envelope is `2 * rank / (s + 1)` and
   * `rank / (s + 1)` respectively (Baddeley et al. 2014); 99 simulations with `rank = 1` give a
   * 0.01 global test, 39 give 0.025.
   */
  simulationCount: number;
  /**
   * Which sorted simulated value bounds the pointwise envelope and, for the global envelope, which
   * ranked maximum deviation sets the half-width. Compile-time integer at least 1, default 1.
   */
  rank?: number;
  /** Scale of the reported envelopes. Default `'l-minus-r'` (zero under CSR). */
  statistic?: GPURipleyEnvelopeStatistic;
  /** Optional row selection of the observed pattern: nonzero includes the row. */
  mask?: GraphDataView<'uint32'>;
  /**
   * Optional observed Ripley's K per radius (the `k` output of the observed `GPURipley` run with
   * the same parameters). It enables the maximum-deviation test statistic and Monte Carlo p-value.
   */
  observedK?: GraphDataView<'float32'>;
  /** Optional caller-owned pointwise lower envelope: the `rank`-th smallest simulated value. */
  lower?: GraphDataView<'float32'>;
  /** Optional caller-owned pointwise upper envelope: the `rank`-th largest simulated value. */
  upper?: GraphDataView<'float32'>;
  /** Optional caller-owned global (simultaneous) lower envelope, `theoretical - criticalDeviation`. */
  globalLower?: GraphDataView<'float32'>;
  /** Optional caller-owned global (simultaneous) upper envelope, `theoretical + criticalDeviation`. */
  globalUpper?: GraphDataView<'float32'>;
  /** Optional caller-owned theoretical CSR value per radius on the chosen scale. */
  theoretical?: GraphDataView<'float32'>;
  /**
   * Optional caller-owned summary `[n, validSimulationCount, observedMaximumDeviation,
   * criticalMaximumDeviation, pValue, rank]` (`GPU_RIPLEY_ENVELOPE_STATISTICS_LENGTH` rows).
   */
  statistics?: GraphDataView<'float32'>;
};

/**
 * Monte Carlo complete-spatial-randomness (CSR) envelopes of Ripley's K, Besag's L or `L(r) - r`
 * for a planar point pattern in a rectangular window (spatstat `envelope(Kest, nsim, nrank)` and
 * the global envelope of Ripley 1981 / Baddeley et al. 2014), entirely on the GPU.
 *
 * Definition:
 * - `n` is the observed included row count (nonzero mask, finite position inside the window, the
 *   `GPURipley` inclusion rule), counted on the GPU. Each of the `simulationCount` simulations is a
 *   binomial CSR pattern: `n` independent uniform points in the window (conditional on `n`), drawn
 *   with Philox4x32-10 keyed by the per-frame `seed` with counter `(row, simulationIndex, 0, 1)`;
 *   word `x` gives the x coordinate and `y` the y coordinate. Draws are a pure function of
 *   `(seed, simulationIndex, row)` and reproduce exactly (CPU mirror: `getPhilox4x32`).
 * - `K_j(r)` of each simulation is computed by a full {@link GPURipley} pass over the simulated
 *   pattern with the same parameters, so edge correction, radii and NaN rules are identical to
 *   the observed estimate. There is no per-simulation CPU readback.
 * - Pointwise envelope (spatstat `envelope(..., global = FALSE)`): at each radius the `rank`-th
 *   smallest and `rank`-th largest finite simulated value, transformed to `statistic`. It is NOT a
 *   simultaneous band: its false positive rate over all radii exceeds `2 * rank / (s + 1)`.
 * - Global envelope (Ripley 1981; Baddeley et al. 2014, Ecological Monographs 84(3), "On tests of
 *   spatial pattern based on simulation envelopes"): `T_j = max_r |L_j(r) - r|` over the radii
 *   where `L_j` is finite, with the theoretical CSR value `L(r) = r`. The critical deviation `t*`
 *   is the `rank`-th largest of the `T_j`; the band is `L(r) = r +- t*` (a constant-width band
 *   around theory, transformed to `statistic`). An observed curve leaves it exactly when its
 *   `T_obs > t*`, a simultaneous Monte Carlo test of level `rank / (s + 1)`.
 * - With `observedK`, `T_obs = max_r |sqrt(K_obs(r) / pi) - r|` and the Monte Carlo p-value is
 *   `(1 + #{j: T_j >= T_obs}) / (s_valid + 1)`, the maximum absolute deviation (MAD) test. Radii
 *   with NaN are skipped per curve, so use an edge correction that is defined at every radius
 *   (`'isotropic'` or `'none'`; `'border'` is NaN once no point is r from the boundary).
 * - The window is the bounds rectangle: a bounding box of an irregular study area makes every
 *   pattern look clustered. The theoretical value assumes the plug-in edge-corrected K is unbiased
 *   under CSR, true for `'isotropic'` and approximately for `'border'`; `'none'` is biased low at
 *   large radii but the simulations share the bias, so the pointwise envelope remains valid.
 * - Undefined results (`n < 2`, fewer than `rank` finite simulated values at a radius) are NaN.
 *
 * Cost: each simulation is one `GPURipley` pass, so total work is `simulationCount + 1` times the
 * observed K. Memory is the transient buffers of one pass (the graph aliases buffers whose
 * lifetimes are disjoint) plus `simulationCount * radiusCount` floats. `simulationCount` is capped
 * at {@link GPU_RIPLEY_ENVELOPE_MAXIMUM_SIMULATION_COUNT} to keep the node graph bounded.
 */
export class GPURipleyEnvelope implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPURipleyEnvelopeProps;
  /** Resolved spatial context (native planar unless the props say otherwise). */
  readonly spatialContext: GPUSpatialContext;
  /** Resolved `rank`. */
  readonly rank: number;
  /** Resolved statistic scale. */
  readonly statistic: GPURipleyEnvelopeStatistic;

  constructor(props: GPURipleyEnvelopeProps) {
    this.id = props.id ?? 'ripley-envelope';
    this.props = props;
    const id = this.id;
    this.spatialContext = resolveGPUSpatialContext(id, props.spatialContext);
    validatePairStatisticsInputs(id, {...props, parameterLength: GPU_RIPLEY_PARAMETER_LENGTH});
    const {radiusCount, simulationCount} = props;
    if (!Number.isInteger(radiusCount) || radiusCount < 1 || radiusCount > MAXIMUM_RADIUS_COUNT) {
      throw new Error(`${id} radiusCount must be an integer in [1, ${MAXIMUM_RADIUS_COUNT}]`);
    }
    this.rank = props.rank ?? 1;
    if (!Number.isInteger(this.rank) || this.rank < 1) {
      throw new Error(`${id} rank must be a positive integer`);
    }
    if (
      !Number.isInteger(simulationCount) ||
      simulationCount < 2 * this.rank ||
      simulationCount > GPU_RIPLEY_ENVELOPE_MAXIMUM_SIMULATION_COUNT
    ) {
      throw new Error(
        `${id} simulationCount must be an integer in [2 * rank, ${GPU_RIPLEY_ENVELOPE_MAXIMUM_SIMULATION_COUNT}]`
      );
    }
    this.statistic = props.statistic ?? 'l-minus-r';
    if (!Object.prototype.hasOwnProperty.call(STATISTIC_CODES, this.statistic)) {
      throw new Error(`${id} statistic must be 'k', 'l' or 'l-minus-r'`);
    }
    const radiusOutputs = [
      ['observedK', props.observedK],
      ['lower', props.lower],
      ['upper', props.upper],
      ['globalLower', props.globalLower],
      ['globalUpper', props.globalUpper],
      ['theoretical', props.theoretical]
    ] as const;
    for (const [name, view] of radiusOutputs) {
      if (view) {
        validatePackedView(view, ['float32'], `${id} ${name}`);
        if (view.length < radiusCount) {
          throw new Error(`${id} ${name} must hold radiusCount rows`);
        }
      }
    }
    if (props.statistics) {
      validatePackedView(props.statistics, ['float32'], `${id} statistics`);
      if (props.statistics.length < GPU_RIPLEY_ENVELOPE_STATISTICS_LENGTH) {
        throw new Error(`${id} statistics must hold ${GPU_RIPLEY_ENVELOPE_STATISTICS_LENGTH} rows`);
      }
    }
    if (
      !props.lower &&
      !props.upper &&
      !props.globalLower &&
      !props.globalUpper &&
      !props.theoretical &&
      !props.statistics
    ) {
      throw new Error(`${id} needs at least one output`);
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      [
        props.lower,
        props.upper,
        props.globalLower,
        props.globalUpper,
        props.theoretical,
        props.statistics
      ],
      [props.positions, props.parameters, props.mask, props.observedK]
    );
  }

  /** Returns the envelope nodes in dependency order. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, rank, statistic} = this;
    const {positions, parameters, gridSize, radiusCount, simulationCount} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      positions,
      parameters,
      props.mask,
      props.observedK,
      props.lower,
      props.upper,
      props.globalLower,
      props.globalUpper,
      props.theoretical,
      props.statistics
    ]);
    const rows = positions.length;
    const nodes: GPUCommandNode<Parameters>[] = [];

    // Observed included row count n, with the exact GPURipley inclusion rule.
    const count = createTransientView(graph, `${id}-count`, 'uint32', 1);
    nodes.push(
      createFillNode<Parameters>(graph, {
        id: `${id}-count-clear`,
        operation: OPERATION,
        view: count,
        type: 'u32',
        value: '0u'
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-count`,
        operation: OPERATION,
        variant: 'count',
        bindings: [
          {name: 'positions', view: positions, type: 'f32', access: 'read'},
          {name: 'parameters', view: parameters, type: 'f32', access: 'read'},
          ...(props.mask
            ? [{name: 'mask', view: props.mask, type: 'u32' as const, access: 'read' as const}]
            : []),
          {name: 'count', view: count, type: 'atomic<u32>', access: 'read_write'}
        ],
        invocationCount: rows,
        declarations: getPairStatisticsSharedWGSL(gridSize),
        body: `let lattice = readLattice();
  let x = positions[positionsOffset + index * 2u];
  let y = positions[positionsOffset + index * 2u + 1u];
  let included = ${props.mask ? 'mask[maskOffset + index] != 0u' : 'true'};
  if (included && isPointValid(lattice, x, y)) {
    atomicAdd(&count[countOffset], 1u);
  }`
      })
    );

    // Every simulation's K goes into one row of allK, copied by a kernel whose row is a pipeline
    // override, so all simulations share one shader module (a view offset would be baked into the
    // source and compile one module per simulation).
    const allK = createTransientView(
      graph,
      `${id}-all-k`,
      'float32',
      simulationCount * radiusCount
    );
    for (let simulation = 0; simulation < simulationCount; simulation++) {
      const simulatedPositions = createTransientView(
        graph,
        `${id}-sim-${simulation}-positions`,
        'float32x2',
        rows
      );
      const simulatedMask = createTransientView(
        graph,
        `${id}-sim-${simulation}-mask`,
        'uint32',
        rows
      );
      const simulatedK = createTransientView(
        graph,
        `${id}-sim-${simulation}-k`,
        'float32',
        radiusCount
      );
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-sim-${simulation}-points`,
          operation: OPERATION,
          variant: 'simulate',
          bindings: [
            {name: 'parameters', view: parameters, type: 'f32', access: 'read'},
            {name: 'count', view: count, type: 'u32', access: 'read'},
            {name: 'positions', view: simulatedPositions, type: 'f32', access: 'read_write'},
            {name: 'mask', view: simulatedMask, type: 'u32', access: 'read_write'}
          ],
          invocationCount: rows,
          constants: {SIMULATION_INDEX: simulation},
          declarations: /* wgsl */ `${PHILOX_WGSL}
override SIMULATION_INDEX: u32;
const SEED_SLOT: u32 = ${GPU_RIPLEY_SEED_SLOT}u;
const KEY_TAG: u32 = ${PHILOX_KEY_TAG}u;`,
          body: `let inside = index < count[countOffset];
  var x = 0.0;
  var y = 0.0;
  if (inside) {
    let words = philox4x32(
      vec4<u32>(index, SIMULATION_INDEX, 0u, 1u),
      vec2<u32>(u32(parameters[parametersOffset + SEED_SLOT]), KEY_TAG)
    );
    let minimumX = parameters[parametersOffset];
    let minimumY = parameters[parametersOffset + 1u];
    x = minimumX + philoxUnitFloat(words.x) * (parameters[parametersOffset + 2u] - minimumX);
    y = minimumY + philoxUnitFloat(words.y) * (parameters[parametersOffset + 3u] - minimumY);
  }
  positions[positionsOffset + index * 2u] = x;
  positions[positionsOffset + index * 2u + 1u] = y;
  mask[maskOffset + index] = select(0u, 1u, inside);`
        }),
        ...new GPURipley({
          id: `${id}-sim-${simulation}`,
          positions: simulatedPositions,
          parameters,
          gridSize,
          radiusCount,
          mask: simulatedMask,
          k: simulatedK
        }).getCommandNodes(graph),
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-sim-${simulation}-gather`,
          operation: OPERATION,
          variant: 'gather',
          bindings: [
            {name: 'simulatedK', view: simulatedK, type: 'f32', access: 'read'},
            {name: 'allK', view: allK, type: 'f32', access: 'read_write'}
          ],
          invocationCount: radiusCount,
          constants: {SIMULATION_INDEX: simulation},
          declarations: `override SIMULATION_INDEX: u32;
const RADIUS_COUNT: u32 = ${radiusCount}u;`,
          body: `allK[allKOffset + SIMULATION_INDEX * RADIUS_COUNT + index] = simulatedK[simulatedKOffset + index];`
        })
      );
    }

    const constantsWGSL = /* wgsl */ `${PAIR_STATISTICS_FLOAT_WGSL}
const RADIUS_COUNT: u32 = ${radiusCount}u;
const SIMULATION_COUNT: u32 = ${simulationCount}u;
const RANK: u32 = ${rank}u;
const STATISTIC: u32 = ${STATISTIC_CODES[statistic]}u;
const PI: f32 = 3.14159265358979;

fn getRadius(radiusIndex: u32) -> f32 {
  return parameters[parametersOffset + 4u] * f32(radiusIndex + 1u) / f32(RADIUS_COUNT);
}

fn getLDeviation(kValue: f32, radius: f32) -> f32 {
  return abs(sqrt(max(kValue, 0.0) / PI) - radius);
}

// Converts K to the reported statistic.
fn fromK(kValue: f32, radius: f32) -> f32 {
  if (STATISTIC == 0u) { return kValue; }
  let lValue = sqrt(max(kValue, 0.0) / PI);
  return select(lValue, lValue - radius, STATISTIC == 2u);
}

// Converts L to the reported statistic.
fn fromL(lValue: f32, radius: f32) -> f32 {
  if (STATISTIC == 0u) { return PI * max(lValue, 0.0) * max(lValue, 0.0); }
  return select(lValue, lValue - radius, STATISTIC == 2u);
}`;

    // T_j = max_r |L_j(r) - r| of every simulation (NaN when it has no finite radius).
    const deviations = createTransientView(graph, `${id}-deviations`, 'float32', simulationCount);
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-deviations`,
        operation: OPERATION,
        variant: 'deviations',
        bindings: [
          {name: 'parameters', view: parameters, type: 'f32', access: 'read'},
          {name: 'allK', view: allK, type: 'f32', access: 'read'},
          {name: 'deviations', view: deviations, type: 'f32', access: 'read_write'}
        ],
        invocationCount: simulationCount,
        declarations: constantsWGSL,
        body: `var deviation = 0.0;
  var anyFinite = false;
  for (var radiusIndex = 0u; radiusIndex < RADIUS_COUNT; radiusIndex++) {
    let kValue = allK[allKOffset + index * RADIUS_COUNT + radiusIndex];
    if (isFiniteFloat(kValue)) {
      deviation = max(deviation, getLDeviation(kValue, getRadius(radiusIndex)));
      anyFinite = true;
    }
  }
  deviations[deviationsOffset + index] = select(getQuietNaN(index), deviation, anyFinite);`
      })
    );

    // One thread: critical deviation (rank-th largest T_j), observed T and Monte Carlo p-value.
    const summary = props.statistics ?? createTransientView(graph, `${id}-summary`, 'float32', 6);
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-summary`,
        operation: OPERATION,
        variant: 'summary',
        bindings: [
          {name: 'parameters', view: parameters, type: 'f32', access: 'read'},
          {name: 'deviations', view: deviations, type: 'f32', access: 'read'},
          {name: 'count', view: count, type: 'u32', access: 'read'},
          ...(props.observedK
            ? [
                {
                  name: 'observedK',
                  view: props.observedK,
                  type: 'f32' as const,
                  access: 'read' as const
                }
              ]
            : []),
          {name: 'summary', view: summary, type: 'f32', access: 'read_write'}
        ],
        invocationCount: 1,
        declarations: constantsWGSL,
        body: `let nan = getQuietNaN(index);
  let n = count[countOffset];
  var validCount = 0u;
  for (var simulation = 0u; simulation < SIMULATION_COUNT; simulation++) {
    if (isFiniteFloat(deviations[deviationsOffset + simulation])) { validCount++; }
  }
  // rank-th largest by (value, index): the total order breaks ties without double counting.
  var critical = nan;
  for (var candidate = 0u; candidate < SIMULATION_COUNT; candidate++) {
    let value = deviations[deviationsOffset + candidate];
    if (!isFiniteFloat(value)) { continue; }
    var greater = 0u;
    for (var other = 0u; other < SIMULATION_COUNT; other++) {
      let otherValue = deviations[deviationsOffset + other];
      if (isFiniteFloat(otherValue) && (otherValue > value || (otherValue == value && other < candidate))) {
        greater++;
      }
    }
    if (greater == RANK - 1u) { critical = value; }
  }
  var observed = nan;
  var pValue = nan;
  ${
    props.observedK
      ? `var observedDeviation = 0.0;
  var observedFinite = false;
  for (var radiusIndex = 0u; radiusIndex < RADIUS_COUNT; radiusIndex++) {
    let kValue = observedK[observedKOffset + radiusIndex];
    if (isFiniteFloat(kValue)) {
      observedDeviation = max(observedDeviation, getLDeviation(kValue, getRadius(radiusIndex)));
      observedFinite = true;
    }
  }
  if (observedFinite) {
    observed = observedDeviation;
    var atLeast = 0u;
    for (var simulation = 0u; simulation < SIMULATION_COUNT; simulation++) {
      let value = deviations[deviationsOffset + simulation];
      if (isFiniteFloat(value) && value >= observedDeviation) { atLeast++; }
    }
    pValue = f32(1u + atLeast) / f32(validCount + 1u);
  }`
      : ''
  }
  summary[summaryOffset] = f32(n);
  summary[summaryOffset + 1u] = f32(validCount);
  summary[summaryOffset + 2u] = observed;
  summary[summaryOffset + 3u] = select(nan, critical, validCount >= RANK);
  summary[summaryOffset + 4u] = pValue;
  summary[summaryOffset + 5u] = f32(RANK);`
      })
    );

    // One thread per radius: pointwise ranks and the global band.
    const outputs = [
      ['lower', props.lower],
      ['upper', props.upper],
      ['globalLower', props.globalLower],
      ['globalUpper', props.globalUpper],
      ['theoretical', props.theoretical]
    ] as const;
    if (outputs.some(([, view]) => view)) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-envelope`,
          operation: OPERATION,
          variant: 'envelope',
          bindings: [
            {name: 'parameters', view: parameters, type: 'f32', access: 'read'},
            {name: 'allK', view: allK, type: 'f32', access: 'read'},
            {name: 'summary', view: summary, type: 'f32', access: 'read'},
            ...outputs.flatMap(([name, view]) =>
              view ? [{name, view, type: 'f32' as const, access: 'read_write' as const}] : []
            )
          ],
          invocationCount: radiusCount,
          declarations: constantsWGSL,
          body: `let nan = getQuietNaN(index);
  let radius = getRadius(index);
  let critical = summary[summaryOffset + 3u];
  ${props.theoretical ? 'theoretical[theoreticalOffset + index] = fromL(radius, radius);' : ''}
  ${
    props.globalLower
      ? `globalLower[globalLowerOffset + index] = select(nan, fromL(radius - critical, radius), isFiniteFloat(critical));`
      : ''
  }
  ${
    props.globalUpper
      ? `globalUpper[globalUpperOffset + index] = select(nan, fromL(radius + critical, radius), isFiniteFloat(critical));`
      : ''
  }
  ${
    props.lower || props.upper
      ? `var validCount = 0u;
  for (var simulation = 0u; simulation < SIMULATION_COUNT; simulation++) {
    if (isFiniteFloat(allK[allKOffset + simulation * RADIUS_COUNT + index])) { validCount++; }
  }
  var lowerValue = nan;
  var upperValue = nan;
  if (validCount >= 2u * RANK) {
    for (var candidate = 0u; candidate < SIMULATION_COUNT; candidate++) {
      let value = allK[allKOffset + candidate * RADIUS_COUNT + index];
      if (!isFiniteFloat(value)) { continue; }
      // Position of this value in the ascending order of finite values, ties by simulation index.
      var below = 0u;
      for (var other = 0u; other < SIMULATION_COUNT; other++) {
        let otherValue = allK[allKOffset + other * RADIUS_COUNT + index];
        if (isFiniteFloat(otherValue) && (otherValue < value || (otherValue == value && other < candidate))) {
          below++;
        }
      }
      if (below == RANK - 1u) { lowerValue = value; }
      if (below == validCount - RANK) { upperValue = value; }
    }
  }
  ${props.lower ? 'lower[lowerOffset + index] = select(nan, fromK(lowerValue, radius), isFiniteFloat(lowerValue));' : ''}
  ${props.upper ? 'upper[upperOffset + index] = select(nan, fromK(upperValue, radius), isFiniteFloat(upperValue));' : ''}`
      : ''
  }`
        })
      );
    }
    return nodes;
  }
}
