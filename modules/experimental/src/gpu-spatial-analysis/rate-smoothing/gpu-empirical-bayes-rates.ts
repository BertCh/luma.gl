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
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {createWGSLKernelNode, type WGSLKernelBinding} from '../../utils/wgsl-kernel-nodes';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../utils/gpu-contributor-utils';
import {SPATIAL_AUTOCORRELATION_FLOAT_WGSL} from '../spatial-autocorrelation/spatial-autocorrelation-kernels';

const OPERATION = 'GPUEmpiricalBayesRates';
/** Lanes of the reduction workgroups. */
const LANES = 256;

/** Fields of the `summary` view of {@link GPUEmpiricalBayesRates}. */
export const GPU_EMPIRICAL_BAYES_SUMMARY = {
  /** Number of included rows `n`. */
  count: 0,
  /** Total events `sum e`. */
  eventSum: 1,
  /** Total population `sum b`. */
  populationSum: 2,
  /** Pooled rate `m = sum e / sum b` (esda `ebi_b`, `r_mean`). */
  pooledRate: 3,
  /** Population-weighted variance of the raw rates, `sum b (y - m)^2 / sum b` (esda `s2`). */
  weightedRateVariance: 4,
  /** Estimated variance of the true rates, `a = s2 - m / mean(b)` (esda `ebi_a`, `r_var`); may be negative. */
  priorVariance: 5,
  /** Number of float32 values in `summary`. */
  length: 8
} as const;

/**
 * Properties for {@link GPUEmpiricalBayesRates}.
 *
 * Per-frame (no rebuild or recompile): the contents of `events`, `populations` and `mask`.
 * Compile-time: the row count and which outputs exist.
 */
export type GPUEmpiricalBayesRatesProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'empirical-bayes-rates'`. */
  id?: string;
  /** Event counts `e`, one per row. */
  events: GraphDataView<'float32'>;
  /** Populations at risk `b`, one per row. Rows with `b <= 0` or a non-finite value are excluded. */
  populations: GraphDataView<'float32'>;
  /** Optional row selection: nonzero includes the row. */
  mask?: GraphDataView<'uint32'>;
  /**
   * Optional caller-owned Assunção-Reis standardized rate per row (esda `assuncao_rate`), quiet NaN
   * for excluded rows. This is the column to give `GPULocalMoran` and `GPUGlobalSpatialStatistics`
   * as `values` for esda's `Moran_Local_Rate` / `Moran_Rate`.
   */
  standardizedRates?: GraphDataView<'float32'>;
  /** Optional caller-owned global empirical-Bayes smoothed rate per row (esda `Empirical_Bayes`), NaN if excluded. */
  smoothedRates?: GraphDataView<'float32'>;
  /** Optional caller-owned raw rate `e / b` per row, NaN if excluded. */
  rawRates?: GraphDataView<'float32'>;
  /** Optional caller-owned moments, `GPU_EMPIRICAL_BAYES_SUMMARY.length` float32 values. */
  summary?: GraphDataView<'float32'>;
};

/**
 * Global empirical-Bayes rate standardization and smoothing for rate variables, matching PySAL
 * esda `smoothing.assuncao_rate` (used by `Moran_Rate` and `Moran_Local_Rate` with
 * `adjusted=True`) and `smoothing.Empirical_Bayes`.
 *
 * Definitions over the `n` included rows (mask nonzero, finite `e`, finite `b > 0`; esda would
 * return NaN for the whole result if any `b` were zero, so such rows are excluded here), with raw
 * rates `y = e / b`:
 * - `m = sum e / sum b`, `s2 = sum b (y - m)^2 / sum b`, `a = s2 - m / mean(b)`.
 * - Standardized rate: `z = (y - m) / sqrt(v)` with `v = a + m / b`, replaced by `m / b` when
 *   `a + m / b < 0` (esda's guard), exactly `assuncao_rate`.
 * - Smoothed rate: `r = w y + (1 - w) m` with `w = a / (a + m / b)`, exactly `Empirical_Bayes.r`;
 *   esda does not clamp `a`, so `w` can leave `[0, 1]` when `a < 0`, and `summary` exposes `a`.
 *
 * The totals are reduced in a fixed order (a workgroup tree per block of rows, then a tree over
 * the block partials), straight from `events` and `populations` with no intermediate table, so
 * results are bitwise reproducible. Compose it in front of a statistic in one graph: write `standardizedRates` into a
 * transient view and pass that view as `values`. For the neighborhood-pooled variant (esda
 * `Spatial_Empirical_Bayes`) use {@link GPUSpatialEmpiricalBayesRates}.
 */
export class GPUEmpiricalBayesRates implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUEmpiricalBayesRatesProps;

  constructor(props: GPUEmpiricalBayesRatesProps) {
    const id = props.id ?? 'empirical-bayes-rates';
    this.id = id;
    this.props = props;
    validatePackedView(props.events, ['float32'], `${id} events`);
    validatePackedView(props.populations, ['float32'], `${id} populations`);
    const rows = props.events.length;
    if (rows < 1) {
      throw new Error(`${id} events must hold at least one row`);
    }
    if (props.populations.length !== rows) {
      throw new Error(`${id} populations length must equal the events length`);
    }
    if (props.mask) {
      validatePackedUint32View(props.mask, `${id} mask`);
      if (props.mask.length !== rows) {
        throw new Error(`${id} mask length must equal the events length`);
      }
    }
    for (const [name, view] of [
      ['standardizedRates', props.standardizedRates],
      ['smoothedRates', props.smoothedRates],
      ['rawRates', props.rawRates]
    ] as const) {
      if (view) {
        validatePackedView(view, ['float32'], `${id} ${name}`);
        if (view.length !== rows) {
          throw new Error(`${id} ${name} length must equal the events length`);
        }
      }
    }
    if (props.summary) {
      validatePackedView(props.summary, ['float32'], `${id} summary`);
      if (props.summary.length < GPU_EMPIRICAL_BAYES_SUMMARY.length) {
        throw new Error(
          `${id} summary must hold ${GPU_EMPIRICAL_BAYES_SUMMARY.length} float32 values`
        );
      }
    }
    if (!props.standardizedRates && !props.smoothedRates && !props.rawRates && !props.summary) {
      throw new Error(`${id} requests no output`);
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      [props.standardizedRates, props.smoothedRates, props.rawRates, props.summary],
      [props.events, props.populations, props.mask]
    );
  }

  /** Returns the rate nodes in dependency order. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {events, populations, mask} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      events,
      populations,
      mask,
      props.standardizedRates,
      props.smoothedRates,
      props.rawRates,
      props.summary
    ]);
    const rows = events.length;
    const groups = Math.ceil(rows / LANES);
    const partialsA = createTransientView(graph, `${id}-partials-a`, 'float32', 3 * groups);
    const partialsB = createTransientView(graph, `${id}-partials-b`, 'float32', groups);
    const totalsA = createTransientView(graph, `${id}-totals-a`, 'float32', 3);
    const summary =
      props.summary ??
      createTransientView(graph, `${id}-summary`, 'float32', GPU_EMPIRICAL_BAYES_SUMMARY.length);
    const S = GPU_EMPIRICAL_BAYES_SUMMARY;
    const constants = `const ROWS: u32 = ${rows}u;
const LANES: u32 = ${LANES}u;
const GROUPS: u32 = ${groups}u;
${SPATIAL_AUTOCORRELATION_FLOAT_WGSL}
var<workgroup> scratch: array<f32, ${LANES}>;
// Fixed-order tree sum of one value per lane. Every lane of the workgroup must call it together.
fn reduceSum(value: f32, lane: u32) -> f32 {
  workgroupBarrier();
  scratch[lane] = value;
  workgroupBarrier();
  for (var stride = LANES / 2u; stride > 0u; stride = stride / 2u) {
    if (lane < stride) {
      scratch[lane] += scratch[lane + stride];
    }
    workgroupBarrier();
  }
  return scratch[0];
}`;
    const maskBinding: WGSLKernelBinding[] = mask
      ? [{name: 'mask', view: mask, type: 'u32', access: 'read'}]
      : [];
    const includedWGSL = `${mask ? 'mask[maskOffset + index] != 0u && ' : ''}isFiniteFloat(e) && isFiniteFloat(b) && b > 0.0`;
    // Reads this lane's row; `included` is false for the padding lanes of the last block.
    const readRowWGSL = `var e = 0.0;
  var b = 0.0;
  var included = false;
  if (index < ROWS) {
    e = events[eventsOffset + index];
    b = populations[populationsOffset + index];
    included = ${includedWGSL};
  }`;
    const nodes: GPUCommandNode<Parameters>[] = [
      // Block partials of (count, sum e, sum b): one workgroup per block of rows.
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-partials-a`,
        operation: OPERATION,
        variant: 'partials-a',
        bindings: [
          {name: 'events', view: events, type: 'f32', access: 'read'},
          {name: 'populations', view: populations, type: 'f32', access: 'read'},
          ...maskBinding,
          {name: 'partialsA', view: partialsA, type: 'f32', access: 'read_write'}
        ],
        invocationCount: groups * LANES,
        guardIndex: false,
        declarations: constants,
        body: `let lane = localInvocationIndex;
  let group = (index - lane) / LANES;
  ${readRowWGSL}
  let count = reduceSum(select(0.0, 1.0, included), lane);
  let eventSum = reduceSum(select(0.0, e, included), lane);
  let populationSum = reduceSum(select(0.0, b, included), lane);
  if (lane == 0u) {
    partialsA[partialsAOffset + group] = count;
    partialsA[partialsAOffset + GROUPS + group] = eventSum;
    partialsA[partialsAOffset + 2u * GROUPS + group] = populationSum;
  }`
      }),
      // One workgroup per column sums the block partials: lane l takes blocks l, l + 256, ...
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-totals-a`,
        operation: OPERATION,
        variant: 'totals-a',
        bindings: [
          {name: 'partialsA', view: partialsA, type: 'f32', access: 'read'},
          {name: 'totalsA', view: totalsA, type: 'f32', access: 'read_write'}
        ],
        invocationCount: 3 * LANES,
        guardIndex: false,
        declarations: constants,
        body: `let lane = localInvocationIndex;
  let column = index / LANES;
  var sum = 0.0;
  for (var group = lane; group < GROUPS; group += LANES) {
    sum += partialsA[partialsAOffset + column * GROUPS + group];
  }
  let total = reduceSum(sum, lane);
  if (lane == 0u) {
    totalsA[totalsAOffset + column] = total;
  }`
      }),
      // The deviations about the pooled rate need the totals first, so this is the second pass.
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-partials-b`,
        operation: OPERATION,
        variant: 'partials-b',
        bindings: [
          {name: 'events', view: events, type: 'f32', access: 'read'},
          {name: 'populations', view: populations, type: 'f32', access: 'read'},
          ...maskBinding,
          {name: 'totalsA', view: totalsA, type: 'f32', access: 'read'},
          {name: 'partialsB', view: partialsB, type: 'f32', access: 'read_write'}
        ],
        invocationCount: groups * LANES,
        guardIndex: false,
        declarations: constants,
        body: `let lane = localInvocationIndex;
  let group = (index - lane) / LANES;
  ${readRowWGSL}
  let pooledRate = totalsA[totalsAOffset + 1u] / totalsA[totalsAOffset + 2u];
  let deviation = e / b - pooledRate;
  let total = reduceSum(select(0.0, b * deviation * deviation, included), lane);
  if (lane == 0u) {
    partialsB[partialsBOffset + group] = total;
  }`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-summary`,
        operation: OPERATION,
        variant: 'summary',
        bindings: [
          {name: 'totalsA', view: totalsA, type: 'f32', access: 'read'},
          {name: 'partialsB', view: partialsB, type: 'f32', access: 'read'},
          {name: 'summary', view: summary, type: 'f32', access: 'read_write'}
        ],
        invocationCount: LANES,
        guardIndex: false,
        declarations: constants,
        body: `let lane = localInvocationIndex;
  var sum = 0.0;
  for (var group = lane; group < GROUPS; group += LANES) {
    sum += partialsB[partialsBOffset + group];
  }
  let deviationSum = reduceSum(sum, lane);
  if (lane == 0u) {
    let count = totalsA[totalsAOffset];
    let eventSum = totalsA[totalsAOffset + 1u];
    let populationSum = totalsA[totalsAOffset + 2u];
    let pooledRate = eventSum / populationSum;
    let weightedVariance = deviationSum / populationSum;
    summary[summaryOffset + ${S.count}u] = count;
    summary[summaryOffset + ${S.eventSum}u] = eventSum;
    summary[summaryOffset + ${S.populationSum}u] = populationSum;
    summary[summaryOffset + ${S.pooledRate}u] = pooledRate;
    summary[summaryOffset + ${S.weightedRateVariance}u] = weightedVariance;
    summary[summaryOffset + ${S.priorVariance}u] = weightedVariance - pooledRate / (populationSum / count);
    for (var field = ${S.priorVariance + 1}u; field < ${S.length}u; field++) {
      summary[summaryOffset + field] = 0.0;
    }
  }`
      })
    ];

    const outputBindings: WGSLKernelBinding[] = [];
    const outputWGSL: string[] = [];
    const excludedWGSL: string[] = [];
    for (const [name, view, expression] of [
      ['standardizedRates', props.standardizedRates, 'standardized'],
      ['smoothedRates', props.smoothedRates, 'smoothed'],
      ['rawRates', props.rawRates, 'rate']
    ] as const) {
      if (view) {
        outputBindings.push({name, view, type: 'f32', access: 'read_write'});
        outputWGSL.push(`${name}[${name}Offset + index] = ${expression};`);
        excludedWGSL.push(`${name}[${name}Offset + index] = nan;`);
      }
    }
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-rates`,
        operation: OPERATION,
        variant: 'rates',
        bindings: [
          {name: 'events', view: events, type: 'f32', access: 'read'},
          {name: 'populations', view: populations, type: 'f32', access: 'read'},
          ...maskBinding,
          {name: 'summary', view: summary, type: 'f32', access: 'read'},
          ...outputBindings
        ],
        invocationCount: rows,
        declarations: constants,
        body: `let e = events[eventsOffset + index];
  let b = populations[populationsOffset + index];
  let nan = getQuietNaN(index);
  if (!(${includedWGSL})) {
    ${excludedWGSL.join('\n    ')}
    return;
  }
  let pooledRate = summary[summaryOffset + ${S.pooledRate}u];
  let priorVariance = summary[summaryOffset + ${S.priorVariance}u];
  let rate = e / b;
  let noise = pooledRate / b;
  let rawVariance = priorVariance + noise;
  let variance = select(rawVariance, noise, rawVariance < 0.0);
  let standardized = (rate - pooledRate) / sqrt(variance);
  let weight = priorVariance / rawVariance;
  let smoothed = weight * rate + (1.0 - weight) * pooledRate;
  ${outputWGSL.join('\n  ')}`
      })
    );
    return nodes;
  }
}
