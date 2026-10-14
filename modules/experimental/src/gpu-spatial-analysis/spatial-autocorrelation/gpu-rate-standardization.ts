// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
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
import {GET_NAN_WGSL, IS_FINITE_VALUE_WGSL} from '../../utils/wgsl-common';

const OPERATION = 'GPUAssuncaoReisStandardization';

/** Rows reduced by one workgroup in the first level of the deterministic sums. */
const BLOCK_ROWS = 4096;

/** Threads of the sum workgroups. */
const WORKGROUP_SIZE = 256;

/** Rows are counted in u32 but divided as f32, which is exact only below 2^24. */
const MAXIMUM_ROWS = 2 ** 24;

/**
 * Number of float32 values of the optional `statistics` view of
 * {@link GPUAssuncaoReisStandardization}: `[n, b, s2, a, totalPopulation]`.
 */
export const GPU_RATE_STANDARDIZATION_STATISTICS_LENGTH = 5;

/** Offsets inside `statistics` (internal). */
const STATISTIC = {count: 0, mean: 1, variance: 2, priorVariance: 3, population: 4} as const;

/**
 * Properties for {@link GPUAssuncaoReisStandardization}.
 *
 * Per-frame (no rebuild or recompile): the contents of `events`, `populations` and `mask`.
 * Compile-time: the row count and which optional views are present.
 */
export type GPUAssuncaoReisStandardizationProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'assuncao-reis-standardization'`. */
  id?: string;
  /** Event counts `e_i`, one per row. Negative or non-finite events mark the row as missing. */
  events: GraphDataView<'float32'>;
  /** Population at risk `n_i`, one per row. Populations that are not finite or not > 0 are missing. */
  populations: GraphDataView<'float32'>;
  /** Optional row selection: nonzero includes the row; masked rows are not part of the analysis. */
  mask?: GraphDataView<'uint32'>;
  /**
   * Caller-owned output `z_i`, one float32 per row: the EB-standardized rate, NaN for missing and
   * masked rows and where the standardizing variance is not positive.
   */
  standardizedRates: GraphDataView<'float32'>;
  /**
   * Optional caller-owned global quantities, at least
   * {@link GPU_RATE_STANDARDIZATION_STATISTICS_LENGTH} float32 values:
   * `[n, b, s2, a, totalPopulation]` (see {@link GPUAssuncaoReisStandardization}).
   */
  statistics?: GraphDataView<'float32'>;
  /**
   * Optional caller-owned count of unmasked rows that were rejected as missing (non-finite or
   * negative events, or a population that is not finite and positive). This is the
   * `GPUInvalidStatusPort.invalidCount` of the status vocabulary.
   */
  invalidCount?: GraphDataView<'uint32'>;
};

/** One fixed-order block accumulator of {@link createBlockSumNodes}. */
type SumAccumulator = {
  name: string;
  type: 'f32' | 'u32';
  /** WGSL expression of the row term, with `row` in scope. */
  term: string;
  /** Rows that contribute: valid rows or unmasked-but-invalid rows. */
  rows: 'valid' | 'invalid';
};

/**
 * Builds a blocks kernel and a reduce kernel for fixed-order tree sums of the given accumulators:
 * workgroup `b` sums rows `b * 4096 + t`, `+ 256`, ... per thread and combines the threads with a
 * binary tree, then one workgroup sums the block partials the same way. No float atomics, so the
 * result depends only on the shape. `finalize` runs on thread 0 with `total_<name>` in scope.
 */
function createBlockSumNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    variant: string;
    rows: number;
    accumulators: readonly SumAccumulator[];
    rowBindings: readonly WGSLKernelBinding[];
    rowDeclarations: string;
    finalizeBindings: readonly WGSLKernelBinding[];
    finalize: string;
  }
): GPUCommandNode<Parameters>[] {
  const {id, variant, rows, accumulators} = props;
  const blockCount = Math.ceil(rows / BLOCK_ROWS);
  const zero = (type: 'f32' | 'u32') => (type === 'f32' ? '0.0' : '0u');
  const partials = accumulators.map(accumulator => ({
    ...accumulator,
    view: createTransientView(
      graph,
      `${id}-${variant}-${accumulator.name}-partials`,
      accumulator.type === 'f32' ? 'float32' : 'uint32',
      blockCount
    )
  }));
  const workgroupDeclarations = accumulators
    .map(a => `var<workgroup> shared_${a.name}: array<${a.type}, ${WORKGROUP_SIZE}>;`)
    .join('\n');
  const treeReduction = `workgroupBarrier();
  for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride / 2u) {
    if (localInvocationIndex < stride) {
      ${accumulators.map(a => `shared_${a.name}[localInvocationIndex] += shared_${a.name}[localInvocationIndex + stride];`).join('\n      ')}
    }
    workgroupBarrier();
  }`;
  const partialBindings = (access: 'read' | 'read_write'): WGSLKernelBinding[] =>
    partials.map(p => ({name: `${p.name}Partials`, view: p.view, type: p.type, access}));

  const blocks = createWGSLKernelNode<Parameters>(graph, {
    id: `${id}-${variant}-blocks`,
    operation: OPERATION,
    variant: `${variant}-blocks`,
    bindings: [...props.rowBindings, ...partialBindings('read_write')],
    workgroupSize: WORKGROUP_SIZE,
    invocationCount: blockCount * WORKGROUP_SIZE,
    guardIndex: false,
    declarations: `${props.rowDeclarations}
const ROW_COUNT: u32 = ${rows}u;
const BLOCK_ROWS: u32 = ${BLOCK_ROWS}u;
const WORKGROUP_SIZE: u32 = ${WORKGROUP_SIZE}u;
${workgroupDeclarations}`,
    // No early return: every invocation of a workgroup must reach the barriers.
    body: `let block = index / WORKGROUP_SIZE;
  let isInRange = index < INVOCATION_COUNT;
  ${accumulators.map(a => `var local_${a.name} = ${zero(a.type)};`).join('\n  ')}
  if (isInRange) {
    let end = min((block + 1u) * BLOCK_ROWS, ROW_COUNT);
    for (var row = block * BLOCK_ROWS + localInvocationIndex; row < end; row += WORKGROUP_SIZE) {
      let isValid = isRowValid(row);
      let isInvalid = isRowIncluded(row) && !isValid;
      ${accumulators
        .map(
          a =>
            `if (${a.rows === 'valid' ? 'isValid' : 'isInvalid'}) { local_${a.name} += ${a.term}; }`
        )
        .join('\n      ')}
    }
  }
  ${accumulators.map(a => `shared_${a.name}[localInvocationIndex] = local_${a.name};`).join('\n  ')}
  ${treeReduction}
  if (isInRange && localInvocationIndex == 0u) {
    ${accumulators.map(a => `${a.name}Partials[${a.name}PartialsOffset + block] = shared_${a.name}[0];`).join('\n    ')}
  }`
  });

  const reduce = createWGSLKernelNode<Parameters>(graph, {
    id: `${id}-${variant}-reduce`,
    operation: OPERATION,
    variant: `${variant}-reduce`,
    bindings: [...partialBindings('read'), ...props.finalizeBindings],
    workgroupSize: WORKGROUP_SIZE,
    invocationCount: WORKGROUP_SIZE,
    guardIndex: false,
    declarations: `${GET_NAN_WGSL}
const BLOCK_COUNT: u32 = ${blockCount}u;
const WORKGROUP_SIZE: u32 = ${WORKGROUP_SIZE}u;
${workgroupDeclarations}`,
    body: `${accumulators.map(a => `var local_${a.name} = ${zero(a.type)};`).join('\n  ')}
  for (var block = localInvocationIndex; block < BLOCK_COUNT; block += WORKGROUP_SIZE) {
    ${accumulators.map(a => `local_${a.name} += ${a.name}Partials[${a.name}PartialsOffset + block];`).join('\n    ')}
  }
  ${accumulators.map(a => `shared_${a.name}[localInvocationIndex] = local_${a.name};`).join('\n  ')}
  ${treeReduction}
  if (localInvocationIndex == 0u) {
    ${accumulators.map(a => `let total_${a.name} = shared_${a.name}[0];`).join('\n    ')}
    ${props.finalize}
  }`
  });
  return [blocks, reduce];
}

/**
 * Empirical-Bayes standardized rates of Assuncao and Reis (1999), the transformation behind
 * `esda.Moran_Rate` and `esda.Moran_Local_Rate`. It is a transformation node: it writes one
 * standardized value `z_i` per row and no spatial statistic, so the Moran statistics of rate data
 * come from composing it with the existing nodes (no statistic code is duplicated).
 *
 * Definition (identical to `esda.smoothing.assuncao_rate`), over the `m` valid rows, with rates
 * `p_i = e_i / n_i`:
 *
 * - `b = sum e / sum n` (the global EB mean),
 * - `s2 = sum n_i (p_i - b)^2 / sum n` (population-weighted variance of the rates),
 * - `a = s2 - b / (sum n / m)` (the EB prior variance),
 * - `v_i = a + b / n_i`, replaced by `b / n_i` where `v_i < 0`,
 * - `z_i = (p_i - b) / sqrt(v_i)`.
 *
 * Handling of a negative prior variance: `a` can be negative when the rates vary less than
 * Poisson noise alone predicts. esda does NOT clamp `a` globally. It keeps the (negative) `a`
 * wherever `v_i` stays non-negative and falls back to the pure Poisson variance `b / n_i` only for
 * the rows where `v_i < 0` (the smallest populations). This node does exactly the same for parity
 * (`statistics[3]` reports the unclamped `a`). Note that the EB rate smoothing in `rate-smoothing`
 * clamps the prior variance at zero instead, which is a different (also standard) convention for
 * shrinkage; the two are not meant to agree when `a < 0`.
 *
 * Missing rows: a row is valid when it is unmasked, `events` is finite and >= 0, and `populations`
 * is finite and > 0. Every other row writes NaN, is excluded from `b`, `s2`, `m` and the mean
 * population (rejected unmasked rows are counted in `invalidCount`), and a NaN row is in turn
 * excluded by the downstream statistics. Rows where `v_i` is not positive (for example `b = 0`
 * when no valid row has an event) write NaN, as esda's 0 / 0 does. With no valid row, `statistics`
 * `[1..4]` are NaN.
 *
 * Precision: the count `m` is a u32 and the sums use fixed-order two-level workgroup trees (no
 * float atomics, bitwise reproducible). `b` gets a second-pass correction,
 * `b += sum(e_i - b n_i) / sum n`, and `s2` is formed from centered deviations about the
 * corrected `b`. Row count must be below 2^24.
 *
 * Composition recipes:
 *
 * - EB Moran's I for rates (`esda.Moran_Rate`, default `transformation='r'`): run this node, feed
 *   `standardizedRates` as `values` to `GPUGlobalSpatialStatistics` with `statistics: ['moran']`
 *   and row-standardized weights (`weights.info.transform === 'row'`, from
 *   `GPUSpatialWeightsTransform`). `I` equals `Moran_Rate.I`, `EI` equals `Moran_Rate.EI`, and
 *   `zRandomization` equals `Moran_Rate.z_rand` (esda recomputes the Moran of the standardized
 *   values, ignoring the EB sampling variance). Use `islands: 'keep'` for esda's convention.
 * - EB local Moran (`esda.Moran_Local_Rate`): feed the same `standardizedRates` to `GPULocalMoran`
 *   (analytic z/p, quadrants) or `GPULocalPermutationTest` with the `'localMoran'` statistic
 *   (conditional permutations). `GPULocalMoran.localI` equals `Moran_Local_Rate.Is`.
 *
 * Rows that are NaN here are excluded from the downstream analysis and from its `n`, which is the
 * same convention as dropping them in esda. Weights built over the full table stay valid, because
 * the downstream node re-standardizes row-standardized weights over the active neighbors.
 */
export class GPUAssuncaoReisStandardization implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUAssuncaoReisStandardizationProps;

  constructor(props: GPUAssuncaoReisStandardizationProps) {
    const id = props.id ?? 'assuncao-reis-standardization';
    this.id = id;
    this.props = props;
    validatePackedView(props.events, ['float32'], `${id} events`);
    validatePackedView(props.populations, ['float32'], `${id} populations`);
    validatePackedView(props.standardizedRates, ['float32'], `${id} standardizedRates`);
    const rows = props.events.length;
    if (rows >= MAXIMUM_ROWS) {
      throw new Error(`${id} events must hold fewer than 2^24 rows`);
    }
    for (const [name, view] of [
      ['populations', props.populations],
      ['standardizedRates', props.standardizedRates]
    ] as const) {
      if (view.length !== rows) {
        throw new Error(`${id} ${name} length must equal the events length`);
      }
    }
    if (props.mask) {
      validatePackedUint32View(props.mask, `${id} mask`);
      if (props.mask.length !== rows) {
        throw new Error(`${id} mask length must equal the events length`);
      }
    }
    if (props.statistics) {
      validatePackedView(props.statistics, ['float32'], `${id} statistics`);
      if (props.statistics.length < GPU_RATE_STANDARDIZATION_STATISTICS_LENGTH) {
        throw new Error(
          `${id} statistics must hold ${GPU_RATE_STANDARDIZATION_STATISTICS_LENGTH} float32 values`
        );
      }
    }
    if (props.invalidCount) {
      validatePackedUint32View(props.invalidCount, `${id} invalidCount`);
      if (props.invalidCount.length < 1) {
        throw new Error(`${id} invalidCount must hold 1 uint32 value`);
      }
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      [props.standardizedRates, props.statistics, props.invalidCount],
      [props.events, props.populations, props.mask]
    );
  }

  /** Returns the standardization nodes in dependency order. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {events, populations, mask, standardizedRates, invalidCount} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      events,
      populations,
      mask,
      standardizedRates,
      props.statistics,
      invalidCount
    ]);
    const rows = events.length;
    const statistics =
      props.statistics ??
      createTransientView(
        graph,
        `${id}-statistics`,
        'float32',
        GPU_RATE_STANDARDIZATION_STATISTICS_LENGTH
      );
    const rowBindings: WGSLKernelBinding[] = [
      {name: 'events', view: events, type: 'f32', access: 'read'},
      {name: 'populations', view: populations, type: 'f32', access: 'read'},
      ...(mask ? [{name: 'mask', view: mask, type: 'u32' as const, access: 'read' as const}] : [])
    ];
    const rowDeclarations = `${IS_FINITE_VALUE_WGSL}
fn getSquare(value: f32) -> f32 { return value * value; }
fn isRowIncluded(row: u32) -> bool {
  return ${mask ? 'mask[maskOffset + row] != 0u' : 'true'};
}
fn isRowValid(row: u32) -> bool {
  let event = events[eventsOffset + row];
  let population = populations[populationsOffset + row];
  return isRowIncluded(row) && isFiniteValue(event) && isFiniteValue(population) &&
    event >= 0.0 && population > 0.0;
}`;
    const statisticsBinding = (access: 'read' | 'read_write'): WGSLKernelBinding => ({
      name: 'statistics',
      view: statistics,
      type: 'f32',
      access
    });
    const S = STATISTIC;
    const stat = (slot: number) => `statistics[statisticsOffset + ${slot}u]`;
    const eventTerm = 'events[eventsOffset + row]';
    const populationTerm = 'populations[populationsOffset + row]';

    const nodes: GPUCommandNode<Parameters>[] = [];
    // Pass 1: count, sum of events, sum of populations, rejected rows -> provisional b.
    nodes.push(
      ...createBlockSumNodes(graph, {
        id,
        variant: 'totals',
        rows,
        accumulators: [
          {name: 'events', type: 'f32', term: eventTerm, rows: 'valid'},
          {name: 'population', type: 'f32', term: populationTerm, rows: 'valid'},
          {name: 'count', type: 'u32', term: '1u', rows: 'valid'},
          {name: 'invalid', type: 'u32', term: '1u', rows: 'invalid'}
        ],
        rowBindings,
        rowDeclarations,
        finalizeBindings: [
          statisticsBinding('read_write'),
          ...(invalidCount
            ? [
                {
                  name: 'invalidCount',
                  view: invalidCount,
                  type: 'u32' as const,
                  access: 'read_write' as const
                }
              ]
            : [])
        ],
        // The count is exact in u32 and becomes f32 only here. b is provisional until pass 2.
        finalize: `let count = f32(total_count);
    ${invalidCount ? 'invalidCount[invalidCountOffset] = total_invalid;' : ''}
    ${stat(S.count)} = count;
    ${stat(S.population)} = total_population;
    if (count >= 1.0) {
      ${stat(S.mean)} = total_events / total_population;
    } else {
      ${stat(S.mean)} = getNaN();
      ${stat(S.population)} = getNaN();
    }
    ${stat(S.variance)} = getNaN();
    ${stat(S.priorVariance)} = getNaN();`
      })
    );
    // Pass 2: second-pass correction of b from the deviations e_i - b n_i.
    nodes.push(
      ...createBlockSumNodes(graph, {
        id,
        variant: 'mean-correction',
        rows,
        accumulators: [
          {
            name: 'deviation',
            type: 'f32',
            term: `${eventTerm} - ${stat(S.mean)} * ${populationTerm}`,
            rows: 'valid'
          }
        ],
        rowBindings: [...rowBindings, statisticsBinding('read')],
        rowDeclarations,
        finalizeBindings: [statisticsBinding('read_write')],
        finalize: `if (${stat(S.count)} >= 1.0) {
      ${stat(S.mean)} = ${stat(S.mean)} + total_deviation / ${stat(S.population)};
    }`
      })
    );
    // Pass 3: s2 from centered deviations, then the prior variance a.
    nodes.push(
      ...createBlockSumNodes(graph, {
        id,
        variant: 'variance',
        rows,
        accumulators: [
          {
            name: 'square',
            type: 'f32',
            term: `${populationTerm} * getSquare(${eventTerm} / ${populationTerm} - ${stat(S.mean)})`,
            rows: 'valid'
          }
        ],
        rowBindings: [...rowBindings, statisticsBinding('read')],
        rowDeclarations,
        finalizeBindings: [statisticsBinding('read_write')],
        finalize: `if (${stat(S.count)} >= 1.0) {
      let variance = total_square / ${stat(S.population)};
      ${stat(S.variance)} = variance;
      ${stat(S.priorVariance)} = variance - ${stat(S.mean)} / (${stat(S.population)} / ${stat(S.count)});
    }`
      })
    );
    // Per row: z = (p - b) / sqrt(v), v = a + b / n, or b / n where that is negative (as esda).
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-standardize`,
        operation: OPERATION,
        variant: 'standardize',
        bindings: [
          ...rowBindings,
          statisticsBinding('read'),
          {name: 'standardizedRates', view: standardizedRates, type: 'f32', access: 'read_write'}
        ],
        invocationCount: rows,
        declarations: `${GET_NAN_WGSL}
${rowDeclarations}`,
        body: `if (!isRowValid(index)) {
    standardizedRates[standardizedRatesOffset + index] = getNaN();
    return;
  }
  let population = populations[populationsOffset + index];
  let rate = events[eventsOffset + index] / population;
  let mean = ${stat(S.mean)};
  let poissonVariance = mean / population;
  let rawVariance = ${stat(S.priorVariance)} + poissonVariance;
  let variance = select(rawVariance, poissonVariance, rawVariance < 0.0);
  standardizedRates[standardizedRatesOffset + index] =
    select(getNaN(), (rate - mean) / sqrt(variance), variance > 0.0);`
      })
    );
    return nodes;
  }
}
