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
import {createWGSLKernelNode} from '../../utils/wgsl-kernel-nodes';
import {GET_NAN_WGSL, IS_NAN_BITS_WGSL} from '../../utils/wgsl-common';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../utils/gpu-contributor-utils';
import {validateGPUStatusPort, type GPUIterationStatusPort} from '../contracts/index';
import {GPU_UPPER_INCOMPLETE_GAMMA_WGSL} from './incomplete-gamma-wgsl';

const OPERATION = 'GPUTransitionDiagnostics';
const MAXIMUM_CLASS_COUNT = 64;

/** Default {@link GPUTransitionDiagnosticsProps.minimumTransitions}. */
export const GPU_TRANSITION_DEFAULT_MINIMUM_TRANSITIONS = 5;

/** Row layout of the `homogeneity` output of {@link GPUTransitionDiagnostics}. */
export const GPU_TRANSITION_HOMOGENEITY = {
  /** Chi-square statistic Q (giddy `Spatial_Markov.Q`). */
  Q: 0,
  /** Upper-tail chi-square p-value of Q (`Q_p_value`); NaN when the degrees of freedom are 0. */
  Q_P_VALUE: 1,
  /** Likelihood-ratio statistic LR (`LR`). */
  LR: 2,
  /** Upper-tail chi-square p-value of LR (`LR_p_value`); NaN when the degrees of freedom are 0. */
  LR_P_VALUE: 3,
  /** Degrees of freedom (`dof_hom`), exact in f32. */
  DEGREES_OF_FREEDOM: 4
} as const;

/** Number of rows of the `homogeneity` output. */
export const GPU_TRANSITION_HOMOGENEITY_LENGTH = 5;

/** Row layout of the `stateSparsity` output of {@link GPUTransitionDiagnostics}. */
export const GPU_TRANSITION_STATE_SPARSITY = {
  /** Number of `(condition, from)` states with at least one but fewer than `minimumTransitions` transitions. */
  SPARSE: 0,
  /** Number of `(condition, from)` states with no transitions. */
  EMPTY: 1
} as const;

/** Number of rows of the `stateSparsity` output. */
export const GPU_TRANSITION_STATE_SPARSITY_LENGTH = 2;

/** Caller-owned outputs of {@link GPUTransitionDiagnostics}. */
export type GPUTransitionDiagnosticsOutput = {
  /**
   * Optional homogeneity test of the conditional transition matrices (see
   * {@link GPU_TRANSITION_HOMOGENEITY}), at least {@link GPU_TRANSITION_HOMOGENEITY_LENGTH} rows.
   * Needs `conditionCount > 1`.
   */
  homogeneity?: GraphDataView<'float32'>;
  /**
   * Optional steady-state (ergodic) distributions, `chains * classCount` rows with `chains = 1`
   * for `conditionCount = 1` and `conditionCount + 1` otherwise. Chain 0 is the pooled chain
   * (counts summed over the conditions); chain `1 + c` is the chain of condition `c`. Row
   * `chain * classCount + s` is the long-run probability of state `s`. A chain without any
   * transition, with a state that is entered but never left, or that is reducible after dropping
   * its unused states, is all NaN.
   */
  steadyStates?: GraphDataView<'float32'>;
  /**
   * Optional minimum-cell-count diagnostic (see {@link GPU_TRANSITION_STATE_SPARSITY}), at least
   * {@link GPU_TRANSITION_STATE_SPARSITY_LENGTH} rows.
   */
  stateSparsity?: GraphDataView<'uint32'>;
  /**
   * Optional status. `converged` is one when every chain with at least one transition got a
   * steady state, zero otherwise (see the NaN rows of `steadyStates` for which chain). The
   * solver is direct, so `iterationLimitReached` is never written. Needs `steadyStates`.
   */
  status?: Pick<GPUIterationStatusPort, 'converged'>;
};

/** Properties for {@link GPUTransitionDiagnostics}. */
export type GPUTransitionDiagnosticsProps = {
  /** Prefix for generated node IDs. Defaults to `'transition-diagnostics'`. */
  id?: string;
  /**
   * Exact transition counts, `conditionCount * classCount * classCount` rows, laid out as the
   * `counts` of {@link GPUTransitionMatrix}.
   */
  counts: GraphDataView<'uint32'>;
  /** Number of classes `K`, 1 to 64. */
  classCount: number;
  /** Number of condition classes `C`. Defaults to 1. */
  conditionCount?: number;
  /**
   * Compile-time. A `(condition, from)` state with fewer transitions than this is counted as
   * sparse. Defaults to {@link GPU_TRANSITION_DEFAULT_MINIMUM_TRANSITIONS}, the classic rule of
   * thumb of five observations per cell row for chi-square tests of transition matrices.
   */
  minimumTransitions?: number;
  /** Caller-owned outputs. */
  output: GPUTransitionDiagnosticsOutput;
};

/**
 * Diagnostics of a transition-count tensor: homogeneity of the conditional matrices, steady
 * states of the pooled and conditional chains, and a minimum-cell-count count.
 *
 * Homogeneity (Bickenbach and Bode 2003, giddy `homogeneity` behind `Spatial_Markov.Q`, `LR`,
 * `dof_hom`, `Q_p_value`, `LR_p_value`): the null is that every conditional matrix equals the
 * pooled matrix. `Q = sum n_im (p_ijm - p_ij)^2 / p_ij` and `LR = 2 sum n_ijm ln(p_ijm / p_ij)` are
 * asymptotically chi-square with `sum_i (b_i - 1)(A_i - 1)` degrees of freedom, `b_i` the number of
 * conditions with a transition out of state `i` and `A_i` the number of non-zero pooled cells of
 * row `i`. A from-state without any transition adds 0, whereas giddy's expression adds 1 for it
 * (`(0 - 1)(0 - 1)`), so the degrees of freedom differ from giddy's exactly by the number of empty
 * pooled rows. The p-values are upper-tail (`'greater'`) chi-square probabilities from an f32
 * incomplete gamma, see `GPU_UPPER_INCOMPLETE_GAMMA_WGSL`. The test is asymptotic: it is not
 * trustworthy with sparse rows, see `stateSparsity`.
 *
 * Steady states: the stationary distribution of the row-normalized counts, by the
 * Grassmann-Taksar-Heyman elimination (subtraction free, so stable in f32), one invocation per
 * chain. States with no transition in or out are dropped and get probability 0, as giddy's
 * `Markov` drops classes absent from the data; the rest must form one communicating class and
 * every one of them must have an outgoing transition. Otherwise the chain is not ergodic (giddy
 * raises or returns several distributions) and the row is NaN.
 *
 * Counts are exact; the statistics are f32 sums, accurate to about 1e-5 relative.
 */
export class GPUTransitionDiagnostics implements GPUCommandNodeProducer {
  /** Prefix for every node ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUTransitionDiagnosticsProps;
  /** Resolved condition class count. */
  readonly conditionCount: number;
  /** Number of steady-state chains: 1, or `conditionCount + 1` with conditions. */
  readonly chainCount: number;
  /** Resolved minimum transitions per state. */
  readonly minimumTransitions: number;

  constructor(props: GPUTransitionDiagnosticsProps) {
    const id = props.id ?? 'transition-diagnostics';
    this.id = id;
    this.props = props;
    this.conditionCount = props.conditionCount ?? 1;
    this.chainCount = this.conditionCount > 1 ? this.conditionCount + 1 : 1;
    this.minimumTransitions =
      props.minimumTransitions ?? GPU_TRANSITION_DEFAULT_MINIMUM_TRANSITIONS;
    const {classCount, output} = props;
    if (!Number.isInteger(classCount) || classCount < 1 || classCount > MAXIMUM_CLASS_COUNT) {
      throw new Error(`${id} classCount must be an integer in [1, ${MAXIMUM_CLASS_COUNT}]`);
    }
    if (!Number.isInteger(this.conditionCount) || this.conditionCount < 1) {
      throw new Error(`${id} conditionCount must be a positive integer`);
    }
    if (!Number.isInteger(this.minimumTransitions) || this.minimumTransitions < 1) {
      throw new Error(`${id} minimumTransitions must be a positive integer`);
    }
    validatePackedUint32View(props.counts, `${id} counts`);
    if (props.counts.length < this.conditionCount * classCount * classCount) {
      throw new Error(`${id} counts must hold conditionCount * classCount^2 rows`);
    }
    if (output.homogeneity) {
      validatePackedView(output.homogeneity, ['float32'], `${id} output.homogeneity`);
      if (output.homogeneity.length < GPU_TRANSITION_HOMOGENEITY_LENGTH) {
        throw new Error(
          `${id} output.homogeneity must hold ${GPU_TRANSITION_HOMOGENEITY_LENGTH} rows`
        );
      }
      if (this.conditionCount < 2) {
        // A homogeneity test compares at least two conditional matrices.
        throw new Error(`${id} output.homogeneity needs conditionCount of at least 2`);
      }
    }
    if (output.steadyStates) {
      validatePackedView(output.steadyStates, ['float32'], `${id} output.steadyStates`);
      if (output.steadyStates.length < this.chainCount * classCount) {
        throw new Error(`${id} output.steadyStates must hold chains * classCount rows`);
      }
    }
    if (output.stateSparsity) {
      validatePackedUint32View(output.stateSparsity, `${id} output.stateSparsity`);
      if (output.stateSparsity.length < GPU_TRANSITION_STATE_SPARSITY_LENGTH) {
        throw new Error(
          `${id} output.stateSparsity must hold ${GPU_TRANSITION_STATE_SPARSITY_LENGTH} rows`
        );
      }
    }
    if (output.status) {
      validateGPUStatusPort(id, output.status);
      if (output.status.converged && !output.steadyStates) {
        throw new Error(`${id} output.status.converged needs output.steadyStates`);
      }
    }
    const outputs = [
      output.homogeneity,
      output.steadyStates,
      output.stateSparsity,
      output.status?.converged
    ];
    validateGraphOutputsDisjointFromInputs(id, outputs, [props.counts]);
    const buffers = outputs.filter(view => view !== undefined).map(view => view.buffer);
    if (new Set(buffers).size !== buffers.length) {
      throw new Error(`${id} outputs must use separate buffers`);
    }
  }

  /** Returns the summary node (homogeneity, sparsity, status reset) and the steady-state node. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, conditionCount, chainCount, minimumTransitions} = this;
    const {counts, classCount, output} = props;
    const converged = output.status?.converged;
    validateGraphViewsBelongToGraph(id, graph, [
      counts,
      output.homogeneity,
      output.steadyStates,
      output.stateSparsity,
      converged
    ]);
    const constants = `const K: u32 = ${classCount}u;
const C: u32 = ${conditionCount}u;
const MINIMUM_TRANSITIONS: u32 = ${minimumTransitions}u;`;
    const nodes: GPUCommandNode<Parameters>[] = [];
    if (output.homogeneity || output.stateSparsity || converged) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-summary`,
          operation: OPERATION,
          variant: 'summary',
          bindings: [
            {name: 'counts', view: counts, type: 'u32', access: 'read'},
            ...(output.homogeneity
              ? [
                  {
                    name: 'homogeneity',
                    view: output.homogeneity,
                    type: 'f32' as const,
                    access: 'read_write' as const
                  }
                ]
              : []),
            ...(output.stateSparsity
              ? [
                  {
                    name: 'stateSparsity',
                    view: output.stateSparsity,
                    type: 'u32' as const,
                    access: 'read_write' as const
                  }
                ]
              : []),
            ...(converged
              ? [
                  {
                    name: 'converged',
                    view: converged,
                    type: 'u32' as const,
                    access: 'read_write' as const
                  }
                ]
              : [])
          ],
          invocationCount: 1,
          declarations: `${constants}
${GET_NAN_WGSL}
${IS_NAN_BITS_WGSL}
${GPU_UPPER_INCOMPLETE_GAMMA_WGSL}`,
          body: `${converged ? 'converged[convergedOffset] = 1u;' : ''}
  ${
    output.stateSparsity
      ? `var sparse = 0u;
  var empty = 0u;
  for (var state = 0u; state < C * K; state++) {
    var stateTotal = 0u;
    for (var to = 0u; to < K; to++) {
      stateTotal += counts[countsOffset + state * K + to];
    }
    if (stateTotal == 0u) {
      empty += 1u;
    } else if (stateTotal < MINIMUM_TRANSITIONS) {
      sparse += 1u;
    }
  }
  stateSparsity[stateSparsityOffset + ${GPU_TRANSITION_STATE_SPARSITY.SPARSE}u] = sparse;
  stateSparsity[stateSparsityOffset + ${GPU_TRANSITION_STATE_SPARSITY.EMPTY}u] = empty;`
      : ''
  }
  ${
    output.homogeneity
      ? `var q = 0.0;
  var likelihoodRatio = 0.0;
  var degreesOfFreedom = 0.0;
  var pooledRow: array<u32, ${classCount}>;
  for (var origin = 0u; origin < K; origin++) {
    var pooledTotal = 0u;
    var nonZeroCells = 0u;
    for (var to = 0u; to < K; to++) {
      var pooled = 0u;
      for (var condition = 0u; condition < C; condition++) {
        pooled += counts[countsOffset + (condition * K + origin) * K + to];
      }
      pooledRow[to] = pooled;
      pooledTotal += pooled;
      if (pooled > 0u) {
        nonZeroCells += 1u;
      }
    }
    if (pooledTotal == 0u) {
      continue;
    }
    var regimes = 0u;
    for (var condition = 0u; condition < C; condition++) {
      var regimeTotal = 0u;
      for (var to = 0u; to < K; to++) {
        regimeTotal += counts[countsOffset + (condition * K + origin) * K + to];
      }
      if (regimeTotal == 0u) {
        continue;
      }
      regimes += 1u;
      for (var to = 0u; to < K; to++) {
        let observed = f32(counts[countsOffset + (condition * K + origin) * K + to]);
        let expected = f32(regimeTotal) * f32(pooledRow[to]) / f32(pooledTotal);
        if (pooledRow[to] > 0u) {
          q += (observed - expected) * (observed - expected) / expected;
        }
        if (observed > 0.0) {
          likelihoodRatio += observed * log(observed / expected);
        }
      }
    }
    degreesOfFreedom += f32(regimes - 1u) * f32(nonZeroCells - 1u);
  }
  likelihoodRatio = max(2.0 * likelihoodRatio, 0.0);
  let nan = getNaN();
  homogeneity[homogeneityOffset + ${GPU_TRANSITION_HOMOGENEITY.Q}u] = q;
  homogeneity[homogeneityOffset + ${GPU_TRANSITION_HOMOGENEITY.Q_P_VALUE}u] =
    select(nan, getChiSquareSurvivalValue(q, degreesOfFreedom), degreesOfFreedom > 0.0);
  homogeneity[homogeneityOffset + ${GPU_TRANSITION_HOMOGENEITY.LR}u] = likelihoodRatio;
  homogeneity[homogeneityOffset + ${GPU_TRANSITION_HOMOGENEITY.LR_P_VALUE}u] =
    select(nan, getChiSquareSurvivalValue(likelihoodRatio, degreesOfFreedom), degreesOfFreedom > 0.0);
  homogeneity[homogeneityOffset + ${GPU_TRANSITION_HOMOGENEITY.DEGREES_OF_FREEDOM}u] = degreesOfFreedom;`
      : ''
  }`
        })
      );
    }
    if (output.steadyStates) {
      // The elimination matrix of every chain (K * K floats each) lives in storage: at 64 classes
      // it would take 16 KiB of private memory per invocation, twice the core limit of 8,192 bytes.
      const chainWork = createTransientView(
        graph,
        `${id}-chain-work`,
        'float32',
        chainCount * classCount * classCount
      );
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-steady-states`,
          operation: OPERATION,
          variant: 'steady-states',
          bindings: [
            {name: 'counts', view: counts, type: 'u32', access: 'read'},
            {
              name: 'steadyStates',
              view: output.steadyStates,
              type: 'f32',
              access: 'read_write'
            },
            {name: 'chainWork', view: chainWork, type: 'f32', access: 'read_write'},
            ...(converged
              ? [
                  {
                    name: 'converged',
                    view: converged,
                    type: 'atomic<u32>' as const,
                    access: 'read_write' as const
                  }
                ]
              : [])
          ],
          invocationCount: chainCount,
          declarations: `${constants}
${GET_NAN_WGSL}
// Private memory: 2 * K words (at most 512 bytes), far below the 8,192-byte core limit.
var<private> usedStates: array<u32, ${classCount}>;
var<private> solution: array<f32, ${classCount}>;

// Transition count of one chain: chain 0 is the pooled chain when there are conditions.
fn getChainCount(chain: u32, origin: u32, to: u32) -> u32 {
  if (C == 1u) {
    return counts[countsOffset + origin * K + to];
  }
  if (chain == 0u) {
    var pooled = 0u;
    for (var condition = 0u; condition < C; condition++) {
      pooled += counts[countsOffset + (condition * K + origin) * K + to];
    }
    return pooled;
  }
  return counts[countsOffset + ((chain - 1u) * K + origin) * K + to];
}`,
          body: `let chain = index;
  let workBase = chainWorkOffset + chain * K * K;
  let nan = getNaN();
  var activeCount = 0u;
  var solvable = true;
  for (var state = 0u; state < K; state++) {
    var outgoing = 0u;
    var incoming = 0u;
    for (var other = 0u; other < K; other++) {
      outgoing += getChainCount(chain, state, other);
      incoming += getChainCount(chain, other, state);
    }
    if (outgoing > 0u || incoming > 0u) {
      usedStates[activeCount] = state;
      activeCount += 1u;
      if (outgoing == 0u) {
        solvable = false;
      }
    }
  }
  if (activeCount == 0u) {
    // No transitions at all: nothing to solve, and not a failure of the solver.
    for (var state = 0u; state < K; state++) {
      steadyStates[steadyStatesOffset + chain * K + state] = nan;
    }
    return;
  }
  let m = activeCount;
  if (solvable) {
    for (var a = 0u; a < m; a++) {
      var rowTotal = 0u;
      for (var b = 0u; b < m; b++) {
        rowTotal += getChainCount(chain, usedStates[a], usedStates[b]);
      }
      for (var b = 0u; b < m; b++) {
        chainWork[workBase + a * m + b] = f32(getChainCount(chain, usedStates[a], usedStates[b])) / f32(rowTotal);
      }
    }
    // Grassmann-Taksar-Heyman: eliminate the last state repeatedly. The sum of the remaining
    // outgoing probabilities of a state is zero only when it cannot reach the lower states.
    for (var k = m - 1u; k >= 1u; k--) {
      var escape = 0.0;
      for (var j = 0u; j < k; j++) {
        escape += chainWork[workBase + k * m + j];
      }
      if (!(escape > 0.0)) {
        solvable = false;
        break;
      }
      for (var i = 0u; i < k; i++) {
        chainWork[workBase + i * m + k] = chainWork[workBase + i * m + k] / escape;
      }
      for (var i = 0u; i < k; i++) {
        for (var j = 0u; j < k; j++) {
          chainWork[workBase + i * m + j] = chainWork[workBase + i * m + j] + chainWork[workBase + i * m + k] * chainWork[workBase + k * m + j];
        }
      }
    }
  }
  if (solvable) {
    solution[0] = 1.0;
    var total = 1.0;
    for (var k = 1u; k < m; k++) {
      var weight = 0.0;
      for (var i = 0u; i < k; i++) {
        weight += solution[i] * chainWork[workBase + i * m + k];
      }
      solution[k] = weight;
      total += weight;
    }
    for (var state = 0u; state < K; state++) {
      steadyStates[steadyStatesOffset + chain * K + state] = 0.0;
    }
    for (var a = 0u; a < m; a++) {
      steadyStates[steadyStatesOffset + chain * K + usedStates[a]] = solution[a] / total;
    }
  } else {
    for (var state = 0u; state < K; state++) {
      steadyStates[steadyStatesOffset + chain * K + state] = nan;
    }
    ${converged ? 'atomicMin(&converged[convergedOffset], 0u);' : ''}
  }`
        })
      );
    }
    return nodes;
  }
}
