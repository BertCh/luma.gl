// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {createWGSLKernelNode, getWGSLFloatLiteral} from '../../utils/wgsl-kernel-nodes';
import {
  validateCapacity,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validatePackedView
} from '../../utils/gpu-contributor-utils';
import {
  type GPUSpatialWeights,
  validateGPUSpatialWeights
} from '../spatial-weights/spatial-weights';
import {GET_NAN_WGSL} from '../../utils/wgsl-common';
import {getGPUSegregationLayout} from './segregation-layout';

/**
 * WGSL accessors of one scale's `environment` table: `units * K` weighted group counts, then
 * `units` environment totals, then the scale index (as f32). Expects an `environment` binding.
 */
const ENVIRONMENT_LAYOUT_WGSL = (unitCount: number, groupCount: number) => `
const ENVIRONMENT_TOTALS: u32 = ${unitCount * groupCount}u;
const ENVIRONMENT_SCALE: u32 = ${unitCount * groupCount + unitCount}u;
fn getEnvironmentTotal(unit: u32) -> f32 {
  return environment[environmentOffset + ENVIRONMENT_TOTALS + unit];
}
fn getScaleIndex() -> u32 {
  return u32(environment[environmentOffset + ENVIRONMENT_SCALE]);
}`;

const OPERATION = 'GPUSegregation';
/** Lanes of the reduction workgroups (one workgroup per block of units, or per output column). */
const LANES = 256;
/** Columns a block-reduction round reduces together, so each barrier round serves several terms. */
const COLUMN_BATCH = 8;

/** Largest supported number of groups. */
export const GPU_SEGREGATION_MAXIMUM_GROUPS = 16;

/**
 * Caller-owned outputs of {@link GPUSegregation}: the global `indices` and optional per-unit
 * (local) outputs, each holding one block per scale.
 */
export type GPUSegregationOutput = {
  /**
   * `scales.length * layout.stride` global indices; scale `s` starts at `s * stride`. See
   * `getGPUSegregationLayout`.
   */
  indices: GraphDataView<'float32'>;
  /**
   * `scales * units * K` local composition `pi_im` (share of group `m` in the environment of unit
   * `i`), index `(scale * units + i) * K + m`. Environment shares sum to 1 for a populated unit.
   */
  localEnvironment?: GraphDataView<'float32'>;
  /** `scales * units` local entropy `E_i = sum_m pi_im ln(1 / pi_im)` of the environment, in nats. */
  localEntropy?: GraphDataView<'float32'>;
  /**
   * `scales * units * K` unit contributions to `D_g`: `t_i |pi_ig - P_g| / (2 T P_g (1 - P_g))`.
   * The sum over units is the dissimilarity `D_g` of the same scale.
   */
  localDissimilarity?: GraphDataView<'float32'>;
  /**
   * `scales * units` unit contributions to H: `t_i (E - E_i) / (E T)`. The sum over units is the
   * entropy index H of the same scale.
   */
  localTheil?: GraphDataView<'float32'>;
};

/**
 * Properties for {@link GPUSegregation}.
 *
 * Compile-time: the unit count (`groupCounts.length / groupCount`), `groupCount`, the scales
 * (their number and weights structure), `selfWeight`, `atkinsonInequalityAversion` and which
 * optional outputs exist. Per-frame: the contents of
 * `groupCounts` and of every weights view.
 */
export type GPUSegregationProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'segregation'`. */
  id?: string;
  /** Number of population groups `K`, in `[2, GPU_SEGREGATION_MAXIMUM_GROUPS]`. */
  groupCount: number;
  /**
   * Population per unit and group, `units * K` float32 values, row-major
   * (`groupCounts[i * K + m]`); the unit count (rows, for example census tracts or grid cells) is
   * `groupCounts.length / K`. Negative and non-finite entries count as 0.
   */
  groupCounts: GraphDataView<'float32'>;
  /**
   * The scales of the profile. Each is a square {@link GPUSpatialWeights} over the units (a kernel
   * or distance band built by `GPUNeighborSearch` and reweighted with `GPUSpatialWeightsTransform`,
   * one per bandwidth) defining the local environment of every unit, or `null` for the aspatial
   * indices. Defaults to `[null]`. Entry `s` of the result is the multiscale profile at scale `s`.
   */
  scales?: readonly (GPUSpatialWeights | null)[];
  /**
   * Weight of a unit in its own environment, added to the neighbor weights (which never list the
   * unit itself). Defaults to 1; use 0 to exclude it.
   */
  selfWeight?: number;
  /**
   * How a spatial scale uses its environment. `'environment'` (default, Reardon and O'Sullivan):
   * the sums keep the actual unit populations `t_i` and totals `T`, and only the compositions
   * `pi_im` come from the environment. `'smoothed-population'` (what PySAL's spatially implicit
   * indices do): the smoothed table `x~_im` replaces the population table altogether, so the
   * aspatial formulas run on it with `t~_i`, `T~ = sum t~_i` and `P~_m`. PySAL also rounds the
   * smoothed counts to integers; this contributor does not. Ignored for the aspatial scale.
   */
  spatialForm?: 'environment' | 'smoothed-population';
  /** Atkinson inequality-aversion parameter `b`, in the open interval (0, 1). Defaults to 0.5. */
  atkinsonInequalityAversion?: number;
  /** Caller-owned outputs. */
  output: GPUSegregationOutput;
};

/**
 * Residential segregation indices over population groups per unit (the PySAL `segregation`
 * package), aspatial or spatial, global or local, and as a multiscale profile.
 *
 * Notation. Unit `i` has group counts `x_im` and total `t_i = sum_m x_im`; `X_m = sum_i x_im`,
 * `T = sum_i t_i`, `P_m = X_m / T`. A scale's environment of unit `i` is the weighted population
 * `x~_im = s x_im + sum_j w_ij x_jm` and `t~_i = sum_m x~_im` (`s` is `selfWeight`), with
 * composition `pi_im = x~_im / t~_i`. The aspatial scale (`null` weights) uses `pi_im = x_im / t_i`,
 * the usual unit composition. The spatial indices are the Reardon and O'Sullivan (2004) forms: every
 * index below uses `pi_im` in place of `p_im = x_im / t_i`, while the weights of the sums stay the
 * unit populations. A zero total environment contributes nothing.
 *
 * Global indices per scale (columns in `getGPUSegregationLayout`):
 * - Entropy H (Theil, multigroup information-theory index):
 *   `H = sum_i t_i (E - E_i) / (E T)`, `E = sum_m P_m ln(1 / P_m)`, `E_i = sum_m pi_im ln(1 / pi_im)`.
 * - Multigroup dissimilarity D (Reardon and Firebaugh, PySAL `MultiDissim`):
 *   `D = sum_i sum_m t_i |pi_im - P_m| / (2 T I)`, `I = sum_m P_m (1 - P_m)`.
 * - Dissimilarity of group `g` against all others (PySAL `Dissim`; for the aspatial scale it equals
 *   `1/2 sum_i |x_ig / X_g - (t_i - x_ig) / (T - X_g)|`):
 *   `D_g = sum_i t_i |pi_ig - P_g| / (2 T P_g (1 - P_g))`.
 * - Isolation `xPx_g = sum_i (x_ig / X_g) pi_ig` and interaction `xPy_gh = sum_i (x_ig / X_g) pi_ih`
 *   (PySAL `Isolation`, `Interaction`; Lieberson). `xPx_g + sum_{h != g} xPy_gh = 1`.
 * - Atkinson of group `g` (PySAL `Atkinson`, parameter `b`):
 *   `A_g = 1 - P_g / (1 - P_g) * | sum_i (1 - pi_ig)^(1 - b) pi_ig^b t_i / (P_g T) |^(1 / (1 - b))`.
 * Indices that are undefined are written as NaN, never 0 (0 means perfectly even): H and the
 * diversity `E` with an empty total (H also with `E = 0`, a single present group), D with an empty
 * total or `I = 0`, `D_g` and `A_g` for a group with zero or all of the population (or an empty
 * total), and isolation / interaction of a group that has no population. The diversity `E` is 0,
 * a defined value, when only one group is present.
 *
 * Local indices (optional `output.local*` views) are the per-unit terms of these sums, so they add up to
 * the global index of the same scale: environment composition, local entropy, and the unit
 * contributions to `D_g` and to H. A unit whose environment is empty (no population in it or its
 * neighbors) has no composition: all its local values are NaN and it contributes nothing to the
 * global sums, so sum local values with a NaN-skipping sum to reproduce the globals. The unit
 * contributions to `D_g` and H are also NaN when that global index is undefined. Multiscale: one global result per scale, from the aspatial
 * scale through growing bandwidths, shows at which distance the segregation arises.
 *
 * Algorithm: one fused pass over each CSR for the environments (this does not compose
 * `GPUNeighborhoodSummary`: it summarizes one column per pass and writes packed columns, so K
 * groups would cost K passes plus interleave copies), then one kernel that computes every per-unit
 * term of the sums and reduces them in the same workgroup (shared-memory trees, `COLUMN_BATCH`
 * terms per barrier round) into one partial per (term, block of 256 units), and one fixed-order
 * tree per term over the block partials (bitwise reproducible). The `terms x units` table is never
 * materialized, so the sums cost one read of the environment instead of a write and a read of
 * `(2 + 2K + K^2)` columns. Cost per scale is O(nonzeros * K + units * K^2).
 * Precision is float32: indices are sums over units, so expect about 1e-5 relative error against
 * a float64 reference for moderate populations. Counts and totals are float32, so integer
 * populations are exact only up to 2^24 (16,777,216) in total; above that the totals `T` and
 * `X_m` round and shares carry a relative error near 6e-8 per addition.
 *
 * Small-sample bias. The indices are computed from the observed counts and are not bias
 * corrected: `D` and `H` are biased upward when units are small (few people per unit), because
 * sampling noise alone makes unit compositions differ from the global one even when people are
 * assigned to units at random (Cortese, Falk and Cohen 1976; Winship 1977; Carrington and Troske
 * 1997). Compare units of similar size, or against a random-allocation null, before reading small
 * `D` / `H` values as segregation; the isolation and interaction indices are less affected but
 * depend on the group share. Permutation / bootstrap inference is not part of this operator.
 */
export class GPUSegregation implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUSegregationProps;
  /** Number of scales. */
  readonly scaleCount: number;
  /** Number of units, `groupCounts.length / groupCount`. */
  readonly unitCount: number;

  constructor(props: GPUSegregationProps) {
    const id = props.id ?? 'segregation';
    this.id = id;
    this.props = props;
    const {groupCount} = props;
    validateCapacity(id, 'groupCount', groupCount, 2, GPU_SEGREGATION_MAXIMUM_GROUPS);
    const scales = props.scales ?? [null];
    if (scales.length < 1) {
      throw new Error(`${id} scales must not be empty`);
    }
    this.scaleCount = scales.length;
    validatePackedView(props.groupCounts, ['float32'], `${id} groupCounts`);
    const unitCount = props.groupCounts.length / groupCount;
    if (!Number.isInteger(unitCount) || unitCount < 1) {
      throw new Error(`${id} groupCounts length must be a positive multiple of groupCount`);
    }
    this.unitCount = unitCount;
    scales.forEach((weights, scaleIndex) => {
      if (
        weights &&
        validateGPUSpatialWeights(id, weights, `scales[${scaleIndex}]`) !== unitCount
      ) {
        throw new Error(`${id} scales[${scaleIndex}] must have one row per unit`);
      }
    });
    const b = props.atkinsonInequalityAversion ?? 0.5;
    if (!(b > 0 && b < 1)) {
      throw new Error(`${id} atkinsonInequalityAversion must be in (0, 1)`);
    }
    const selfWeight = props.selfWeight ?? 1;
    if (!Number.isFinite(selfWeight) || selfWeight < 0) {
      throw new Error(`${id} selfWeight must be finite and non-negative`);
    }
    const layout = getGPUSegregationLayout(groupCount);
    const {output} = props;
    const expectedLengths: [string, GraphDataView<'float32'> | undefined, number][] = [
      ['output.indices', output.indices, this.scaleCount * layout.stride],
      [
        'output.localEnvironment',
        output.localEnvironment,
        this.scaleCount * unitCount * groupCount
      ],
      ['output.localEntropy', output.localEntropy, this.scaleCount * unitCount],
      [
        'output.localDissimilarity',
        output.localDissimilarity,
        this.scaleCount * unitCount * groupCount
      ],
      ['output.localTheil', output.localTheil, this.scaleCount * unitCount]
    ];
    for (const [name, view, length] of expectedLengths) {
      if (view) {
        validatePackedView(view, ['float32'], `${id} ${name}`);
        if (view.length !== length) {
          throw new Error(`${id} ${name} must contain ${length} rows`);
        }
      }
    }
    validateGraphOutputsDisjointFromInputs(id, this.getOutputViews(), this.getInputViews());
  }

  private getInputViews(): (GraphDataView | undefined)[] {
    const views: (GraphDataView | undefined)[] = [this.props.groupCounts];
    for (const weights of this.props.scales ?? []) {
      if (weights) {
        views.push(weights.offsets, weights.neighbors, weights.weights);
      }
    }
    return views;
  }

  private getOutputViews(): (GraphDataView | undefined)[] {
    const {output} = this.props;
    return [
      output.indices,
      output.localEnvironment,
      output.localEntropy,
      output.localDissimilarity,
      output.localTheil
    ];
  }

  /** Returns the totals, environment, term, reduction and finalize nodes in dependency order. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, unitCount} = this;
    validateGraphViewsBelongToGraph(id, graph, [...this.getInputViews(), ...this.getOutputViews()]);
    const {groupCount: K, groupCounts, output} = props;
    const {indices} = output;
    const local = {
      environment: output.localEnvironment,
      entropy: output.localEntropy,
      dissimilarity: output.localDissimilarity,
      theil: output.localTheil
    };
    const scales = props.scales ?? [null];
    const selfWeight = props.selfWeight ?? 1;
    const b = props.atkinsonInequalityAversion ?? 0.5;
    const layout = getGPUSegregationLayout(K);
    const termCount = 2 + 2 * K + K * K;
    const nodes: GPUCommandNode<Parameters>[] = [];

    const declarations = `${GET_NAN_WGSL}
const UNITS: u32 = ${unitCount}u;
const K: u32 = ${K}u;
const TERM_COUNT: u32 = ${termCount}u;
const ATKINSON_B: f32 = ${getWGSLFloatLiteral(b)};
const SELF_WEIGHT: f32 = ${getWGSLFloatLiteral(selfWeight)};
fn countAt(unit: u32, group: u32) -> f32 {
  let value = groupCounts[groupCountsOffset + unit * K + group];
  // Negative, NaN and infinite counts are 0.
  let isFinite = (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u;
  return select(0.0, value, isFinite && value > 0.0);
}
fn unitTotal(unit: u32) -> f32 {
  var total = 0.0;
  for (var group = 0u; group < K; group++) {
    total += countAt(unit, group);
  }
  return total;
}
fn entropyTerm(share: f32) -> f32 {
  return select(0.0, -share * log(share), share > 0.0);
}`;

    // Block partials and fixed-order column totals, both fused tree reductions. A kernel that
    // fills `columnValues` for its unit and calls `reduceColumns(lane, group)` writes one partial
    // per (column, block of units); `addColumnTotals` then sums each column's partials.
    const unitGroups = Math.ceil(unitCount / LANES);
    const reductionDeclarations = (columns: number, base: string) => `${base}
const LANES: u32 = ${LANES}u;
const GROUPS: u32 = ${unitGroups}u;
const COLUMNS: u32 = ${columns}u;
const COLUMN_BATCH: u32 = ${COLUMN_BATCH}u;
var<private> columnValues: array<f32, ${columns}>;
var<workgroup> batchScratch: array<f32, ${COLUMN_BATCH * LANES}>;
fn reduceColumns(lane: u32, group: u32) {
  for (var first = 0u; first < COLUMNS; first += COLUMN_BATCH) {
    workgroupBarrier();
    for (var slot = 0u; slot < COLUMN_BATCH; slot++) {
      let column = first + slot;
      batchScratch[slot * LANES + lane] = select(0.0, columnValues[min(column, COLUMNS - 1u)], column < COLUMNS);
    }
    workgroupBarrier();
    for (var stride = LANES / 2u; stride > 0u; stride = stride / 2u) {
      if (lane < stride) {
        for (var slot = 0u; slot < COLUMN_BATCH; slot++) {
          batchScratch[slot * LANES + lane] += batchScratch[slot * LANES + lane + stride];
        }
      }
      workgroupBarrier();
    }
    if (lane < COLUMN_BATCH && first + lane < COLUMNS) {
      partials[partialsOffset + (first + lane) * GROUPS + group] = batchScratch[lane * LANES];
    }
  }
}`;
    const addColumnTotals = (
      label: string,
      partials: GraphDataView<'float32'>,
      columns: number,
      output: GraphDataView<'float32'>,
      // Copies the totals into row `scale` of `view` (rows of `stride` values); the scale index is
      // read from the scale's environment table (rule 12: no scale index in the WGSL text).
      collected?: {
        view: GraphDataView<'float32'>;
        stride: number;
        environment: GraphDataView<'float32'>;
      }
    ) =>
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-${label}`,
        operation: OPERATION,
        variant: 'column-totals',
        bindings: [
          {name: 'partials', view: partials, type: 'f32', access: 'read'},
          {name: 'output', view: output, type: 'f32', access: 'read_write'},
          ...(collected
            ? [
                {
                  name: 'collected',
                  view: collected.view,
                  type: 'f32' as const,
                  access: 'read_write' as const
                },
                {
                  name: 'environment',
                  view: collected.environment,
                  type: 'f32' as const,
                  access: 'read' as const
                }
              ]
            : [])
        ],
        invocationCount: columns * LANES,
        workgroupSize: LANES,
        guardIndex: false,
        declarations: `const LANES: u32 = ${LANES}u;
const GROUPS: u32 = ${unitGroups}u;
${
  collected
    ? `const COLLECTED_STRIDE: u32 = ${collected.stride}u;
${ENVIRONMENT_LAYOUT_WGSL(unitCount, K)}`
    : ''
}
var<workgroup> scratch: array<f32, ${LANES}>;`,
        // One workgroup per column; lane l sums partials l, l + 256, ... then a fixed tree.
        body: `let column = index / LANES;
  let lane = localInvocationIndex;
  var sum = 0.0;
  for (var group = lane; group < GROUPS; group += LANES) {
    sum += partials[partialsOffset + column * GROUPS + group];
  }
  scratch[lane] = sum;
  workgroupBarrier();
  for (var stride = LANES / 2u; stride > 0u; stride = stride / 2u) {
    if (lane < stride) {
      scratch[lane] += scratch[lane + stride];
    }
    workgroupBarrier();
  }
  if (lane == 0u) {
    output[outputOffset + column] = scratch[0];
    ${collected ? 'collected[collectedOffset + getScaleIndex() * COLLECTED_STRIDE + column] = scratch[0];' : ''}
  }`
      });

    // Totals X_m and T. Shared by every scale unless `spatialForm` is 'smoothed-population',
    // which totals each scale's smoothed table.
    const allTotals = createTransientView(
      graph,
      `${id}-all-totals`,
      'float32',
      scales.length * (K + 1)
    );
    const smoothedPopulation = props.spatialForm === 'smoothed-population';
    const addTotals = (
      label: string,
      counts: GraphDataView<'float32'>,
      scaleEnvironment?: GraphDataView<'float32'>
    ) => {
      const totalPartials = createTransientView(
        graph,
        `${id}-${label}-partials`,
        'float32',
        (K + 1) * unitGroups
      );
      const totals = createTransientView(graph, `${id}-${label}`, 'float32', K + 1);
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-${label}-partials`,
          operation: OPERATION,
          variant: 'total-partials',
          bindings: [
            {name: 'groupCounts', view: counts, type: 'f32', access: 'read'},
            {name: 'partials', view: totalPartials, type: 'f32', access: 'read_write'}
          ],
          invocationCount: unitGroups * LANES,
          workgroupSize: LANES,
          guardIndex: false,
          declarations: reductionDeclarations(K + 1, declarations),
          body: `let lane = localInvocationIndex;
  let group = (index - lane) / LANES;
  if (index < UNITS) {
    for (var column = 0u; column < K; column++) {
      columnValues[column] = countAt(index, column);
    }
    columnValues[K] = unitTotal(index);
  }
  reduceColumns(lane, group);`
        }),
        addColumnTotals(
          label,
          totalPartials,
          K + 1,
          totals,
          scaleEnvironment
            ? {view: allTotals, stride: K + 1, environment: scaleEnvironment}
            : undefined
        )
      );
      return totals;
    };
    const collectTotals = (
      scaleId: string,
      totals: GraphDataView<'float32'>,
      environment: GraphDataView<'float32'>
    ) =>
      createWGSLKernelNode<Parameters>(graph, {
        id: `${scaleId}-collect-totals`,
        operation: OPERATION,
        variant: 'collect-totals',
        bindings: [
          {name: 'totals', view: totals, type: 'f32', access: 'read'},
          {name: 'allTotals', view: allTotals, type: 'f32', access: 'read_write'},
          {name: 'environment', view: environment, type: 'f32', access: 'read'}
        ],
        invocationCount: K + 1,
        declarations: `const K: u32 = ${K}u;
${ENVIRONMENT_LAYOUT_WGSL(unitCount, K)}`,
        body: `allTotals[allTotalsOffset + getScaleIndex() * (K + 1u) + index] = totals[totalsOffset + index];`
      });
    const sharedTotals = smoothedPopulation ? undefined : addTotals('totals', groupCounts);

    // The composition share and environment helpers shared by the term and local kernels.
    const environmentDeclarations = `${declarations}
${ENVIRONMENT_LAYOUT_WGSL(unitCount, K)}
fn getShare(unit: u32, group: u32) -> f32 {
  let total = getEnvironmentTotal(unit);
  return select(0.0, environment[environmentOffset + unit * K + group] / total, total > 0.0);
}
fn getPopulationShare(group: u32) -> f32 {
  let total = totals[totalsOffset + K];
  return select(0.0, totals[totalsOffset + group] / total, total > 0.0);
}
fn getDiversity() -> f32 {
  var diversity = 0.0;
  for (var group = 0u; group < K; group++) {
    diversity += entropyTerm(getPopulationShare(group));
  }
  return diversity;
}`;

    const allSums = createTransientView(
      graph,
      `${id}-term-sums`,
      'float32',
      scales.length * termCount
    );
    // Scale cursor: the environment kernel of each scale advances it and records the scale index
    // in its environment table, which every later kernel of the scale reads.
    const scaleCursor = createTransientView(graph, `${id}-scale-cursor`, 'uint32', 1);
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-scale-cursor-clear`,
        operation: OPERATION,
        variant: 'scale-cursor-clear',
        bindings: [{name: 'scaleCursor', view: scaleCursor, type: 'u32', access: 'read_write'}],
        invocationCount: 1,
        body: 'scaleCursor[scaleCursorOffset] = 0xffffffffu;'
      })
    );
    scales.forEach((weights, scaleIndex) => {
      const scaleId = `${id}-scale-${scaleIndex}`;
      // Layout: unitCount * K weighted group counts, unitCount totals, then the scale index.
      const environment = createTransientView(
        graph,
        `${scaleId}-environment`,
        'float32',
        unitCount * K + unitCount + 1
      );
      const sums = createTransientView(graph, `${scaleId}-sums`, 'float32', termCount);
      const termPartials = createTransientView(
        graph,
        `${scaleId}-term-partials`,
        'float32',
        termCount * unitGroups
      );
      // In the smoothed-population form the indices treat the smoothed table as the population.
      const counts = smoothedPopulation ? environment : groupCounts;
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${scaleId}-environment`,
          operation: OPERATION,
          variant: weights ? 'environment' : 'environment-aspatial',
          bindings: [
            {name: 'groupCounts', view: groupCounts, type: 'f32', access: 'read'},
            ...(weights
              ? [
                  {
                    name: 'offsets',
                    view: weights.offsets,
                    type: 'u32' as const,
                    access: 'read' as const
                  },
                  {
                    name: 'neighbors',
                    view: weights.neighbors,
                    type: 'u32' as const,
                    access: 'read' as const
                  },
                  {
                    name: 'weights',
                    view: weights.weights,
                    type: 'f32' as const,
                    access: 'read' as const
                  }
                ]
              : []),
            {name: 'environment', view: environment, type: 'f32', access: 'read_write'},
            {name: 'scaleCursor', view: scaleCursor, type: 'u32', access: 'read_write'}
          ],
          invocationCount: unitCount,
          declarations,
          // Only invocation 0 touches the cursor. The scale index is small, so f32 holds it exactly.
          body: `if (index == 0u) {
    let scale = scaleCursor[scaleCursorOffset] + 1u;
    scaleCursor[scaleCursorOffset] = scale;
    environment[environmentOffset + UNITS * K + UNITS] = f32(scale);
  }
  ${
    weights
      ? `var total = 0.0;
  for (var group = 0u; group < K; group++) {
    var sum = SELF_WEIGHT * countAt(index, group);
    for (var slot = offsets[offsetsOffset + index]; slot < offsets[offsetsOffset + index + 1u]; slot++) {
      let neighbor = neighbors[neighborsOffset + slot];
      if (neighbor < UNITS) {
        sum += weights[weightsOffset + slot] * countAt(neighbor, group);
      }
    }
    environment[environmentOffset + index * K + group] = sum;
    total += sum;
  }
  environment[environmentOffset + UNITS * K + index] = total;`
      : `for (var group = 0u; group < K; group++) {
    environment[environmentOffset + index * K + group] = countAt(index, group);
  }
  environment[environmentOffset + UNITS * K + index] = unitTotal(index);`
  }`
        })
      );
      const scaleTotals =
        sharedTotals ?? addTotals(`scale-${scaleIndex}-totals`, counts, environment);
      nodes.push(
        ...(sharedTotals ? [collectTotals(scaleId, scaleTotals, environment)] : []),
        createWGSLKernelNode<Parameters>(graph, {
          id: `${scaleId}-terms`,
          operation: OPERATION,
          variant: 'terms',
          bindings: [
            {name: 'groupCounts', view: counts, type: 'f32', access: 'read'},
            {name: 'environment', view: environment, type: 'f32', access: 'read'},
            {name: 'totals', view: scaleTotals, type: 'f32', access: 'read'},
            {name: 'partials', view: termPartials, type: 'f32', access: 'read_write'}
          ],
          invocationCount: unitGroups * LANES,
          workgroupSize: LANES,
          guardIndex: false,
          declarations: reductionDeclarations(termCount, environmentDeclarations),
          body: `let lane = localInvocationIndex;
  let group = (index - lane) / LANES;
  // A unit whose environment is empty has no composition: it contributes nothing to any sum.
  if (index < UNITS && getEnvironmentTotal(index) > 0.0) {
    let unitPopulation = unitTotal(index);
    var entropy = 0.0;
    var absolute = 0.0;
    for (var member = 0u; member < K; member++) {
      let share = getShare(index, member);
      let deviation = abs(share - getPopulationShare(member));
      entropy += entropyTerm(share);
      absolute += deviation;
      columnValues[2u + member] = unitPopulation * deviation;
      let base = max(1.0 - share, 0.0);
      let atkinson = select(0.0, pow(base, 1.0 - ATKINSON_B) * pow(share, ATKINSON_B), base > 0.0 && share > 0.0);
      columnValues[2u + K + member] = unitPopulation * atkinson;
      for (var other = 0u; other < K; other++) {
        columnValues[2u + 2u * K + member * K + other] = countAt(index, member) * getShare(index, other);
      }
    }
    columnValues[0] = unitPopulation * entropy;
    columnValues[1] = unitPopulation * absolute;
  }
  reduceColumns(lane, group);`
        }),
        addColumnTotals(`${scaleId}-sums`, termPartials, termCount, sums, {
          view: allSums,
          stride: termCount,
          environment
        })
      );
      if (local.environment || local.entropy || local.dissimilarity || local.theil) {
        const outputs = [
          ['localEnvironment', local.environment],
          ['localEntropy', local.entropy],
          ['localDissimilarity', local.dissimilarity],
          ['localTheil', local.theil]
        ] as const;
        nodes.push(
          createWGSLKernelNode<Parameters>(graph, {
            id: `${scaleId}-local`,
            operation: OPERATION,
            variant: `local-${outputs.map(([, view]) => (view ? 1 : 0)).join('')}`,
            bindings: [
              {name: 'groupCounts', view: counts, type: 'f32', access: 'read'},
              {name: 'environment', view: environment, type: 'f32', access: 'read'},
              {name: 'totals', view: scaleTotals, type: 'f32', access: 'read'},
              ...outputs.flatMap(([name, view]) =>
                view ? [{name, view, type: 'f32' as const, access: 'read_write' as const}] : []
              )
            ],
            invocationCount: unitCount,
            declarations: environmentDeclarations,
            body: `let unitPopulation = unitTotal(index);
  let scaleIndex = getScaleIndex();
  // An empty environment has no composition, so every local value of the unit is NaN.
  let hasEnvironment = getEnvironmentTotal(index) > 0.0;
  let populationTotal = totals[totalsOffset + K];
  let diversity = getDiversity();
  var entropy = 0.0;
  for (var group = 0u; group < K; group++) {
    let share = getShare(index, group);
    entropy += entropyTerm(share);
    ${
      local.environment
        ? `localEnvironment[localEnvironmentOffset + (scaleIndex * UNITS + index) * K + group] =
      select(getNaN(), share, hasEnvironment);`
        : ''
    }
    ${
      local.dissimilarity
        ? `let populationShare = getPopulationShare(group);
    let scale = 2.0 * populationTotal * populationShare * (1.0 - populationShare);
    localDissimilarity[localDissimilarityOffset + (scaleIndex * UNITS + index) * K + group] =
      select(getNaN(), unitPopulation * abs(share - populationShare) / scale, scale > 0.0 && hasEnvironment);`
        : ''
    }
  }
  ${
    local.entropy
      ? `localEntropy[localEntropyOffset + scaleIndex * UNITS + index] =
    select(getNaN(), entropy, hasEnvironment);`
      : ''
  }
  ${
    local.theil
      ? `localTheil[localTheilOffset + scaleIndex * UNITS + index] =
    select(getNaN(), unitPopulation * (diversity - entropy) / (diversity * populationTotal), diversity > 0.0 && populationTotal > 0.0 && hasEnvironment);`
      : ''
  }`
          })
        );
      }
    });

    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-finalize`,
        operation: OPERATION,
        variant: 'finalize',
        bindings: [
          {name: 'allTotals', view: allTotals, type: 'f32', access: 'read'},
          {name: 'allSums', view: allSums, type: 'f32', access: 'read'},
          {name: 'indices', view: indices, type: 'f32', access: 'read_write'}
        ],
        invocationCount: scales.length,
        declarations: `${GET_NAN_WGSL}
const K: u32 = ${K}u;
const TERM_COUNT: u32 = ${termCount}u;
const STRIDE: u32 = ${layout.stride}u;
const ATKINSON_B: f32 = ${getWGSLFloatLiteral(b)};
fn entropyTerm(share: f32) -> f32 {
  return select(0.0, -share * log(share), share > 0.0);
}`,
        body: `let populationTotal = allTotals[allTotalsOffset + index * (K + 1u) + K];
  let sums = index * TERM_COUNT;
  let base = index * STRIDE;
  var diversity = 0.0;
  var variance = 0.0;
  for (var group = 0u; group < K; group++) {
    let share = select(0.0, allTotals[allTotalsOffset + index * (K + 1u) + group] / populationTotal, populationTotal > 0.0);
    diversity += entropyTerm(share);
    variance += share * (1.0 - share);
  }
  let hasPopulation = populationTotal > 0.0;
  indices[indicesOffset + base + ${layout.entropy}u] = select(
    getNaN(), 1.0 - allSums[allSumsOffset + sums] / (diversity * populationTotal), hasPopulation && diversity > 0.0);
  indices[indicesOffset + base + ${layout.multiGroupDissimilarity}u] = select(
    getNaN(), allSums[allSumsOffset + sums + 1u] / (2.0 * populationTotal * variance), hasPopulation && variance > 0.0);
  indices[indicesOffset + base + ${layout.diversity}u] = select(getNaN(), diversity, hasPopulation);
  for (var group = 0u; group < K; group++) {
    let groupTotal = allTotals[allTotalsOffset + index * (K + 1u) + group];
    let share = select(0.0, groupTotal / populationTotal, hasPopulation);
    let isMixed = hasPopulation && share > 0.0 && share < 1.0;
    let spread = 2.0 * populationTotal * share * (1.0 - share);
    indices[indicesOffset + base + ${layout.dissimilarity}u + group] = select(
      getNaN(), allSums[allSumsOffset + sums + 2u + group] / spread, isMixed);
    let ratio = abs(allSums[allSumsOffset + sums + 2u + K + group] / max(share * populationTotal, 1e-30));
    let atkinson = 1.0 - share / (1.0 - share) * pow(ratio, 1.0 / (1.0 - ATKINSON_B));
    indices[indicesOffset + base + ${layout.atkinson}u + group] = select(getNaN(), atkinson, isMixed);
    for (var other = 0u; other < K; other++) {
      let exposure = allSums[allSumsOffset + sums + 2u + 2u * K + group * K + other];
      let value = select(getNaN(), exposure / groupTotal, groupTotal > 0.0);
      indices[indicesOffset + base + ${layout.interaction}u + group * K + other] = value;
      if (other == group) {
        indices[indicesOffset + base + ${layout.isolation}u + group] = value;
      }
    }
  }`
      })
    );
    return nodes;
  }
}
