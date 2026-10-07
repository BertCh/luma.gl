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
import {createWGSLKernelNode, getWGSLFloatLiteral} from '../../utils/wgsl-kernel-nodes';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../utils/gpu-contributor-utils';
import {
  type GPUSpatialWeights,
  validateGPUSpatialWeights
} from '../spatial-weights/spatial-weights';
import {getGPUSegregationLayout} from './segregation-layout';

const OPERATION = 'GPUSegregation';
/** Lanes of the reduction workgroups (one workgroup per block of units, or per output column). */
const LANES = 256;
/** Columns a block-reduction round reduces together, so each barrier round serves several terms. */
const COLUMN_BATCH = 8;

/** Largest supported number of groups. */
export const GPU_SEGREGATION_MAXIMUM_GROUPS = 16;

/** Optional per-unit outputs of {@link GPUSegregation}; each holds one block per scale. */
export type GPUSegregationLocalOutputs = {
  /**
   * `scales * units * K` local composition `pi_im` (share of group `m` in the environment of unit
   * `i`), index `(scale * units + i) * K + m`. Environment shares sum to 1 for a populated unit.
   */
  environment?: GraphDataView<'float32'>;
  /** `scales * units` local entropy `E_i = sum_m pi_im ln(1 / pi_im)` of the environment, in nats. */
  entropy?: GraphDataView<'float32'>;
  /**
   * `scales * units * K` unit contributions to `D_g`: `t_i |pi_ig - P_g| / (2 T P_g (1 - P_g))`.
   * The sum over units is the dissimilarity `D_g` of the same scale.
   */
  dissimilarity?: GraphDataView<'float32'>;
  /**
   * `scales * units` unit contributions to H: `t_i (E - E_i) / (E T)`. The sum over units is the
   * entropy index H of the same scale.
   */
  theil?: GraphDataView<'float32'>;
};

/**
 * Properties for {@link GPUSegregation}.
 *
 * Compile-time: `unitCount`, `groupCount`, the scales (their number and weights structure),
 * `selfWeight`, `atkinsonB` and which optional outputs exist. Per-frame: the contents of
 * `groupCounts` and of every weights view.
 */
export type GPUSegregationProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'segregation'`. */
  id?: string;
  /** Number of units (rows), for example census tracts or grid cells. */
  unitCount: number;
  /** Number of population groups `K`, in `[2, GPU_SEGREGATION_MAXIMUM_GROUPS]`. */
  groupCount: number;
  /**
   * Population per unit and group, `unitCount * K` float32 values, row-major
   * (`groupCounts[i * K + m]`). Negative and non-finite entries count as 0.
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
  atkinsonB?: number;
  /**
   * Caller-owned `scales.length * layout.stride` global indices; scale `s` starts at
   * `s * stride`. See `getGPUSegregationLayout`.
   */
  indices: GraphDataView<'float32'>;
  /** Optional caller-owned per-unit outputs. */
  local?: GPUSegregationLocalOutputs;
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
 * Indices that are undefined (empty total, a group with zero or all of the population, `E = 0`,
 * `I = 0`) are written as 0.
 *
 * Local indices (optional `local` outputs) are the per-unit terms of these sums, so they add up to
 * the global index of the same scale: environment composition, local entropy, and the unit
 * contributions to `D_g` and to H. Multiscale: one global result per scale, from the aspatial
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
 * a float64 reference for moderate populations.
 */
export class GPUSegregation implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUSegregationProps;
  /** Number of scales. */
  readonly scaleCount: number;

  constructor(props: GPUSegregationProps) {
    const id = props.id ?? 'segregation';
    this.id = id;
    this.props = props;
    const {unitCount, groupCount} = props;
    if (!Number.isSafeInteger(unitCount) || unitCount < 1) {
      throw new Error(`${id} unitCount must be a positive integer`);
    }
    if (
      !Number.isSafeInteger(groupCount) ||
      groupCount < 2 ||
      groupCount > GPU_SEGREGATION_MAXIMUM_GROUPS
    ) {
      throw new Error(
        `${id} groupCount must be an integer in [2, ${GPU_SEGREGATION_MAXIMUM_GROUPS}]`
      );
    }
    const scales = props.scales ?? [null];
    if (scales.length < 1) {
      throw new Error(`${id} scales must not be empty`);
    }
    this.scaleCount = scales.length;
    validatePackedView(props.groupCounts, ['float32'], `${id} groupCounts`);
    if (props.groupCounts.length !== unitCount * groupCount) {
      throw new Error(`${id} groupCounts must contain unitCount * groupCount rows`);
    }
    scales.forEach((weights, scaleIndex) => {
      if (
        weights &&
        validateGPUSpatialWeights(id, weights, `scales[${scaleIndex}]`) !== unitCount
      ) {
        throw new Error(`${id} scales[${scaleIndex}] must have one row per unit`);
      }
    });
    const b = props.atkinsonB ?? 0.5;
    if (!(b > 0 && b < 1)) {
      throw new Error(`${id} atkinsonB must be in (0, 1)`);
    }
    const selfWeight = props.selfWeight ?? 1;
    if (!Number.isFinite(selfWeight) || selfWeight < 0) {
      throw new Error(`${id} selfWeight must be finite and non-negative`);
    }
    const layout = getGPUSegregationLayout(groupCount);
    const expectedLengths: [string, GraphDataView<'float32'> | undefined, number][] = [
      ['indices', props.indices, this.scaleCount * layout.stride],
      ['local.environment', props.local?.environment, this.scaleCount * unitCount * groupCount],
      ['local.entropy', props.local?.entropy, this.scaleCount * unitCount],
      ['local.dissimilarity', props.local?.dissimilarity, this.scaleCount * unitCount * groupCount],
      ['local.theil', props.local?.theil, this.scaleCount * unitCount]
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
    const {local} = this.props;
    return [
      this.props.indices,
      local?.environment,
      local?.entropy,
      local?.dissimilarity,
      local?.theil
    ];
  }

  /** Returns the totals, environment, term, reduction and finalize nodes in dependency order. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    validateGraphViewsBelongToGraph(id, graph, [...this.getInputViews(), ...this.getOutputViews()]);
    const {unitCount, groupCount: K, groupCounts, indices, local} = props;
    const scales = props.scales ?? [null];
    const selfWeight = props.selfWeight ?? 1;
    const b = props.atkinsonB ?? 0.5;
    const layout = getGPUSegregationLayout(K);
    const termCount = 2 + 2 * K + K * K;
    const nodes: GPUCommandNode<Parameters>[] = [];

    const declarations = `const UNITS: u32 = ${unitCount}u;
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
      collected?: {view: GraphDataView<'float32'>; base: number}
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
                }
              ]
            : [])
        ],
        invocationCount: columns * LANES,
        guardIndex: false,
        declarations: `const LANES: u32 = ${LANES}u;
const GROUPS: u32 = ${unitGroups}u;
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
    ${collected ? `collected[collectedOffset + ${collected.base}u + column] = scratch[0];` : ''}
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
    const addTotals = (label: string, counts: GraphDataView<'float32'>, scaleIndex?: number) => {
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
          scaleIndex === undefined ? undefined : {view: allTotals, base: scaleIndex * (K + 1)}
        )
      );
      return totals;
    };
    const collectTotals = (scaleIndex: number, totals: GraphDataView<'float32'>) =>
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-collect-totals-${scaleIndex}`,
        operation: OPERATION,
        variant: 'collect-totals',
        bindings: [
          {name: 'totals', view: totals, type: 'f32', access: 'read'},
          {name: 'allTotals', view: allTotals, type: 'f32', access: 'read_write'}
        ],
        invocationCount: K + 1,
        body: `allTotals[allTotalsOffset + ${scaleIndex * (K + 1)}u + index] = totals[totalsOffset + index];`
      });
    const sharedTotals = smoothedPopulation ? undefined : addTotals('totals', groupCounts);

    // The composition share and environment helpers shared by the term and local kernels.
    const environmentDeclarations = `${declarations}
fn getShare(unit: u32, group: u32) -> f32 {
  let total = environmentTotal[environmentTotalOffset + unit];
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
    scales.forEach((weights, scaleIndex) => {
      const scaleId = `${id}-scale-${scaleIndex}`;
      const environment = createTransientView(
        graph,
        `${scaleId}-environment`,
        'float32',
        unitCount * K
      );
      const environmentTotal = createTransientView(
        graph,
        `${scaleId}-environment-total`,
        'float32',
        unitCount
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
            {name: 'environmentTotal', view: environmentTotal, type: 'f32', access: 'read_write'}
          ],
          invocationCount: unitCount,
          declarations,
          body: weights
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
  environmentTotal[environmentTotalOffset + index] = total;`
            : `for (var group = 0u; group < K; group++) {
    environment[environmentOffset + index * K + group] = countAt(index, group);
  }
  environmentTotal[environmentTotalOffset + index] = unitTotal(index);`
        })
      );
      const scaleTotals =
        sharedTotals ?? addTotals(`scale-${scaleIndex}-totals`, counts, scaleIndex);
      nodes.push(
        ...(sharedTotals ? [collectTotals(scaleIndex, scaleTotals)] : []),
        createWGSLKernelNode<Parameters>(graph, {
          id: `${scaleId}-terms`,
          operation: OPERATION,
          variant: 'terms',
          bindings: [
            {name: 'groupCounts', view: counts, type: 'f32', access: 'read'},
            {name: 'environment', view: environment, type: 'f32', access: 'read'},
            {name: 'environmentTotal', view: environmentTotal, type: 'f32', access: 'read'},
            {name: 'totals', view: scaleTotals, type: 'f32', access: 'read'},
            {name: 'partials', view: termPartials, type: 'f32', access: 'read_write'}
          ],
          invocationCount: unitGroups * LANES,
          guardIndex: false,
          declarations: reductionDeclarations(termCount, environmentDeclarations),
          body: `let lane = localInvocationIndex;
  let group = (index - lane) / LANES;
  if (index < UNITS) {
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
          base: scaleIndex * termCount
        })
      );
      if (local && (local.environment || local.entropy || local.dissimilarity || local.theil)) {
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
              {name: 'environmentTotal', view: environmentTotal, type: 'f32', access: 'read'},
              {name: 'totals', view: scaleTotals, type: 'f32', access: 'read'},
              ...outputs.flatMap(([name, view]) =>
                view ? [{name, view, type: 'f32' as const, access: 'read_write' as const}] : []
              )
            ],
            invocationCount: unitCount,
            declarations: environmentDeclarations,
            body: `let unitPopulation = unitTotal(index);
  let populationTotal = totals[totalsOffset + K];
  let diversity = getDiversity();
  var entropy = 0.0;
  for (var group = 0u; group < K; group++) {
    let share = getShare(index, group);
    entropy += entropyTerm(share);
    ${local.environment ? `localEnvironment[localEnvironmentOffset + (${scaleIndex}u * UNITS + index) * K + group] = share;` : ''}
    ${
      local.dissimilarity
        ? `let populationShare = getPopulationShare(group);
    let scale = 2.0 * populationTotal * populationShare * (1.0 - populationShare);
    localDissimilarity[localDissimilarityOffset + (${scaleIndex}u * UNITS + index) * K + group] =
      select(0.0, unitPopulation * abs(share - populationShare) / scale, scale > 0.0);`
        : ''
    }
  }
  ${local.entropy ? `localEntropy[localEntropyOffset + ${scaleIndex}u * UNITS + index] = entropy;` : ''}
  ${
    local.theil
      ? `localTheil[localTheilOffset + ${scaleIndex}u * UNITS + index] =
    select(0.0, unitPopulation * (diversity - entropy) / (diversity * populationTotal), diversity > 0.0 && populationTotal > 0.0);`
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
        declarations: `const K: u32 = ${K}u;
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
    0.0, 1.0 - allSums[allSumsOffset + sums] / (diversity * populationTotal), hasPopulation && diversity > 0.0);
  indices[indicesOffset + base + ${layout.multiGroupDissimilarity}u] = select(
    0.0, allSums[allSumsOffset + sums + 1u] / (2.0 * populationTotal * variance), hasPopulation && variance > 0.0);
  indices[indicesOffset + base + ${layout.diversity}u] = diversity;
  for (var group = 0u; group < K; group++) {
    let groupTotal = allTotals[allTotalsOffset + index * (K + 1u) + group];
    let share = select(0.0, groupTotal / populationTotal, hasPopulation);
    let isMixed = hasPopulation && share > 0.0 && share < 1.0;
    let spread = 2.0 * populationTotal * share * (1.0 - share);
    indices[indicesOffset + base + ${layout.dissimilarity}u + group] = select(
      0.0, allSums[allSumsOffset + sums + 2u + group] / spread, isMixed);
    let ratio = abs(allSums[allSumsOffset + sums + 2u + K + group] / max(share * populationTotal, 1e-30));
    let atkinson = 1.0 - share / (1.0 - share) * pow(ratio, 1.0 / (1.0 - ATKINSON_B));
    indices[indicesOffset + base + ${layout.atkinson}u + group] = select(0.0, atkinson, isMixed);
    for (var other = 0u; other < K; other++) {
      let exposure = allSums[allSumsOffset + sums + 2u + 2u * K + group * K + other];
      let value = select(0.0, exposure / groupTotal, groupTotal > 0.0);
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
