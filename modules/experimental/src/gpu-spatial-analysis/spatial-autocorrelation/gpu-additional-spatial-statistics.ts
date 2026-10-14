// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
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
import type {GPUInferenceResultPort, GPUPermutationStatisticAdapter} from '../contracts/index';
import {
  isGPUSpatialWeightsRowStandardized,
  resolveGPUIslandPolicy,
  type GPUIslandPolicy,
  type GPUSpatialWeights,
  validateGPUSpatialWeights
} from '../spatial-weights/index';
import {
  getSpatialAutocorrelationConstantsWGSL,
  getSpatialAutocorrelationInputNodes,
  getSpatialWeightsNeighborLoopWGSL,
  SPATIAL_AUTOCORRELATION_FLOAT_WGSL,
  validateSpatialAutocorrelationInputs
} from './spatial-autocorrelation-kernels';

/** Lanes of the single workgroup that reduces a global edge-association statistic. */
const EDGE_ASSOCIATION_WORKGROUP_SIZE = 256;

/** Inputs shared by global edge-association statistics. */
type GPUEdgeAssociationProps = {
  id?: string;
  weights: GPUSpatialWeights;
  values: GraphDataView<'float32'>;
  secondValues?: GraphDataView<'float32'>;
  mask?: GraphDataView<'uint32'>;
  /** One caller-owned float32 statistic. */
  result: GPUInferenceResultPort;
};

/** Properties for {@link GPULocalGeary}. */
export type GPULocalGearyProps = {
  id?: string;
  /**
   * Square self-join weights. Used as given, except that when `weights.info.transform` is `'row'`
   * each row is re-standardized over the neighbors that remain after masking. A neighbor is a slot
   * with a positive weight that points at another included row.
   */
  weights: GPUSpatialWeights;
  values: GraphDataView<'float32'>;
  parameters: GraphDataView<'float32'>;
  mask?: GraphDataView<'uint32'>;
  /**
   * Compile-time. `'exclude'` (default) removes islands from the moments and writes NaN for them;
   * `'keep'` keeps them in the moments (esda) with a local Geary value of 0. The count is
   * reported in `globalStatistics[4]`.
   */
  islands?: GPUIslandPolicy;
  /** One local Geary value per weights row. */
  result: GPUInferenceResultPort;
  /** Optional `[n, mean, variance, standardDeviation, islandCount]` used this frame. */
  globalStatistics?: GraphDataView<'float32'>;
};

/**
 * Local Geary plugs into the shared conditional-permutation contributor. Small `c_i` means
 * similar neighbors (a cluster), so the tail that signals clustering is the lower one:
 * `'lesser'`.
 */
export const GPU_LOCAL_GEARY_PERMUTATION_ADAPTER: GPUPermutationStatisticAdapter = {
  scope: 'local',
  statistic: 'localGeary',
  permutedVariable: 'values',
  alternative: 'lesser'
};

/**
 * Local bivariate Moran (esda `Moran_Local_BV`) runs in `GPULocalPermutationTest` with
 * `statistic: 'localBivariateMoran'`: the focus keeps `x`, the neighbors' `y` are drawn.
 */
export const GPU_LOCAL_BIVARIATE_MORAN_PERMUTATION_ADAPTER: GPUPermutationStatisticAdapter = {
  scope: 'local',
  statistic: 'localBivariateMoran',
  permutedVariable: 'secondValues',
  alternative: 'two-sided'
};

/**
 * Local join counts (Anselin and Li 2019; esda `Join_Counts_Local`, `Join_Counts_Local_BV`) run in
 * `GPULocalPermutationTest` with `statistic: 'localJoinCount'` or `'localBivariateJoinCount'`.
 * Clustering is more joins than expected, so the tail is `'greater'`.
 */
export const GPU_LOCAL_JOIN_COUNT_PERMUTATION_ADAPTER: GPUPermutationStatisticAdapter = {
  scope: 'local',
  statistic: 'localJoinCount',
  permutedVariable: 'values',
  alternative: 'greater'
};

/** Bivariate local join counts permute `secondValues` (and the paired `values` of the drawn rows). */
export const GPU_LOCAL_BIVARIATE_JOIN_COUNT_PERMUTATION_ADAPTER: GPUPermutationStatisticAdapter = {
  scope: 'local',
  statistic: 'localBivariateJoinCount',
  permutedVariable: 'secondValues',
  alternative: 'greater'
};

/**
 * Lee's spatial Pearson (esda `Spatial_Pearson`, `Spatial_Pearson_Local`) runs in
 * `GPUGlobalPermutationTest` (`statistic: 'leesL'`) and `GPULocalPermutationTest`
 * (`statistic: 'localLeeL'`). Both permute the rows of `(x, y)` together, so the tests ask whether
 * the spatial pattern of the pair is unusual given their pairing; esda's own p-value is the
 * 'directed' one, which is anti-conservative. The default `'two-sided'` is kept.
 */
export const GPU_LEES_L_PERMUTATION_ADAPTERS: {
  global: GPUPermutationStatisticAdapter;
  local: GPUPermutationStatisticAdapter;
} = {
  global: {
    scope: 'global',
    statistic: 'leesL',
    permutedVariable: 'values',
    alternative: 'two-sided'
  },
  local: {
    scope: 'local',
    statistic: 'localLeeL',
    permutedVariable: 'values',
    alternative: 'two-sided'
  }
};

/**
 * Local Geary `c_i = sum_j w_ij (z_i - z_j)^2`, where `z` is standardized by the population
 * variance of the analysis rows. The sum runs over active neighbors (positive weight, included
 * row other than `i`); row-standardized weights are re-standardized over the neighbors that remain
 * after masking. Excluded rows return NaN, as do islands under `islands: 'exclude'` (default);
 * under `'keep'` an island returns 0. A constant variable (zero variance) returns NaN. CSR slot
 * order fixes the summation order. There is no analytic classification; use permutation inference
 * with `GPU_LOCAL_GEARY_PERMUTATION_ADAPTER` (`GPULocalPermutationTest` with `statistic:
 * 'localGeary'` and its `classes` output gives the positive / negative association classes).
 */
export class GPULocalGeary implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPULocalGearyProps;

  /** Resolved island policy. */
  readonly islands: GPUIslandPolicy;

  constructor(props: GPULocalGearyProps) {
    this.id = props.id ?? 'local-geary';
    this.props = props;
    this.islands = resolveGPUIslandPolicy(this.id, props.islands);
    const rows = validateSpatialAutocorrelationInputs({...props, id: this.id});
    validatePackedView(props.result.statistic, ['float32'], `${this.id} result.statistic`);
    if (props.result.statistic.length !== rows) {
      throw new Error(`${this.id} result.statistic length must equal the weights row count`);
    }
    if (props.result.pValues || props.result.adjustedPValues || props.result.classifications) {
      throw new Error(`${this.id} analytic inference is not defined; use permutation inference`);
    }
    validateGraphOutputsDisjointFromInputs(
      this.id,
      [props.result.statistic, props.globalStatistics],
      [
        props.weights.offsets,
        props.weights.neighbors,
        props.weights.weights,
        props.weights.status?.overflow,
        props.values,
        props.parameters,
        props.mask
      ]
    );
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    validateGraphViewsBelongToGraph(id, graph, [
      props.weights.offsets,
      props.weights.neighbors,
      props.weights.weights,
      props.weights.status?.overflow,
      props.values,
      props.parameters,
      props.mask,
      props.result.statistic,
      props.globalStatistics
    ]);
    const rows = props.values.length;
    const inputs = getSpatialAutocorrelationInputNodes(graph, {
      ...props,
      id,
      islands: this.islands,
      operation: 'GPULocalGeary'
    });
    inputs.nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-statistic`,
        operation: 'GPULocalGeary',
        variant: 'local-geary',
        invocationCount: rows,
        bindings: [
          {name: 'offsets', view: props.weights.offsets, type: 'u32', access: 'read'},
          {name: 'neighbors', view: props.weights.neighbors, type: 'u32', access: 'read'},
          {name: 'weights', view: props.weights.weights, type: 'f32', access: 'read'},
          {name: 'statistics', view: inputs.statistics, type: 'f32', access: 'read'},
          {name: 'result', view: props.result.statistic, type: 'f32', access: 'read_write'}
        ],
        declarations: `${SPATIAL_AUTOCORRELATION_FLOAT_WGSL}
${getSpatialAutocorrelationConstantsWGSL(rows, props.weights.neighbors.length)}
const STANDARDIZE: bool = ${isGPUSpatialWeightsRowStandardized(props.weights)};`,
        body: `let focus = statistics[statisticsOffset + index];
  if (!isFiniteFloat(focus)) {
    result[resultOffset + index] = getQuietNaN(index);
    return;
  }
  let variance = statistics[statisticsOffset + MOMENTS + 2u];
  var sum = 0.0;
  var weightSum = 0.0;
  ${getSpatialWeightsNeighborLoopWGSL(`let other = statistics[statisticsOffset + neighbor];
    let difference = focus - other;
    sum += weight * difference * difference;
    weightSum += weight;`)}
  if (STANDARDIZE && weightSum > 0.0) {
    // Row-standardized weights are divided by what remains of their row sum.
    sum = sum / weightSum;
  }
  // A constant variable has no standardized values: the statistic is undefined, not zero.
  result[resultOffset + index] = select(getQuietNaN(index), sum / variance, variance > 0.0 && isFiniteFloat(variance));`
      })
    );
    return inputs.nodes;
  }
}

/** Properties for {@link GPUSpatialPearson}. */
export type GPUSpatialPearsonProps = GPUEdgeAssociationProps & {
  /** Second variable is required for the spatial cross-correlation. */
  secondValues: GraphDataView<'float32'>;
};

/** Spatial Pearson shares the global bivariate cross-product permutation engine. */
export const GPU_SPATIAL_PEARSON_PERMUTATION_ADAPTER: GPUPermutationStatisticAdapter = {
  scope: 'global',
  statistic: 'bivariateMoran',
  permutedVariable: 'secondValues',
  alternative: 'two-sided'
};

/**
 * Weighted Pearson correlation across directed spatial edges `(i,j)`: the weighted covariance of
 * `x_i` and `y_j`, divided by their weighted edge standard deviations. This definition is bounded
 * to `[-1, 1]`, is valid for asymmetric weights, and returns NaN for zero edge weight or variance.
 */
export class GPUSpatialPearson implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPUSpatialPearsonProps;

  constructor(props: GPUSpatialPearsonProps) {
    this.id = props.id ?? 'spatial-pearson';
    this.props = props;
    validateEdgeAssociationProps(this.id, props);
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    return [createEdgeAssociationNode(graph, this.id, this.props, 'pearson')];
  }
}

/** Gamma statistic operation. */
export type GPUGammaOperation = 'cross-product' | 'squared-difference' | 'absolute-difference';

/** Properties for {@link GPUGammaStatistic}. */
export type GPUGammaStatisticProps = GPUEdgeAssociationProps & {
  operation?: GPUGammaOperation;
};

/** Returns the existing global permutation statistic with the same order-preserving pair sum. */
export function getGPUGammaPermutationAdapter(
  operation: GPUGammaOperation
): GPUPermutationStatisticAdapter {
  if (operation === 'absolute-difference') {
    throw new Error('absolute-difference Gamma has no shared permutation adapter');
  }
  return {
    scope: 'global',
    statistic: operation === 'squared-difference' ? 'geary' : 'moran',
    permutedVariable: 'values',
    alternative: 'two-sided'
  };
}

/**
 * Hubert's Gamma over directed weighted edges. The operation is either centered cross-product,
 * squared difference, or absolute difference. The result is the weighted mean over valid edges.
 */
export class GPUGammaStatistic implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPUGammaStatisticProps;
  readonly operation: GPUGammaOperation;

  constructor(props: GPUGammaStatisticProps) {
    this.id = props.id ?? 'gamma-statistic';
    this.props = props;
    this.operation = props.operation ?? 'cross-product';
    if (!['cross-product', 'squared-difference', 'absolute-difference'].includes(this.operation)) {
      throw new Error(`${this.id} unknown operation ${this.operation}`);
    }
    validateEdgeAssociationProps(this.id, props);
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    return [createEdgeAssociationNode(graph, this.id, this.props, this.operation)];
  }
}

function validateEdgeAssociationProps(id: string, props: GPUEdgeAssociationProps): void {
  const rows = validateGPUSpatialWeights(id, props.weights);
  validatePackedView(props.values, ['float32'], `${id} values`);
  if (props.values.length !== rows) throw new Error(`${id} values length must equal weights rows`);
  if (props.secondValues) {
    validatePackedView(props.secondValues, ['float32'], `${id} secondValues`);
    if (props.secondValues.length !== rows)
      throw new Error(`${id} secondValues length must equal weights rows`);
  }
  if (props.mask) {
    validatePackedUint32View(props.mask, `${id} mask`);
    if (props.mask.length !== rows) throw new Error(`${id} mask length must equal weights rows`);
  }
  validatePackedView(props.result.statistic, ['float32'], `${id} result.statistic`);
  if (props.result.statistic.length < 1)
    throw new Error(`${id} result.statistic must hold one value`);
  if (props.result.pValues || props.result.adjustedPValues || props.result.classifications) {
    throw new Error(`${id} analytic inference is not defined; use permutation inference`);
  }
  validateGraphOutputsDisjointFromInputs(
    id,
    [props.result.statistic],
    [
      props.weights.offsets,
      props.weights.neighbors,
      props.weights.weights,
      props.values,
      props.secondValues,
      props.mask
    ]
  );
}

function createEdgeAssociationNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  props: GPUEdgeAssociationProps,
  operation: 'pearson' | GPUGammaOperation
): GPUCommandNode<Parameters> {
  validateGraphViewsBelongToGraph(id, graph, [
    props.weights.offsets,
    props.weights.neighbors,
    props.weights.weights,
    props.values,
    props.secondValues,
    props.mask,
    props.result.statistic
  ]);
  const secondValues = props.secondValues ?? props.values;
  const bindings: WGSLKernelBinding[] = [
    {name: 'offsets', view: props.weights.offsets, type: 'u32', access: 'read'},
    {name: 'neighbors', view: props.weights.neighbors, type: 'u32', access: 'read'},
    {name: 'weights', view: props.weights.weights, type: 'f32', access: 'read'},
    {name: 'values', view: props.values, type: 'f32', access: 'read'},
    {name: 'secondValues', view: secondValues, type: 'f32', access: 'read'},
    ...(props.mask
      ? [{name: 'mask', view: props.mask, type: 'u32' as const, access: 'read' as const}]
      : []),
    {name: 'result', view: props.result.statistic, type: 'f32', access: 'read_write'}
  ];
  const term =
    operation === 'cross-product' || operation === 'pearson'
      ? '(x - meanX) * (y - meanY)'
      : operation === 'squared-difference'
        ? '(x - y) * (x - y)'
        : operation === 'absolute-difference'
          ? 'abs(x - y)'
          : '0.0';
  return createWGSLKernelNode(graph, {
    id: `${id}-statistic`,
    operation: operation === 'pearson' ? 'GPUSpatialPearson' : 'GPUGammaStatistic',
    variant: operation,
    // One workgroup reduces all edges: lanes stripe rows and combine in a fixed tree order.
    workgroupSize: EDGE_ASSOCIATION_WORKGROUP_SIZE,
    invocationCount: EDGE_ASSOCIATION_WORKGROUP_SIZE,
    guardIndex: false,
    bindings,
    declarations: `${SPATIAL_AUTOCORRELATION_FLOAT_WGSL}
const ROWS: u32 = ${props.values.length}u;
const LANES: u32 = ${EDGE_ASSOCIATION_WORKGROUP_SIZE}u;
var<workgroup> sharedA: array<f32, ${EDGE_ASSOCIATION_WORKGROUP_SIZE}>;
var<workgroup> sharedB: array<f32, ${EDGE_ASSOCIATION_WORKGROUP_SIZE}>;
var<workgroup> sharedC: array<f32, ${EDGE_ASSOCIATION_WORKGROUP_SIZE}>;
`,
    body: `let lane = localInvocationIndex;
  var laneWeight = 0.0;
  var laneX = 0.0;
  var laneY = 0.0;
  for (var row = lane; row < ROWS; row += LANES) {
    if (%ROW_OK%) { continue; }
    for (var slot = offsets[offsetsOffset + row]; slot < offsets[offsetsOffset + row + 1u]; slot++) {
      let neighbor = neighbors[neighborsOffset + slot];
      if (%NEIGHBOR_OK%) { continue; }
      let weight = weights[weightsOffset + slot];
      laneWeight += weight;
      laneX += weight * values[valuesOffset + row];
      laneY += weight * secondValues[secondValuesOffset + neighbor];
    }
  }
  sharedA[lane] = laneWeight;
  sharedB[lane] = laneX;
  sharedC[lane] = laneY;
  workgroupBarrier();
  for (var stride = LANES / 2u; stride > 0u; stride /= 2u) {
    if (lane < stride) {
      sharedA[lane] += sharedA[lane + stride];
      sharedB[lane] += sharedB[lane + stride];
      sharedC[lane] += sharedC[lane + stride];
    }
    workgroupBarrier();
  }
  let weightTotal = sharedA[0];
  let meanX = sharedB[0] / weightTotal;
  let meanY = sharedC[0] / weightTotal;
  workgroupBarrier();
  var laneNumerator = 0.0;
  var laneSquareX = 0.0;
  var laneSquareY = 0.0;
  for (var row = lane; row < ROWS; row += LANES) {
    if (%ROW_OK%) { continue; }
    for (var slot = offsets[offsetsOffset + row]; slot < offsets[offsetsOffset + row + 1u]; slot++) {
      let neighbor = neighbors[neighborsOffset + slot];
      if (%NEIGHBOR_OK%) { continue; }
      let weight = weights[weightsOffset + slot];
      let x = values[valuesOffset + row];
      let y = secondValues[secondValuesOffset + neighbor];
      laneNumerator += weight * ${term};
      laneSquareX += weight * (x - meanX) * (x - meanX);
      laneSquareY += weight * (y - meanY) * (y - meanY);
    }
  }
  sharedA[lane] = laneNumerator;
  sharedB[lane] = laneSquareX;
  sharedC[lane] = laneSquareY;
  workgroupBarrier();
  for (var stride = LANES / 2u; stride > 0u; stride /= 2u) {
    if (lane < stride) {
      sharedA[lane] += sharedA[lane + stride];
      sharedB[lane] += sharedB[lane + stride];
      sharedC[lane] += sharedC[lane + stride];
    }
    workgroupBarrier();
  }
  if (lane == 0u) {
    let numerator = sharedA[0];
    let squareX = sharedB[0];
    let squareY = sharedC[0];
    var answer = getQuietNaN(0u);
    ${
      operation === 'pearson'
        ? 'if (weightTotal > 0.0 && squareX > 0.0 && squareY > 0.0) { answer = clamp(numerator / sqrt(squareX * squareY), -1.0, 1.0); }'
        : 'if (weightTotal > 0.0) { answer = numerator / weightTotal; }'
    }
    result[resultOffset] = answer;
  }`
      .replaceAll(
        '%ROW_OK%',
        `${props.mask ? 'mask[maskOffset + row] == 0u || ' : ''}!isFiniteFloat(values[valuesOffset + row])`
      )
      .replaceAll(
        '%NEIGHBOR_OK%',
        `neighbor >= ROWS || neighbor == row || !(weights[weightsOffset + slot] > 0.0) || ${props.mask ? 'mask[maskOffset + neighbor] == 0u || ' : ''}!isFiniteFloat(secondValues[secondValuesOffset + neighbor])`
      )
  });
}
