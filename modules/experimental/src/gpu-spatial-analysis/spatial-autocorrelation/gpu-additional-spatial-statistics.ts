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
import {type GPUSpatialWeights, validateGPUSpatialWeights} from '../spatial-weights/index';
import {
  getSpatialAutocorrelationInputNodes,
  SPATIAL_AUTOCORRELATION_FLOAT_WGSL,
  validateSpatialAutocorrelationInputs
} from './spatial-autocorrelation-kernels';

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
  weights: GPUSpatialWeights;
  values: GraphDataView<'float32'>;
  parameters: GraphDataView<'float32'>;
  mask?: GraphDataView<'uint32'>;
  /** One local Geary value per weights row. */
  result: GPUInferenceResultPort;
  globalStatistics?: GraphDataView<'float32'>;
};

/** Local Geary plugs into the shared conditional-permutation contributor. */
export const GPU_LOCAL_GEARY_PERMUTATION_ADAPTER: GPUPermutationStatisticAdapter = {
  scope: 'local',
  statistic: 'localGeary',
  permutedVariable: 'values',
  alternative: 'two-sided'
};

/**
 * Local Geary `c_i = sum_j w_ij (z_i - z_j)^2`, where `z` is standardized by the population
 * variance of all included rows. Islands return zero and excluded rows return NaN. CSR slot order
 * fixes the summation order.
 */
export class GPULocalGeary implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPULocalGearyProps;

  constructor(props: GPULocalGearyProps) {
    this.id = props.id ?? 'local-geary';
    this.props = props;
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
        declarations: `${SPATIAL_AUTOCORRELATION_FLOAT_WGSL}\nconst ROWS: u32 = ${rows}u;\nconst MOMENTS: u32 = ${rows}u;`,
        body: `let focus = statistics[statisticsOffset + index];
  if (!isFiniteFloat(focus)) {
    result[resultOffset + index] = getQuietNaN(index);
    return;
  }
  let variance = statistics[statisticsOffset + MOMENTS + 2u];
  var sum = 0.0;
  if (variance > 0.0 && isFiniteFloat(variance)) {
    let begin = offsets[offsetsOffset + index];
    let end = offsets[offsetsOffset + index + 1u];
    for (var slot = begin; slot < end; slot++) {
      let neighbor = neighbors[neighborsOffset + slot];
      if (neighbor < ROWS && neighbor != index) {
        let other = statistics[statisticsOffset + neighbor];
        if (isFiniteFloat(other)) {
          let difference = focus - other;
          sum += weights[weightsOffset + slot] * difference * difference / variance;
        }
      }
    }
  }
  result[resultOffset + index] = sum;`
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
    invocationCount: 1,
    bindings,
    declarations: `${SPATIAL_AUTOCORRELATION_FLOAT_WGSL}\nconst ROWS: u32 = ${props.values.length}u;`,
    body: `var weightTotal = 0.0;
  var sumX = 0.0;
  var sumY = 0.0;
  for (var row = 0u; row < ROWS; row++) {
    if (${props.mask ? 'mask[maskOffset + row] == 0u || ' : ''}!isFiniteFloat(values[valuesOffset + row])) { continue; }
    for (var slot = offsets[offsetsOffset + row]; slot < offsets[offsetsOffset + row + 1u]; slot++) {
      let neighbor = neighbors[neighborsOffset + slot];
      if (neighbor >= ROWS || ${props.mask ? 'mask[maskOffset + neighbor] == 0u || ' : ''}!isFiniteFloat(secondValues[secondValuesOffset + neighbor])) { continue; }
      let weight = weights[weightsOffset + slot];
      weightTotal += weight;
      sumX += weight * values[valuesOffset + row];
      sumY += weight * secondValues[secondValuesOffset + neighbor];
    }
  }
  let meanX = sumX / weightTotal;
  let meanY = sumY / weightTotal;
  var numerator = 0.0;
  var squareX = 0.0;
  var squareY = 0.0;
  for (var row = 0u; row < ROWS; row++) {
    if (${props.mask ? 'mask[maskOffset + row] == 0u || ' : ''}!isFiniteFloat(values[valuesOffset + row])) { continue; }
    for (var slot = offsets[offsetsOffset + row]; slot < offsets[offsetsOffset + row + 1u]; slot++) {
      let neighbor = neighbors[neighborsOffset + slot];
      if (neighbor >= ROWS || ${props.mask ? 'mask[maskOffset + neighbor] == 0u || ' : ''}!isFiniteFloat(secondValues[secondValuesOffset + neighbor])) { continue; }
      let weight = weights[weightsOffset + slot];
      let x = values[valuesOffset + row];
      let y = secondValues[secondValuesOffset + neighbor];
      numerator += weight * ${term};
      squareX += weight * (x - meanX) * (x - meanX);
      squareY += weight * (y - meanY) * (y - meanY);
    }
  }
  var answer = getQuietNaN(0u);
  ${
    operation === 'pearson'
      ? 'if (weightTotal > 0.0 && squareX > 0.0 && squareY > 0.0) { answer = clamp(numerator / sqrt(squareX * squareY), -1.0, 1.0); }'
      : 'if (weightTotal > 0.0) { answer = numerator / weightTotal; }'
  }
  result[resultOffset] = answer;`
  });
}
