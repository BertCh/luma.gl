// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {createWGSLKernelNode} from '../../utils/wgsl-kernel-nodes';
import {createLoadBalancedSegmentSumNodes} from '../../utils/load-balanced-for-each';
import {
  validateCapacity,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validatePackedView
} from '../../utils/gpu-contributor-utils';
import {
  createFloat32ValuesNodes,
  type GPUFloat32OrUint32Values,
  validateFloat32OrUint32Values
} from './float32-values';
import {type GPUSpatialWeights, validateGPUSpatialWeights} from './spatial-weights';

const OPERATION = 'GPUSpatialLag';

/**
 * Properties for {@link GPUSpatialLag}.
 *
 * Compile-time: `normalize`, row count, view lengths and which optional views exist. Per-frame: the
 * contents of `values`, `mask` and `weights`.
 */
export type GPUSpatialLagProps = {
  /** Prefix for generated node IDs. Defaults to `'spatial-lag'`. */
  id?: string;
  /**
   * Per-source values `y`, row-major with `columnCount` columns, `float32` or `uint32` (counts,
   * exact below 2^24). Its row count (`values.length / columnCount`) is the neighbor-ID space: the
   * weights row count for square weights, or the target rows of cross weights (query rows to
   * target neighbors, for example the area-share weights of `GPUArealInterpolation`, whose rows
   * are targets and neighbors are sources). Neighbors at or above that row count are skipped.
   * `mask` requires square weights.
   */
  values: GPUFloat32OrUint32Values;
  /**
   * Number of value columns lagged at once. Defaults to 1. Column `c` of row `i` is
   * `output[i * columnCount + c]`; `normalize` divides each row by the same weight sum.
   */
  columnCount?: number;
  /** Weights `W` whose neighbor IDs index `values`. */
  weights: GPUSpatialWeights;
  /** Optional selection (`rows` entries): zero rows output 0 and zero neighbors are skipped. */
  mask?: GraphDataView<'uint32'>;
  /**
   * When true, divides by the sum of the weights of the included neighbors, which gives the lag
   * under row-standardized weights without writing a transformed copy. Rows whose included
   * weight sum is zero output 0. Defaults to false.
   */
  normalize?: boolean;
  /** Caller-owned lag output with `rows * columnCount` entries. */
  output: GraphDataView<'float32'>;
};

/**
 * Spatial lag `lag_i = sum_j w_ij y_j` (PySAL `lag_spatial`) for a {@link GPUSpatialWeights}.
 *
 * Neighbors `j` with a zero mask, or outside the value rows, are skipped. With `normalize` the
 * result is `sum_j w_ij y_j / sum_j w_ij` over the included neighbors. Rows are summed with the
 * shared load-balanced CSR loop, so hub rows (one row with 1e5 neighbors) spread over many
 * workgroups. The association depends only on the offsets, so the result is bitwise reproducible
 * and matches a sequential CPU sum within f32 rounding. This is a single-purpose row loop; for an arbitrary CSR matrix
 * times a vector, `GPUProgramSpMV` in `@luma.gl/gpgpu` is the general (semantic) form.
 */
export class GPUSpatialLag implements GPUCommandNodeProducer {
  /** Prefix for every node ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUSpatialLagProps;

  constructor(props: GPUSpatialLagProps) {
    const id = props.id ?? 'spatial-lag';
    this.id = id;
    this.props = props;
    const rows = validateGPUSpatialWeights(id, props.weights);
    const valueCount = validateFloat32OrUint32Values(props.values, `${id} values`);
    validatePackedView(props.output, ['float32'], `${id} output`);
    const columnCount = props.columnCount ?? 1;
    validateCapacity(id, 'columnCount', columnCount);
    if (valueCount % columnCount !== 0 || valueCount === 0) {
      throw new Error(`${id} values length must be a multiple of columnCount`);
    }
    if (props.output.length !== rows * columnCount) {
      throw new Error(`${id} output length must equal rows * columnCount`);
    }
    if (props.mask) {
      if (valueCount / columnCount !== rows) {
        throw new Error(`${id} mask requires square weights`);
      }
      validatePackedView(props.mask, ['uint32'], `${id} mask`);
      if (props.mask.length !== rows) {
        throw new Error(`${id} mask length must equal the weights row count`);
      }
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      [props.output],
      [
        props.values,
        props.mask,
        props.weights.offsets,
        props.weights.neighbors,
        props.weights.weights
      ]
    );
  }

  /** Number of value rows, the neighbor-ID space of `weights`. */
  getValueRowCount(): number {
    return this.props.values.length / (this.props.columnCount ?? 1);
  }

  /** Returns the lag nodes: value conversion, load-balanced sum and fixup, optional normalize. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {weights, mask, output} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      props.values,
      mask,
      output,
      weights.offsets,
      weights.neighbors,
      weights.weights
    ]);
    const rows = weights.offsets.length - 1;
    const valueRowCount = this.getValueRowCount();
    const columnCount = props.columnCount ?? 1;
    const values = createFloat32ValuesNodes(graph, `${id}-values`, props.values);
    const rowIncluded = mask ? 'mask[maskOffset + row] != 0u' : 'true';
    const neighborIncluded = mask ? 'mask[maskOffset + neighbor] != 0u' : 'true';
    // Load-balanced CSR sum (algorithms roadmap decision 3): a hub row spreads over many
    // workgroups instead of one thread. One lane per value column.
    // Normalize with one weight sum per row; interleaved (lag, weight) scratch would double
    // the caller's binding size and exceed the device limit for otherwise valid outputs.
    const accumulatorType = 'f32';
    const nodes: GPUCommandNode<Parameters>[] = [
      ...values.nodes,
      ...createLoadBalancedSegmentSumNodes<Parameters>(graph, {
        id: `${id}-lag`,
        operation: OPERATION,
        variant: props.normalize ? 'normalized' : 'sum',
        offsets: weights.offsets,
        itemCapacity: weights.neighbors.length,
        laneCount: columnCount,
        accumulatorType,
        output,
        bindings: [
          {name: 'neighbors', view: weights.neighbors, type: 'u32', access: 'read'},
          {name: 'weights', view: weights.weights, type: 'f32', access: 'read'},
          {name: 'values', view: values.view, type: 'f32', access: 'read'},
          ...(mask
            ? [{name: 'mask', view: mask, type: 'u32' as const, access: 'read' as const}]
            : [])
        ],
        declarations: `const ROWS: u32 = ${valueRowCount}u;
const COLUMNS: u32 = ${columnCount}u;
fn getLoadBalancedContribution(row: u32, item: u32, lane: u32) -> ${accumulatorType} {
  let neighbor = neighbors[neighborsOffset + item];
  if (!(${rowIncluded}) || neighbor >= ROWS || !(${neighborIncluded})) {
    return ${accumulatorType}();
  }
  let weight = weights[weightsOffset + item];
  let lagTerm = weight * values[valuesOffset + neighbor * COLUMNS + lane];
  return lagTerm;
}`
      })
    ];
    if (props.normalize) {
      const weightSums = createTransientView(graph, `${id}-weight-sums`, 'float32', rows);
      nodes.push(
        ...createLoadBalancedSegmentSumNodes<Parameters>(graph, {
          id: `${id}-weights`,
          operation: OPERATION,
          offsets: weights.offsets,
          itemCapacity: weights.neighbors.length,
          accumulatorType: 'f32',
          output: weightSums,
          bindings: [
            {name: 'neighbors', view: weights.neighbors, type: 'u32', access: 'read'},
            {name: 'weights', view: weights.weights, type: 'f32', access: 'read'},
            ...(mask
              ? [{name: 'mask', view: mask, type: 'u32' as const, access: 'read' as const}]
              : [])
          ],
          declarations: `fn getLoadBalancedContribution(row: u32, item: u32, lane: u32) -> f32 {
  let neighbor = neighbors[neighborsOffset + item];
  if (!(${rowIncluded}) || neighbor >= ${valueRowCount}u || !(${neighborIncluded})) {
    return 0.0;
  }
  return weights[weightsOffset + item];
}`
        }),
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-normalize`,
          operation: OPERATION,
          variant: 'normalize',
          bindings: [
            {name: 'weightSums', view: weightSums, type: 'f32', access: 'read'},
            {name: 'output', view: output, type: 'f32', access: 'read_write'}
          ],
          invocationCount: rows * columnCount,
          body: `let lag = output[outputOffset + index];
  let weightSum = weightSums[weightSumsOffset + index / ${columnCount}u];
  output[outputOffset + index] = select(0.0, lag / weightSum, weightSum > 0.0);`
        })
      );
    }
    return nodes;
  }
}
