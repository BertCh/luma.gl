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
import {createWGSLKernelNode} from '../../utils/wgsl-kernel-nodes';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../utils/gpu-contributor-utils';
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
  /** Per-row values `y`. Length must equal the weights row count (square weights). */
  values: GraphDataView<'float32'>;
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
  /** Caller-owned lag output with `rows` entries. */
  output: GraphDataView<'float32'>;
};

/**
 * Spatial lag `lag_i = sum_j w_ij y_j` (PySAL `lag_spatial`) for a {@link GPUSpatialWeights}.
 *
 * Neighbors `j` with a zero mask, or outside `[0, rows)`, are skipped. With `normalize` the result is
 * `sum_j w_ij y_j / sum_j w_ij` over the included neighbors. Each row accumulates in slot order
 * in f32, so the result is deterministic and matches a sequential CPU sum within f32 fused
 * multiply-add differences. This is a single-purpose row loop; for an arbitrary CSR matrix times a
 * vector, `GPUProgramSpMV` in `@luma.gl/gpgpu` is the general (semantic) form.
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
    validatePackedView(props.values, ['float32'], `${id} values`);
    validatePackedView(props.output, ['float32'], `${id} output`);
    if (props.values.length !== rows) {
      throw new Error(`${id} values length must equal the weights row count`);
    }
    if (props.output.length !== rows) {
      throw new Error(`${id} output length must equal the weights row count`);
    }
    if (props.mask) {
      validatePackedUint32View(props.mask, `${id} mask`);
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

  /** Returns the lag node. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {weights, mask, values, output} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      values,
      mask,
      output,
      weights.offsets,
      weights.neighbors,
      weights.weights
    ]);
    const rows = weights.offsets.length - 1;
    return [
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-lag`,
        operation: OPERATION,
        variant: props.normalize ? 'normalized' : 'sum',
        bindings: [
          {name: 'offsets', view: weights.offsets, type: 'u32', access: 'read'},
          {name: 'neighbors', view: weights.neighbors, type: 'u32', access: 'read'},
          {name: 'weights', view: weights.weights, type: 'f32', access: 'read'},
          {name: 'values', view: values, type: 'f32', access: 'read'},
          ...(mask
            ? [{name: 'mask', view: mask, type: 'u32' as const, access: 'read' as const}]
            : []),
          {name: 'output', view: output, type: 'f32', access: 'read_write'}
        ],
        invocationCount: rows,
        declarations: `const ROWS: u32 = ${rows}u;`,
        body: `var lag = 0.0;
  var weightSum = 0.0;
  if (${mask ? 'mask[maskOffset + index] != 0u' : 'true'}) {
    for (var slot = offsets[offsetsOffset + index]; slot < offsets[offsetsOffset + index + 1u]; slot++) {
      let neighbor = neighbors[neighborsOffset + slot];
      if (neighbor < ROWS && ${mask ? 'mask[maskOffset + neighbor] != 0u' : 'true'}) {
        let weight = weights[weightsOffset + slot];
        lag += weight * values[valuesOffset + neighbor];
        weightSum += weight;
      }
    }
  }
  ${props.normalize ? 'lag = select(0.0, lag / weightSum, weightSum > 0.0);' : ''}
  output[outputOffset + index] = lag;`
      })
    ];
  }
}
