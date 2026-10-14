// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUReduction,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {createWGSLKernelNode} from '../../utils/wgsl-kernel-nodes';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validatePackedView
} from '../../utils/gpu-contributor-utils';
import {IS_FINITE_FLOAT_WGSL} from '../../utils/wgsl-common';
import {
  describeGPUSpatialWeights,
  type GPUSpatialWeights,
  validateGPUSpatialWeights
} from './spatial-weights';

const OPERATION = 'GPUSpatialWeightsValidation';

/**
 * Bits of the per-row flags written by {@link GPUSpatialWeightsValidation}. A row's flags are the
 * bitwise OR of the violations found in it; 0 means the row satisfies every invariant.
 */
export const GPU_SPATIAL_WEIGHTS_VIOLATION = {
  /**
   * Bad offsets: `offsets[0] != 0` (reported on row 0), an offset that decreases (`offsets[i] >
   * offsets[i + 1]`, reported on row `i`), or `offsets[rows]` above the slot capacity (reported on
   * the last row).
   */
  offsets: 1,
  /** A neighbor ID `>= rows`. */
  neighborRange: 2,
  /** Neighbor IDs of the row are not strictly ascending (a duplicate or a descending pair). */
  order: 4,
  /** The row lists itself as a neighbor. */
  self: 8,
  /** A weight is NaN, infinite or negative. */
  weight: 16
} as const;

/** Properties for {@link GPUSpatialWeightsValidation}. */
export type GPUSpatialWeightsValidationProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'spatial-weights-validation'`. */
  id?: string;
  /** Square weights to check. Read only. */
  weights: GPUSpatialWeights;
  /** Caller-owned outputs. */
  output: GPUSpatialWeightsValidationOutput;
};

/** Caller-owned outputs of {@link GPUSpatialWeightsValidation}. */
export type GPUSpatialWeightsValidationOutput = {
  /**
   * Optional uint32 per row (length equal to the row count) receiving the violation bits of
   * {@link GPU_SPATIAL_WEIGHTS_VIOLATION}.
   */
  flags?: GraphDataView<'uint32'>;
  status: {
    /**
     * One-row uint32 receiving the number of rows with at least one violation. Zero means the
     * weights satisfy every {@link GPUSpatialWeights} invariant.
     */
    invalidCount: GraphDataView<'uint32'>;
  };
};

/**
 * Checks the {@link GPUSpatialWeights} invariants on the GPU, for CSR that was built elsewhere
 * and uploaded (producers in this package already guarantee them).
 *
 * Uploaded CSR must satisfy the invariants, because consumers rely on them without checking:
 * `findWeight` binary-searches the neighbor IDs of a row, so unsorted or duplicate IDs give wrong
 * `S1`, `S2` and symmetry counts. Run this node once after upload (the CSR rarely changes) and
 * read `output.status.invalidCount` before using the weights. There is no GPU sort or dedupe: fix the CSR on
 * the CPU (sort each row by neighbor ID, merge duplicates by summing, drop self slots) and upload
 * it again.
 *
 * Checked per row `i`: `offsets[0] = 0`, offsets non-decreasing and `offsets[rows] <=
 * neighbors.length`; every neighbor ID `< rows`; neighbor IDs strictly ascending within the row
 * (sorted, no duplicates); no self slot; every weight finite and `>= 0`. Slots past
 * `offsets[rows]` are not inspected.
 *
 * {@link GPUSpatialWeightsValidation.outputWeights} is `weights` with `invalidCount` attached to
 * its `status`.
 */
export class GPUSpatialWeightsValidation implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUSpatialWeightsValidationProps;
  /** The checked weights with `status.invalidCount` attached. */
  readonly outputWeights: GPUSpatialWeights;

  constructor(props: GPUSpatialWeightsValidationProps) {
    const id = props.id ?? 'spatial-weights-validation';
    this.id = id;
    this.props = props;
    const rows = validateGPUSpatialWeights(id, props.weights);
    validatePackedView(props.output.status.invalidCount, ['uint32'], `${id} invalidCount`);
    if (props.output.status.invalidCount.length < 1) {
      throw new Error(`${id} invalidCount must hold one uint32`);
    }
    if (props.output.flags) {
      validatePackedView(props.output.flags, ['uint32'], `${id} flags`);
      if (props.output.flags.length !== rows) {
        throw new Error(`${id} flags length must equal the weights row count`);
      }
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      [props.output.status.invalidCount, props.output.flags],
      [props.weights.offsets, props.weights.neighbors, props.weights.weights]
    );
    this.outputWeights = describeGPUSpatialWeights(props.weights, {
      information: props.weights.information ?? {
        rowCount: rows,
        transform: 'unknown',
        symmetric: 'unknown'
      },
      status: {invalidCount: props.output.status.invalidCount}
    });
  }

  /** Returns the validation nodes in dependency order. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {weights} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      weights.offsets,
      weights.neighbors,
      weights.weights,
      props.output.status.invalidCount,
      props.output.flags
    ]);
    const rows = weights.offsets.length - 1;
    const capacity = weights.neighbors.length;
    const violated = createTransientView(graph, `${id}-violated`, 'uint32', rows);
    const flags = props.output.flags ?? createTransientView(graph, `${id}-flags`, 'uint32', rows);
    const bits = GPU_SPATIAL_WEIGHTS_VIOLATION;
    return [
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-rows`,
        operation: OPERATION,
        variant: 'rows',
        bindings: [
          {name: 'offsets', view: weights.offsets, type: 'u32', access: 'read'},
          {name: 'neighbors', view: weights.neighbors, type: 'u32', access: 'read'},
          {name: 'weights', view: weights.weights, type: 'f32', access: 'read'},
          {name: 'flags', view: flags, type: 'u32', access: 'read_write'},
          {name: 'violated', view: violated, type: 'u32', access: 'read_write'}
        ],
        invocationCount: rows,
        declarations: `${IS_FINITE_FLOAT_WGSL}
const ROWS: u32 = ${rows}u;
const CAPACITY: u32 = ${capacity}u;`,
        body: `let begin = offsets[offsetsOffset + index];
  let end = offsets[offsetsOffset + index + 1u];
  var rowFlags = 0u;
  if ((index == 0u && begin != 0u) || begin > end || (index + 1u == ROWS && end > CAPACITY)) {
    rowFlags |= ${bits.offsets}u;
  }
  var previous = 0u;
  for (var slot = begin; slot < min(end, CAPACITY); slot++) {
    let neighbor = neighbors[neighborsOffset + slot];
    let weight = weights[weightsOffset + slot];
    if (neighbor >= ROWS) {
      rowFlags |= ${bits.neighborRange}u;
    }
    if (neighbor == index) {
      rowFlags |= ${bits.self}u;
    }
    if (slot > begin && neighbor <= previous) {
      rowFlags |= ${bits.order}u;
    }
    if (!isFiniteFloat(weight) || weight < 0.0) {
      rowFlags |= ${bits.weight}u;
    }
    previous = neighbor;
  }
  flags[flagsOffset + index] = rowFlags;
  violated[violatedOffset + index] = select(0u, 1u, rowFlags != 0u);`
      }),
      ...new GPUReduction({
        id: `${id}-count`,
        input: violated,
        output: props.output.status.invalidCount,
        operation: 'sum'
      }).getCommandNodes(graph)
    ];
  }
}
