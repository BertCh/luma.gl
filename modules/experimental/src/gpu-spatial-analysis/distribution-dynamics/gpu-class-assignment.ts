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

const OPERATION = 'GPUClassAssignment';

/** Class value written for a masked row or a NaN value. */
export const GPU_CLASS_ASSIGNMENT_NO_CLASS = 0xffffffff;

/**
 * Properties for {@link GPUClassAssignment}.
 *
 * Per-frame: the contents of every view. Topology: view lengths and whether `mask` is present.
 */
export type GPUClassAssignmentProps = {
  /** Prefix for generated node IDs. Defaults to `'class-assignment'`. */
  id?: string;
  /**
   * Packed float32 column. For distribution dynamics this is the pooled column of every period
   * (`rows * periods` entries, period-major), so that all periods share one set of breaks.
   */
  values: GraphDataView<'float32'>;
  /**
   * Class edges `e[0..k]`, at least `classCount + 1` rows, for example the `breaks` of a
   * `GPUClassBreaks` run on `values`. Only inner edges `e[1..k-1]` are compared.
   */
  breaks: GraphDataView<'float32'>;
  /** One row holding the class count `k`, for example the `classCount` of `GPUClassBreaks`. */
  classCount: GraphDataView<'uint32'>;
  /** Optional per-row selection (same length as `values`): zero rows get {@link GPU_CLASS_ASSIGNMENT_NO_CLASS}. */
  mask?: GraphDataView<'uint32'>;
  /** Caller-owned class per row, `0..k-1`. */
  output: GraphDataView<'uint32'>;
};

/**
 * Assigns each row of a column the class of the shared breaks: the number of inner edges
 * `e[1..k-1]` that are less than or equal to the value, clamped to `k - 1`. This is the class rule
 * of `GPUClassBreaks.classCounts` (infinities clamp to the end classes). NaN and masked rows get
 * {@link GPU_CLASS_ASSIGNMENT_NO_CLASS}. When `k` is 0 every row gets that value.
 *
 * Running one `GPUClassBreaks` over the pooled column of all periods and assigning every period
 * with these breaks keeps a single legend across a time slider, as giddy's pooled classification
 * does. The result feeds `GPUTransitionMatrix`.
 *
 * Integer-exact: a pure function of the float values and the edges.
 */
export class GPUClassAssignment implements GPUCommandNodeProducer {
  /** Prefix for every node ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUClassAssignmentProps;

  constructor(props: GPUClassAssignmentProps) {
    const id = props.id ?? 'class-assignment';
    this.id = id;
    this.props = props;
    validatePackedView(props.values, ['float32'], `${id} values`);
    validatePackedView(props.breaks, ['float32'], `${id} breaks`);
    validatePackedUint32View(props.classCount, `${id} classCount`);
    validatePackedUint32View(props.output, `${id} output`);
    if (props.classCount.length < 1) {
      throw new Error(`${id} classCount must hold one row`);
    }
    if (props.breaks.length < 2) {
      throw new Error(`${id} breaks must hold at least two rows`);
    }
    if (props.output.length !== props.values.length) {
      throw new Error(`${id} output length must equal values length`);
    }
    if (props.mask) {
      validatePackedUint32View(props.mask, `${id} mask`);
      if (props.mask.length !== props.values.length) {
        throw new Error(`${id} mask length must equal values length`);
      }
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      [props.output],
      [props.values, props.breaks, props.classCount, props.mask]
    );
  }

  /** Returns the assignment node. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {values, breaks, classCount, mask, output} = props;
    validateGraphViewsBelongToGraph(id, graph, [values, breaks, classCount, mask, output]);
    return [
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-assign`,
        operation: OPERATION,
        bindings: [
          {name: 'values', view: values, type: 'f32', access: 'read'},
          {name: 'breaks', view: breaks, type: 'f32', access: 'read'},
          {name: 'classCount', view: classCount, type: 'u32', access: 'read'},
          ...(mask
            ? [{name: 'mask', view: mask, type: 'u32' as const, access: 'read' as const}]
            : []),
          {name: 'output', view: output, type: 'u32', access: 'read_write'}
        ],
        invocationCount: values.length,
        declarations: `const BREAK_ROWS: u32 = ${breaks.length}u;`,
        body: `let value = values[valuesOffset + index];
  let count = min(classCount[classCountOffset], BREAK_ROWS - 1u);
  var assigned = ${GPU_CLASS_ASSIGNMENT_NO_CLASS}u;
  // WGSL gives no NaN guarantees to comparisons, so test the exponent and mantissa bits.
  let isNaN = (bitcast<u32>(value) & 0x7fffffffu) > 0x7f800000u;
  if (count > 0u && !isNaN && ${mask ? 'mask[maskOffset + index] != 0u' : 'true'}) {
    var below = 0u;
    for (var edge = 1u; edge < count; edge++) {
      if (breaks[breaksOffset + edge] <= value) {
        below++;
      }
    }
    assigned = below;
  }
  output[outputOffset + index] = assigned;`
      })
    ];
  }
}
