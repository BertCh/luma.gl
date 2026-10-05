// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GPUCommandNode, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {
  createFillNode,
  createWGSLKernelNode,
  type WGSLKernelBinding
} from '../../utils/wgsl-kernel-nodes';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {
  getBreakSearchWGSL,
  getRasterAlgebraValueWGSL,
  validateRasterAlgebraAliasing,
  validateRasterAlgebraCount,
  validateRasterAlgebraGraph,
  validateRasterAlgebraNoData,
  validateRasterAlgebraView
} from './raster-algebra-utils';
import {GPU_RASTER_RECLASSIFY_PARAMETER_LENGTH} from './reclassify-parameters';

const OPERATION = 'GPURasterReclassify';

/** Caller-owned outputs of {@link GPURasterReclassify}. Provide any non-empty subset. */
export type GPURasterReclassifyOutput = {
  /** Class per row: the number of breaks below the value, `0xffffffff` for nodata rows. */
  classes?: GraphDataView<'uint32'>;
  /** `classValues[class]` per row, NaN for nodata rows. Requires `classValues`. */
  reclassified?: GraphDataView<'float32'>;
  /** Rows per class, `maximumBreakCount + 1` rows; nodata rows are not counted. */
  classCounts?: GraphDataView<'uint32'>;
};

/**
 * Properties for {@link GPURasterReclassify}.
 *
 * Per-frame (no recompile): the contents of `values`, `validity`, `breaks`, `classValues`, and
 * `parameters` (active break count, interval closure). Topology: `rowCount`, `breaks.length`
 * (the maximum break count), `noDataValue`, and which optional views exist.
 */
export type GPURasterReclassifyProps = {
  /** Prefix for generated node IDs. Defaults to `'raster-reclassify'`. */
  id?: string;
  /** Packed float32 raster or column. NaN rows are nodata. */
  values: GraphDataView<'float32'>;
  /** Rows to classify. Defaults to `values.length`. */
  rowCount?: number;
  /** Optional row-aligned `uint32` validity; zero marks nodata. */
  validity?: GraphDataView<'uint32'>;
  /** Optional finite nodata sentinel compared exactly against values. */
  noDataValue?: number;
  /** Ascending breaks; the view length is the compile-time maximum break count. */
  breaks: GraphDataView<'float32'>;
  /** Optional value per class, at least `breaks.length + 1` rows. */
  classValues?: GraphDataView<'float32'>;
  /** Per-frame float32 view written with `getGPURasterReclassifyParameterValues`. */
  parameters: GraphDataView<'float32'>;
  /** Caller-owned outputs. */
  output: GPURasterReclassifyOutput;
};

/**
 * Reclassifies a float32 raster or column through a per-frame break table.
 *
 * With `n` active ascending breaks `b`, the class of a value is the number of breaks `<= value`
 * (left-closed intervals `[b[k - 1], b[k])`, the default) or `< value` (right-closed), found by
 * binary search, so classes run from `0` (below `b[0]`) to `n`. `reclassified` maps each class
 * through `classValues`, which gives threshold sliders and remap tables (for example
 * `[NaN, 1, 2, 3]` to restrict the lowest class) without recompiling. `classCounts` counts rows
 * per class with integer atomics, so it is deterministic.
 */
export class GPURasterReclassify implements GPUCommandNodeProducer {
  /** Prefix for every node ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPURasterReclassifyProps;
  /** Rows classified per encoding. */
  readonly rowCount: number;
  /** Compile-time maximum break count (`breaks.length`). */
  readonly maximumBreakCount: number;

  constructor(props: GPURasterReclassifyProps) {
    this.id = props.id ?? 'raster-reclassify';
    this.props = props;
    const {id} = this;
    const {output} = props;
    validateRasterAlgebraView(id, 'values', props.values, 'float32', 0);
    this.rowCount = props.rowCount ?? props.values.length;
    validateRasterAlgebraCount(id, 'rowCount', this.rowCount, 0, props.values.length);
    validateRasterAlgebraNoData(id, props.noDataValue);
    validateRasterAlgebraView(id, 'validity', props.validity, 'uint32', this.rowCount);
    validateRasterAlgebraView(id, 'breaks', props.breaks, 'float32', 1);
    this.maximumBreakCount = props.breaks.length;
    validateRasterAlgebraView(
      id,
      'classValues',
      props.classValues,
      'float32',
      this.maximumBreakCount + 1
    );
    validateRasterAlgebraView(
      id,
      'parameters',
      props.parameters,
      'float32',
      GPU_RASTER_RECLASSIFY_PARAMETER_LENGTH
    );
    if (!output.classes && !output.reclassified && !output.classCounts) {
      throw new Error(`${id} needs at least one output`);
    }
    if (output.reclassified && !props.classValues) {
      throw new Error(`${id} output.reclassified requires classValues`);
    }
    validateRasterAlgebraView(id, 'output.classes', output.classes, 'uint32', this.rowCount);
    validateRasterAlgebraView(
      id,
      'output.reclassified',
      output.reclassified,
      'float32',
      this.rowCount
    );
    validateRasterAlgebraView(
      id,
      'output.classCounts',
      output.classCounts,
      'uint32',
      this.maximumBreakCount + 1
    );
    validateRasterAlgebraAliasing(
      id,
      [output.classes, output.reclassified, output.classCounts],
      [props.values, props.validity, props.breaks, props.classValues, props.parameters]
    );
  }

  /** Returns an optional class-count clear, then one classification kernel. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, rowCount, maximumBreakCount} = this;
    const {output} = props;
    validateRasterAlgebraGraph(id, graph, [
      props.values,
      props.validity,
      props.breaks,
      props.classValues,
      props.parameters,
      output.classes,
      output.reclassified,
      output.classCounts
    ]);
    const nodes: GPUCommandNode<Parameters>[] = [];
    if (output.classCounts) {
      nodes.push(
        createFillNode<Parameters>(graph, {
          id: `${id}-clear-counts`,
          operation: OPERATION,
          view: output.classCounts,
          type: 'u32',
          value: '0u',
          componentCount: maximumBreakCount + 1
        })
      );
    }
    const bindings: WGSLKernelBinding[] = [
      {name: 'source', view: props.values, type: 'f32', access: 'read'},
      {name: 'breaks', view: props.breaks, type: 'f32', access: 'read'},
      {name: 'params', view: props.parameters, type: 'f32', access: 'read'}
    ];
    if (props.validity) {
      bindings.push({name: 'validity', view: props.validity, type: 'u32', access: 'read'});
    }
    if (props.classValues) {
      bindings.push({name: 'classValues', view: props.classValues, type: 'f32', access: 'read'});
    }
    if (output.classes) {
      bindings.push({name: 'classesOut', view: output.classes, type: 'u32', access: 'read_write'});
    }
    if (output.reclassified) {
      bindings.push({
        name: 'reclassifiedOut',
        view: output.reclassified,
        type: 'f32',
        access: 'read_write'
      });
    }
    if (output.classCounts) {
      bindings.push({
        name: 'countsOut',
        view: output.classCounts,
        type: 'atomic<u32>',
        access: 'read_write'
      });
    }
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-classify`,
        operation: OPERATION,
        variant: 'classify',
        bindings,
        invocationCount: rowCount,
        declarations: `const MAXIMUM_BREAK_COUNT: u32 = ${maximumBreakCount}u;
const NO_DATA_CLASS: u32 = 0xffffffffu;
${getRasterAlgebraValueWGSL(props.noDataValue)}
${getBreakSearchWGSL('countBreaksBelow', 'breaks')}`,
        body: `let value = source[sourceOffset + index];
  var isValid = !isNoDataValue(value);
  ${props.validity ? 'isValid = isValid && validity[validityOffset + index] != 0u;' : ''}
  let breakCountValue = params[paramsOffset];
  let breakCount = min(u32(clamp(select(0.0, breakCountValue, isFiniteValue(breakCountValue)), 0.0, 4294967040.0)), MAXIMUM_BREAK_COUNT);
  let closedRight = params[paramsOffset + 1u] != 0.0;
  var classIndex = NO_DATA_CLASS;
  if (isValid) {
    classIndex = countBreaksBelow(0u, breakCount, value, closedRight);
  }
  ${output.classes ? 'classesOut[classesOutOffset + index] = classIndex;' : ''}
  ${
    output.reclassified
      ? `reclassifiedOut[reclassifiedOutOffset + index] = select(getNaN(), classValues[classValuesOffset + min(classIndex, MAXIMUM_BREAK_COUNT)], isValid);`
      : ''
  }
  ${output.classCounts ? 'if (isValid) {\n    atomicAdd(&countsOut[countsOutOffset + classIndex], 1u);\n  }' : ''}`
      })
    );
    return nodes;
  }
}
