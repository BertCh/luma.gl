// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GPUCommandNode, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {createWGSLKernelNode, type WGSLKernelBinding} from '../../utils/wgsl-kernel-nodes';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {GPU_RASTER_CONDITIONAL_PARAMETER_LENGTH} from './local-operations-parameters';
import {
  getRasterAlgebraValueWGSL,
  validateRasterAlgebraAliasing,
  validateRasterAlgebraCount,
  validateRasterAlgebraGraph,
  validateRasterAlgebraNoData,
  validateRasterAlgebraView
} from './raster-algebra-utils';

const OPERATION = 'GPURasterConditional';

/** Caller-owned outputs of {@link GPURasterConditional}. */
export type GPURasterConditionalOutput = {
  /** `condition ? a : b` per cell; NaN where the condition or the chosen operand is nodata. */
  values: GraphDataView<'float32'>;
  /** Optional condition result: `1` true, `0` false or nodata. */
  mask?: GraphDataView<'uint32'>;
};

/**
 * Properties for {@link GPURasterConditional}. Provide exactly one of `mask` and
 * `conditionValues`.
 *
 * Per-frame (no recompile): input contents and `parameters` (comparison, thresholds, constants).
 * Topology: `cellCount`, `noDataValue`, and which views exist (a missing `a` or `b` uses its
 * per-frame constant).
 */
export type GPURasterConditionalProps = {
  /** Prefix for generated node IDs. Defaults to `'raster-conditional'`. */
  id?: string;
  /** Cells to evaluate. */
  cellCount: number;
  /** Boolean condition raster: nonzero is true. */
  mask?: GraphDataView<'uint32'>;
  /** Float condition raster compared per frame against thresholds. NaN cells are nodata. */
  conditionValues?: GraphDataView<'float32'>;
  /** Value where the condition holds. Defaults to the per-frame `constantA`. */
  a?: GraphDataView<'float32'>;
  /** Value where the condition fails. Defaults to the per-frame `constantB`. */
  b?: GraphDataView<'float32'>;
  /** Optional finite sentinel treated as nodata in `conditionValues`, `a`, and `b`. */
  noDataValue?: number;
  /** Per-frame float32 view written with `getGPURasterConditionalParameterValues`. */
  parameters: GraphDataView<'float32'>;
  /** Caller-owned outputs. */
  output: GPURasterConditionalOutput;
};

/**
 * Per-cell `where(condition, a, b)` (ArcGIS `Con`, numpy `where`).
 *
 * The condition is a `uint32` mask or a float raster compared against per-frame thresholds
 * (`<`, `<=`, `>`, `>=`, `==`, `!=`, inclusive `between`); `a` and `b` are rasters or per-frame
 * constants. Changing the comparison, thresholds, or constants never recompiles.
 */
export class GPURasterConditional implements GPUCommandNodeProducer {
  /** Prefix for every node ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPURasterConditionalProps;

  constructor(props: GPURasterConditionalProps) {
    this.id = props.id ?? 'raster-conditional';
    this.props = props;
    const {id} = this;
    const {cellCount, output} = props;
    validateRasterAlgebraCount(id, 'cellCount', cellCount, 0, 0xffffffff);
    validateRasterAlgebraNoData(id, props.noDataValue);
    if (Boolean(props.mask) === Boolean(props.conditionValues)) {
      throw new Error(`${id} needs exactly one of mask and conditionValues`);
    }
    validateRasterAlgebraView(id, 'mask', props.mask, 'uint32', cellCount);
    validateRasterAlgebraView(id, 'conditionValues', props.conditionValues, 'float32', cellCount);
    validateRasterAlgebraView(id, 'a', props.a, 'float32', cellCount);
    validateRasterAlgebraView(id, 'b', props.b, 'float32', cellCount);
    validateRasterAlgebraView(
      id,
      'parameters',
      props.parameters,
      'float32',
      GPU_RASTER_CONDITIONAL_PARAMETER_LENGTH
    );
    if (!output?.values) {
      throw new Error(`${id} needs output.values`);
    }
    validateRasterAlgebraView(id, 'output.values', output.values, 'float32', cellCount);
    validateRasterAlgebraView(id, 'output.mask', output.mask, 'uint32', cellCount);
    validateRasterAlgebraAliasing(
      id,
      [output.values, output.mask],
      [props.mask, props.conditionValues, props.a, props.b, props.parameters]
    );
  }

  /** Returns one per-cell kernel. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {output} = props;
    validateRasterAlgebraGraph(id, graph, [
      props.mask,
      props.conditionValues,
      props.a,
      props.b,
      props.parameters,
      output.values,
      output.mask
    ]);
    const bindings: WGSLKernelBinding[] = [
      {name: 'params', view: props.parameters, type: 'f32', access: 'read'}
    ];
    if (props.mask) {
      bindings.push({name: 'conditionMask', view: props.mask, type: 'u32', access: 'read'});
    }
    if (props.conditionValues) {
      bindings.push({
        name: 'conditionValues',
        view: props.conditionValues,
        type: 'f32',
        access: 'read'
      });
    }
    if (props.a) {
      bindings.push({name: 'aValues', view: props.a, type: 'f32', access: 'read'});
    }
    if (props.b) {
      bindings.push({name: 'bValues', view: props.b, type: 'f32', access: 'read'});
    }
    bindings.push({name: 'valuesOut', view: output.values, type: 'f32', access: 'read_write'});
    if (output.mask) {
      bindings.push({name: 'maskOut', view: output.mask, type: 'u32', access: 'read_write'});
    }
    return [
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-select`,
        operation: OPERATION,
        variant: 'select',
        bindings,
        invocationCount: props.cellCount,
        declarations: getRasterAlgebraValueWGSL(props.noDataValue),
        body: `var conditionDefined = true;
  var condition = false;
  ${
    props.mask
      ? 'condition = conditionMask[conditionMaskOffset + index] != 0u;'
      : `let conditionValue = conditionValues[conditionValuesOffset + index];
  conditionDefined = !isNoDataValue(conditionValue);
  let comparison = u32(params[paramsOffset]);
  let threshold = params[paramsOffset + 1u];
  let upperThreshold = params[paramsOffset + 2u];
  switch comparison {
    case 0u: { condition = conditionValue < threshold; }
    case 1u: { condition = conditionValue <= threshold; }
    case 2u: { condition = conditionValue > threshold; }
    case 3u: { condition = conditionValue >= threshold; }
    case 4u: { condition = conditionValue == threshold; }
    case 5u: { condition = conditionValue != threshold; }
    default: { condition = conditionValue >= threshold && conditionValue <= upperThreshold; }
  }`
  }
  let aValue = ${props.a ? 'aValues[aValuesOffset + index]' : 'params[paramsOffset + 3u]'};
  let bValue = ${props.b ? 'bValues[bValuesOffset + index]' : 'params[paramsOffset + 4u]'};
  let chosen = select(bValue, aValue, condition);
  let isDefined = conditionDefined && !isNoDataValue(chosen);
  valuesOut[valuesOutOffset + index] = select(getNaN(), chosen, isDefined);
  ${output.mask ? 'maskOut[maskOutOffset + index] = select(0u, 1u, conditionDefined && condition);' : ''}`
      })
    ];
  }
}
