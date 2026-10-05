// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GPUCommandNode, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {createWGSLKernelNode, type WGSLKernelBinding} from '../../utils/wgsl-kernel-nodes';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {validateGraphViewsBelongToGraph} from '../../utils/gpu-contributor-utils';
import {GPU_RASTER_SAMPLING_PARAMETER_LENGTH} from './raster-sampling-parameters';
import {
  getRasterSamplingWGSL,
  validateRasterDescription,
  validateRasterSamplingView
} from './raster-sampling-utils';

const OPERATION = 'GPURasterSampling';

/** Caller-owned outputs of {@link GPURasterSampling}. */
export type GPURasterSamplingOutput = {
  /** Sampled value per point; NaN outside the extent, for nodata results and for inactive rows. */
  values: GraphDataView<'float32'>;
  /** Optional `1` where the sampled value is not NaN, otherwise `0`. */
  validity?: GraphDataView<'uint32'>;
};

/**
 * Properties for {@link GPURasterSampling}.
 *
 * Per-frame (no recompile): raster, position and `pointCount` contents and `parameters` (extent,
 * method, nodata policy). Topology: `width`, `height`, `noDataValue`, which optional views exist.
 */
export type GPURasterSamplingProps = {
  /** Prefix for generated node IDs. Defaults to `'raster-sampling'`. */
  id?: string;
  /** Raster width in cells. */
  width: number;
  /** Raster height in cells. */
  height: number;
  /** Row-major raster of at least `width * height` rows; row 0 is at `minY`. NaN is nodata. */
  values: GraphDataView<'float32'>;
  /** Optional finite sentinel treated as nodata. */
  noDataValue?: number;
  /** Optional raster validity; zero marks a nodata cell. */
  validity?: GraphDataView<'uint32'>;
  /** Packed points in extent units. */
  positions: GraphDataView<'float32x2'>;
  /** Optional one-row active point count, clamped to `positions.length`. */
  pointCount?: GraphDataView<'uint32'>;
  /** Per-frame float32 view written with `getGPURasterSamplingParameterValues`. */
  parameters: GraphDataView<'float32'>;
  /** Caller-owned outputs with at least `positions.length` rows. */
  output: GPURasterSamplingOutput;
};

/**
 * Samples a float32 raster at points ("extract raster values to points").
 *
 * The pixel coordinate is `u = (x - minX) * (1 / cellWidth) - 0.5`, so integer `u` is a cell centre.
 * Points outside the closed extent rectangle (or with NaN coordinates) give NaN. Inside it, indices
 * beyond the outermost cell centres clamp to the edge cell.
 *
 * - `nearest`: the cell containing the point (`floor((x - minX) / cellWidth)`, clamped, so the
 *   `maxX`/`maxY` edge belongs to the last cell). A nodata cell gives NaN under both policies.
 * - `bilinear`: the four surrounding cell centres.
 * - `bicubic`: Catmull-Rom (a = -0.5) over the 4x4 support with clamped indices.
 *
 * A neighbour participates only when both of its axis weights are nonzero, so a point exactly on a
 * cell centre returns that cell whatever its neighbours hold. Under `'strict'` any participating
 * nodata cell gives NaN. Under `'renormalize'` bilinear weights over valid neighbours are divided
 * by their sum (NaN when none is valid). Bicubic with nodata in its participating support falls
 * back to the bilinear rule of the same policy.
 *
 * Rows at or beyond the active `pointCount` are written NaN and 0.
 *
 * Precision: nearest is exact. Bilinear and bicubic evaluate in f32; GPUs may contract
 * multiply-adds into FMA, so results agree with an f32 CPU oracle to a few ULP, and exactly when
 * the arithmetic is exact (dyadic coordinates).
 */
export class GPURasterSampling implements GPUCommandNodeProducer {
  /** Prefix for every node ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPURasterSamplingProps;

  constructor(props: GPURasterSamplingProps) {
    this.id = props.id ?? 'raster-sampling';
    this.props = props;
    const {id} = this;
    const {width, height, output} = props;
    validateRasterDescription(id, width, height, props.noDataValue);
    validateRasterSamplingView(id, 'values', props.values, 'float32', width * height);
    if (props.validity) {
      validateRasterSamplingView(id, 'validity', props.validity, 'uint32', width * height);
    }
    validateRasterSamplingView(id, 'positions', props.positions, 'float32x2', 0);
    if (props.pointCount) {
      validateRasterSamplingView(id, 'pointCount', props.pointCount, 'uint32', 1);
    }
    validateRasterSamplingView(
      id,
      'parameters',
      props.parameters,
      'float32',
      GPU_RASTER_SAMPLING_PARAMETER_LENGTH
    );
    if (!output?.values) {
      throw new Error(`${id} needs output.values`);
    }
    const capacity = props.positions.length;
    validateRasterSamplingView(id, 'output.values', output.values, 'float32', capacity);
    if (output.validity) {
      validateRasterSamplingView(id, 'output.validity', output.validity, 'uint32', capacity);
    }
    const outputs = [output.values, output.validity].filter(view => view !== undefined);
    const inputs = [
      props.values,
      props.validity,
      props.positions,
      props.pointCount,
      props.parameters
    ];
    if (new Set(outputs.map(view => view.buffer)).size !== outputs.length) {
      throw new Error(`${id} outputs must not share buffers`);
    }
    const inputBuffers = new Set(
      inputs.filter(view => view !== undefined).map(view => view.buffer)
    );
    if (outputs.some(view => inputBuffers.has(view.buffer))) {
      throw new Error(`${id} outputs must not share buffers with inputs`);
    }
  }

  /** Returns one per-point kernel. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {output} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      props.values,
      props.validity,
      props.positions,
      props.pointCount,
      props.parameters,
      output.values,
      output.validity
    ]);
    const bindings: WGSLKernelBinding[] = [
      {name: 'raster', view: props.values, type: 'f32', access: 'read'}
    ];
    if (props.validity) {
      bindings.push({name: 'validity', view: props.validity, type: 'u32', access: 'read'});
    }
    bindings.push({name: 'positions', view: props.positions, type: 'f32', access: 'read'});
    if (props.pointCount) {
      bindings.push({name: 'pointCount', view: props.pointCount, type: 'u32', access: 'read'});
    }
    bindings.push(
      {name: 'params', view: props.parameters, type: 'f32', access: 'read'},
      {name: 'valuesOut', view: output.values, type: 'f32', access: 'read_write'}
    );
    if (output.validity) {
      bindings.push({
        name: 'validityOut',
        view: output.validity,
        type: 'u32',
        access: 'read_write'
      });
    }
    return [
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-sample`,
        operation: OPERATION,
        variant: 'sample',
        bindings,
        invocationCount: props.positions.length,
        declarations: `${getRasterSamplingWGSL({
          width: props.width,
          height: props.height,
          noDataValue: props.noDataValue,
          hasValidity: Boolean(props.validity)
        })}
const POINT_CAPACITY: u32 = ${props.positions.length}u;`,
        body: `let activeCount = ${props.pointCount ? 'min(pointCount[pointCountOffset], POINT_CAPACITY)' : 'POINT_CAPACITY'};
  var result = getNaN();
  if (index < activeCount) {
    result = sampleRaster(
      positions[positionsOffset + 2u * index],
      positions[positionsOffset + 2u * index + 1u]
    );
  }
  valuesOut[valuesOutOffset + index] = result;
  ${output.validity ? 'validityOut[validityOutOffset + index] = select(1u, 0u, isNaNValue(result));' : ''}`
      })
    ];
  }
}
