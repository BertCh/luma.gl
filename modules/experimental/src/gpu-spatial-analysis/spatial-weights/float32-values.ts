// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {validatePackedView} from '../../utils/gpu-contributor-utils';
import {createWGSLKernelNode} from '../../utils/wgsl-kernel-nodes';

/** A count-like or value column accepted as `float32` or `uint32` (exact in f32 below 2^24). */
export type GPUFloat32OrUint32Values = GraphDataView<'float32'> | GraphDataView<'uint32'>;

/**
 * Validates a `float32 | uint32` column and returns its length. Counts from one contributor (for
 * example the `uint32` `counts` of a points-in-polygons aggregation) then chain into statistics
 * without a hand-written cast.
 *
 * @param name Prop name prefixed with the contributor ID, for the message.
 * @internal
 */
export function validateFloat32OrUint32Values(
  view: GPUFloat32OrUint32Values,
  name: string
): number {
  validatePackedView(view, ['float32', 'uint32'], name);
  return view.length;
}

/**
 * Returns a float32 view of a `float32 | uint32` column: `view` itself when it is float32, else a
 * graph-owned transient written by one conversion node (`f32(u)`, exact below 2^24). Call it from
 * `getCommandNodes` and put `nodes` before the consumers of `view`.
 *
 * @internal
 */
export function createFloat32ValuesNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  view: GPUFloat32OrUint32Values
): {view: GraphDataView<'float32'>; nodes: GPUCommandNode<Parameters>[]} {
  if (view.format === 'float32') {
    return {view: view as GraphDataView<'float32'>, nodes: []};
  }
  const output = createTransientView(graph, `${id}-float32`, 'float32', view.length);
  return {
    view: output,
    nodes: [
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-to-float32`,
        operation: 'GPUFloat32ValuesConversion',
        bindings: [
          {name: 'counts', view, type: 'u32', access: 'read'},
          {name: 'output', view: output, type: 'f32', access: 'read_write'}
        ],
        invocationCount: view.length,
        body: 'output[outputOffset + index] = f32(counts[countsOffset + index]);'
      })
    ]
  };
}
