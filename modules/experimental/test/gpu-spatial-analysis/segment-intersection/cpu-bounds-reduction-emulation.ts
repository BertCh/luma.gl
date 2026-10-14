// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// CPU-executable stand-in for `getBoundsReductionNodes`
// (src/gpu-spatial-analysis/hilbert-keys/bounds-reduction.ts), for the CPU graph executor
// (wgsl-graph-executor.ts). The real reduction tree-reduces with workgroup barriers, which the
// interpreter cannot run. This version keeps the contract (same props, same `readItem` WGSL, same
// four-row `[minX, minY, maxX, maxY]` output, all zero when no item is valid) but scans every item
// in one invocation with no barrier, so the REAL `readItem` of the caller still executes.
//
// Specs call useCPUExecutorTestHooks() to install this function alongside the gpu-core
// emulations for each case. The hooks restore all methods afterward; do not use module mocks
// with the node project's shared module graph (isolate:false).

import type {GPUCommandGraph, GPUCommandNode, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {createWGSLKernelNode, type WGSLKernelBinding} from '../../../src/utils/wgsl-kernel-nodes';

/** Same props as `getBoundsReductionNodes`. */
export type CPUBoundsReductionProps = {
  id: string;
  operation: string;
  variant: string;
  bindings: readonly WGSLKernelBinding[];
  declarations: string;
  itemCount: number;
  output: GraphDataView<'float32'>;
};

/** Barrier-free single-invocation equivalent of `getBoundsReductionNodes`. */
export function getBoundsReductionNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: CPUBoundsReductionProps
): GPUCommandNode<Parameters>[] {
  return [
    createWGSLKernelNode<Parameters>(graph, {
      id: `${props.id}-bounds`,
      operation: props.operation,
      variant: `${props.variant}-cpu`,
      bindings: [
        ...props.bindings,
        {name: 'boundsOut', view: props.output, type: 'f32', access: 'read_write'}
      ],
      invocationCount: 1,
      declarations: `const ITEM_COUNT: u32 = ${props.itemCount}u;
const FLOAT32_MAXIMUM: f32 = 3.402823466e+38;
${props.declarations}`,
      body: `var localMinimum = vec2f(FLOAT32_MAXIMUM);
  var localMaximum = vec2f(-FLOAT32_MAXIMUM);
  for (var row = 0u; row < ITEM_COUNT; row++) {
    let item = readItem(row);
    if (item.z > 0.5) {
      localMinimum = min(localMinimum, item.xy);
      localMaximum = max(localMaximum, item.xy);
    }
  }
  let hasValid = localMinimum.x <= localMaximum.x && localMinimum.y <= localMaximum.y;
  boundsOut[boundsOutOffset] = select(0.0, localMinimum.x, hasValid);
  boundsOut[boundsOutOffset + 1u] = select(0.0, localMinimum.y, hasValid);
  boundsOut[boundsOutOffset + 2u] = select(0.0, localMaximum.x, hasValid);
  boundsOut[boundsOutOffset + 3u] = select(0.0, localMaximum.y, hasValid);`
    })
  ];
}

/** The stand-in, keyed by the module export it replaces. */
export const CPU_BOUNDS_REDUCTION_EMULATION = {getBoundsReductionNodes};
