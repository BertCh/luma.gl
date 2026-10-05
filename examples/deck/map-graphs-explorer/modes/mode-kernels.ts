// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * A small linear compute-node builder shared by the Weights and Pattern modes, the public
 * equivalent of the recipes' internal kernel helper: one 1D dispatch, storage bindings declared by
 * name, a WGSL body that runs once per in-range `index`.
 */

import type {Binding} from '@luma.gl/core';
import {Computation} from '@luma.gl/engine';
import {
  getViewBinding,
  getViewElementOffset,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';

const WORKGROUP_SIZE = 256;

/** One storage binding of a {@link addKernelPass} kernel. */
export type KernelBinding = {
  /** WGSL variable name. `${name}Offset` holds the view's element offset. */
  name: string;
  view: GraphDataView;
  type: 'u32' | 'f32';
  access: 'read' | 'read_write';
};

/**
 * Adds one linear compute pass to `graph`: `invocationCount` invocations in a 1D dispatch of
 * `ceil(invocationCount / 256)` workgroups, each running `body` with `index` in range.
 */
export function addKernelPass<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    bindings: readonly KernelBinding[];
    invocationCount: number;
    declarations?: string;
    body: string;
  }
): void {
  const workgroupCount = Math.ceil(props.invocationCount / WORKGROUP_SIZE);
  if (workgroupCount > graph.device.limits.maxComputeWorkgroupsPerDimension) {
    throw new Error(`${props.id} needs more workgroups than one dispatch dimension allows`);
  }
  const declarations = props.bindings
    .map(
      (
        binding,
        location
      ) => `const ${binding.name}Offset: u32 = ${getViewElementOffset(binding.view)}u;
@group(0) @binding(${location}) var<storage, ${binding.access}> ${binding.name}: array<${binding.type}>;`
    )
    .join('\n');
  const source = /* wgsl */ `
const INVOCATION_COUNT: u32 = ${props.invocationCount}u;
${declarations}
${props.declarations ?? ''}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) invocation: vec3<u32>) {
  let index = invocation.x;
  if (index >= INVOCATION_COUNT) {
    return;
  }
  ${props.body}
}`;
  let readByteLength = 0;
  let writeByteLength = 0;
  for (const binding of props.bindings) {
    const byteLength = binding.view.length * binding.view.rowByteLength;
    if (binding.access === 'read') readByteLength += byteLength;
    else writeByteLength += byteLength;
  }
  graph.addComputePass({
    id: props.id,
    workload: {
      operation: 'ExplorerModeKernel',
      variant: props.id,
      commandCount: 1,
      maximumWorkgroupCount: workgroupCount,
      maximumInvocationCount: workgroupCount * WORKGROUP_SIZE,
      readByteLength,
      writeByteLength
    },
    resources: props.bindings.map(binding => ({
      buffer: binding.view,
      usage: binding.access === 'read' ? ('storage-read' as const) : ('storage-read-write' as const)
    })),
    compile: ({device}) => {
      const computation = new Computation(device, {
        id: props.id,
        source,
        shaderLayout: {
          bindings: props.bindings.map((binding, location) => ({
            name: binding.name,
            type: binding.access === 'read' ? ('read-only-storage' as const) : ('storage' as const),
            group: 0,
            location
          }))
        }
      });
      return {
        encode: ({computePass, getBuffer}) => {
          const resolved: Record<string, Binding> = {};
          for (const binding of props.bindings) {
            resolved[binding.name] = getViewBinding(binding.view, getBuffer);
          }
          computation.setBindings(resolved);
          computation.dispatch(computePass, workgroupCount);
        },
        destroy: () => computation.destroy()
      };
    }
  });
}
