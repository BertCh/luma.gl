// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Binding} from '@luma.gl/core';
import {Kernel} from '@luma.gl/engine';
import {
  createGPUComputeCommandNode,
  getBoundedDispatchLayout,
  getBoundedInvocationIndexSource,
  getViewBinding,
  getViewElementOffset,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphBufferUse,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';

/** One storage binding of a small Frontier Lab compute kernel. */
export type FrontierKernelBinding = {
  name: string;
  view: GraphDataView;
  type: 'u32' | 'i32' | 'f32' | 'atomic<u32>' | 'atomic<i32>';
  access: 'read' | 'read_write';
};

/** Properties of a linear compute pass. Each invocation receives an in-range `index`. */
export type FrontierKernelProps = {
  id: string;
  operation: string;
  variant?: string;
  bindings: readonly FrontierKernelBinding[];
  invocationCount: number;
  workgroupSize?: number;
  declarations?: string;
  body: string;
};

/**
 * Creates a compact linear WGSL graph node from caller-owned views.
 *
 * This intentionally mirrors the contributor kernel utility without importing an internal package
 * path. It keeps the frontier prototypes portable while the reusable v9 recipes mature separately.
 */
export function createFrontierKernelNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: FrontierKernelProps
): GPUCommandNode<Parameters> {
  const workgroupSize = props.workgroupSize ?? 256;
  if (props.bindings.length > graph.device.limits.maxStorageBuffersPerShaderStage) {
    throw new Error(`${props.id} exceeds the device storage-buffer limit`);
  }
  const layout = getBoundedDispatchLayout(
    props.operation,
    Math.max(props.invocationCount, 1),
    workgroupSize,
    graph.device.limits.maxComputeWorkgroupsPerDimension
  );
  const declarations = props.bindings
    .map(
      (binding, location) =>
        `const ${binding.name}Offset: u32 = ${getViewElementOffset(binding.view)}u;\n` +
        `@group(0) @binding(${location}) var<storage, ${binding.access}> ${binding.name}: array<${binding.type}>;`
    )
    .join('\n');
  const references = props.bindings.map(binding => `_ = &${binding.name};`).join('\n  ');
  const source = /* wgsl */ `
const INVOCATION_COUNT: u32 = ${props.invocationCount}u;
${declarations}
${props.declarations ?? ''}

@compute @workgroup_size(${workgroupSize})
fn main(
  @builtin(local_invocation_index) localInvocationIndex: u32,
  @builtin(workgroup_id) workgroupId: vec3<u32>
) {
  ${references}
  ${getBoundedInvocationIndexSource(layout, workgroupSize)}
  if (index >= INVOCATION_COUNT) { return; }
  ${props.body}
}`;
  const resources: GraphBufferUse[] = props.bindings.map(binding => ({
    buffer: binding.view,
    usage: binding.access === 'read' ? 'storage-read' : 'storage-read-write'
  }));
  let readByteLength = 0;
  let writeByteLength = 0;
  for (const binding of props.bindings) {
    const bytes = binding.view.length * binding.view.rowByteLength;
    if (binding.access === 'read') readByteLength += bytes;
    else writeByteLength += bytes;
  }
  return createGPUComputeCommandNode<Parameters>({
    id: props.id,
    resources,
    workload: {
      operation: props.operation,
      ...(props.variant ? {variant: props.variant} : {}),
      commandCount: 1,
      maximumWorkgroupCount: layout.x * layout.y * layout.z,
      maximumInvocationCount: props.invocationCount,
      readByteLength,
      writeByteLength
    },
    compile: ({device}) => {
      const kernel = new Kernel(device, {
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
          const bindings: Record<string, Binding> = {};
          for (const binding of props.bindings) {
            bindings[binding.name] = getViewBinding(binding.view, getBuffer);
          }
          kernel.dispatch(computePass, {bindings, x: layout.x, y: layout.y, z: layout.z});
        },
        destroy: () => kernel.destroy()
      };
    }
  });
}
