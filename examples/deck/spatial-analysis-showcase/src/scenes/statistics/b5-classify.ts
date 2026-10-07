// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';
import {Computation} from '@luma.gl/engine';
import type {Binding} from '@luma.gl/core';
import {
  getViewBinding,
  getViewElementOffset,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {importGraphBuffer} from '../../engine/graph-buffers';

/**
 * Returns an importer that binds each buffer once per graph. A graph rejects two imports that
 * resolve to one physical buffer, so every node that reads or writes a buffer shares one view.
 */
export function createGraphImporter(graph: GPUCommandGraph<void>) {
  const floatViews = new Map<Buffer, GraphDataView<'float32'>>();
  const wordViews = new Map<Buffer, GraphDataView<'uint32'>>();
  return {
    float: (buffer: Buffer, length: number) => {
      let view = floatViews.get(buffer);
      if (!view) {
        view = importGraphBuffer(graph, buffer.id, buffer, 'float32', length);
        floatViews.set(buffer, view);
      }
      return view;
    },
    word: (buffer: Buffer, length: number) => {
      let view = wordViews.get(buffer);
      if (!view) {
        view = importGraphBuffer(graph, buffer.id, buffer, 'uint32', length);
        wordViews.set(buffer, view);
      }
      return view;
    }
  };
}

/**
 * Adds a one-invocation-per-edge pass that copies one of two `GPUClassBreaks` results into the
 * shared `breaks` and `classCount` views: the alternate pair when the per-frame method code in
 * `parameters[0]` equals `alternateMethodCode`, otherwise the primary pair.
 *
 * One `GPUClassBreaks` that compiles quantile, standard-deviation and head/tail together binds
 * nine storage buffers in its finish kernel, one over the default limit, so head/tail splits
 * into a second instance and this pass joins them without a recompile.
 */
export function addBreaksSelectPass<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    maximumClassCount: number;
    parameters: GraphDataView<'float32'>;
    alternateMethodCode: number;
    primaryBreaks: GraphDataView<'float32'>;
    primaryClassCount: GraphDataView<'uint32'>;
    alternateBreaks: GraphDataView<'float32'>;
    alternateClassCount: GraphDataView<'uint32'>;
    breaks: GraphDataView<'float32'>;
    classCount: GraphDataView<'uint32'>;
  }
): void {
  const edgeCount = props.maximumClassCount + 1;
  const bindings = [
    {name: 'parameters', view: props.parameters, access: 'read' as const, type: 'f32'},
    {name: 'primaryBreaks', view: props.primaryBreaks, access: 'read' as const, type: 'f32'},
    {
      name: 'primaryClassCount',
      view: props.primaryClassCount,
      access: 'read' as const,
      type: 'u32'
    },
    {name: 'alternateBreaks', view: props.alternateBreaks, access: 'read' as const, type: 'f32'},
    {
      name: 'alternateClassCount',
      view: props.alternateClassCount,
      access: 'read' as const,
      type: 'u32'
    },
    {name: 'selectedBreaks', view: props.breaks, access: 'read_write' as const, type: 'f32'},
    {name: 'selectedClassCount', view: props.classCount, access: 'read_write' as const, type: 'u32'}
  ];
  const declarations = bindings
    .map(
      (
        binding,
        location
      ) => `const ${binding.name}Offset: u32 = ${getViewElementOffset(binding.view)}u;
@group(0) @binding(${location}) var<storage, ${binding.access}> ${binding.name}: array<${binding.type}>;`
    )
    .join('\n');
  const source = /* wgsl */ `
${declarations}
@compute @workgroup_size(${edgeCount})
fn main(@builtin(local_invocation_id) invocation: vec3<u32>) {
  let useAlternate = u32(parameters[parametersOffset]) == ${props.alternateMethodCode}u;
  let index = invocation.x;
  selectedBreaks[selectedBreaksOffset + index] = select(
    primaryBreaks[primaryBreaksOffset + index],
    alternateBreaks[alternateBreaksOffset + index],
    useAlternate
  );
  if (index == 0u) {
    selectedClassCount[selectedClassCountOffset] = select(
      primaryClassCount[primaryClassCountOffset],
      alternateClassCount[alternateClassCountOffset],
      useAlternate
    );
  }
}`;
  graph.addComputePass({
    id: props.id,
    workload: {
      operation: 'ClassificationBreaksSelect',
      variant: props.id,
      commandCount: 1,
      maximumWorkgroupCount: 1,
      maximumInvocationCount: edgeCount,
      readByteLength: edgeCount * 8 + 12,
      writeByteLength: edgeCount * 4 + 4
    },
    resources: bindings.map(binding => ({
      buffer: binding.view as GraphDataView<'uint32'>,
      usage: binding.access === 'read' ? ('storage-read' as const) : ('storage-read-write' as const)
    })),
    compile: ({device}) => {
      const computation = new Computation(device, {
        id: props.id,
        source,
        shaderLayout: {
          bindings: bindings.map((binding, location) => ({
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
          for (const binding of bindings) {
            resolved[binding.name] = getViewBinding(
              binding.view as GraphDataView<'uint32'>,
              getBuffer
            );
          }
          computation.setBindings(resolved);
          computation.dispatch(computePass, 1);
        },
        destroy: () => computation.destroy()
      };
    }
  });
}
