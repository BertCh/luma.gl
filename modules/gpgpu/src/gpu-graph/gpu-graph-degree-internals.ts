// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
// SPDX-FileComment: Independently implemented for WebGPU; inspired by NVIDIA RAPIDS cuGraph.

import {type Binding} from '@luma.gl/core';
import {Computation} from '@luma.gl/engine';
import type {GPUCommandGraph, GraphDataView} from '../gpu-core/gpu-command-graph';
import {
  type GPUBoundedDispatchLayout,
  getBoundedDispatchLayout,
  getBoundedInvocationIndexSource
} from '../gpu-core/gpu-dispatch-utils';
import {getViewBinding, getViewElementOffset} from '../gpu-core/graph-data-view-utils';
import type {GPUGraphDegree} from './gpu-graph-degree';
import {
  importGPUGraphColumn,
  usesGPUGraphViews,
  validateDistinctGPUGraphHandles,
  type GPUGraphColumn,
  type GPUGraphTopologyLike
} from './gpu-graph-topology-view';

const GRAPH_DEGREE_WORKGROUP_SIZE = 256;

/** Adds one exact CSR degree pass using an explicit device dispatch limit. @internal */
export function addGPUGraphDegreeToGraphWithDispatchLimit<Parameters>(
  degree: GPUGraphDegree<GPUGraphTopologyLike, GPUGraphColumn<'uint32'>>,
  commandGraph: GPUCommandGraph<Parameters>,
  maxComputeWorkgroupsPerDimension: number
): void {
  if (degree.topology.graph.vertexCount === 0) {
    return;
  }

  const adjacency =
    degree.direction === 'incoming' && degree.topology.graph.directed
      ? degree.topology.reverse!
      : degree.topology.forward;
  const offsets = importGPUGraphColumn(commandGraph, `${degree.id}-offsets`, adjacency.offsets);
  const output = importGPUGraphColumn(commandGraph, `${degree.id}-output`, degree.output);
  if (usesGPUGraphViews(degree.topology, [degree.output])) {
    validateDistinctGPUGraphHandles(degree.id, [offsets], [{name: 'output', view: output}]);
  }
  const dispatchLayout = getGPUGraphDegreeDispatchLayout(
    degree.topology.graph.vertexCount,
    maxComputeWorkgroupsPerDimension
  );
  const source = /* wgsl */ `
const VERTEX_COUNT: u32 = ${degree.topology.graph.vertexCount}u;
const OFFSETS_OFFSET: u32 = ${getViewElementOffset(offsets)}u;
const OUTPUT_OFFSET: u32 = ${getViewElementOffset(output)}u;
@group(0) @binding(0) var<storage, read> offsets: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;

@compute @workgroup_size(${GRAPH_DEGREE_WORKGROUP_SIZE})
fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${getBoundedInvocationIndexSource(dispatchLayout, GRAPH_DEGREE_WORKGROUP_SIZE)}
  if (index >= VERTEX_COUNT) { return; }
  output[OUTPUT_OFFSET + index] =
    offsets[OFFSETS_OFFSET + index + 1u] - offsets[OFFSETS_OFFSET + index];
}`;

  addDegreePass(commandGraph, {id: degree.id, offsets, output, source, dispatchLayout});
}

/** Compiles one two-binding degree kernel without allocating graph-owned scratch. */
function addDegreePass<Parameters>(
  commandGraph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    offsets: GraphDataView<'uint32'>;
    output: GraphDataView<'uint32'>;
    source: string;
    dispatchLayout: GPUBoundedDispatchLayout;
  }
): void {
  commandGraph.addComputePass({
    id: props.id,
    resources: [
      {buffer: props.offsets, usage: 'storage-read'},
      {buffer: props.output, usage: 'storage-write'}
    ],
    compile: ({device}) => {
      const computation = new Computation(device, {
        id: props.id,
        source: props.source,
        shaderLayout: {
          bindings: [
            {name: 'offsets', type: 'storage', group: 0, location: 0},
            {name: 'output', type: 'storage', group: 0, location: 1}
          ]
        }
      });

      return {
        encode: ({computePass, getBuffer}) => {
          const bindings: Record<string, Binding> = {
            offsets: getViewBinding(props.offsets, getBuffer),
            output: getViewBinding(props.output, getBuffer)
          };
          computation.setBindings(bindings);
          computation.dispatch(
            computePass,
            props.dispatchLayout.x,
            props.dispatchLayout.y,
            props.dispatchLayout.z
          );
        },
        destroy: () => computation.destroy()
      };
    }
  });
}

/** Plans a bounded three-dimensional dispatch for exact vertex degree extraction. @internal */
export function getGPUGraphDegreeDispatchLayout(
  elementCount: number,
  maxComputeWorkgroupsPerDimension: number
): GPUBoundedDispatchLayout {
  return getBoundedDispatchLayout(
    'GPUGraphDegree',
    elementCount,
    GRAPH_DEGREE_WORKGROUP_SIZE,
    maxComputeWorkgroupsPerDimension
  );
}
