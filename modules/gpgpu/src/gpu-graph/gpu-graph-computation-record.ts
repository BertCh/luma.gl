// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
// SPDX-FileComment: Independently implemented for WebGPU; inspired by NVIDIA RAPIDS cuGraph.

import type {
  GPUComputationDatumBinding,
  GPUComputationNodeAnnotation
} from '../computation-record/computation-record-types';
import {hashGPUComputationText} from '../computation-record/hash-computation-text';
import type {GPUCommandGraph, GraphBufferUse, GraphDataView} from '../gpu-core/gpu-command-graph';
import {getViewElementOffset} from '../gpu-core/graph-data-view-utils';

/** One storage binding of a gpu-graph pass, in `@binding` order. @internal */
export type GPUGraphRecordedBinding = {
  name: string;
  view: GraphDataView;
  usage: GraphBufferUse['usage'];
};

/** Records one gpu-graph compute pass only when the command graph records. @internal */
export function getGPUGraphPassAnnotation<Parameters>(
  commandGraph: GPUCommandGraph<Parameters>,
  props: {
    operation: string;
    source: string;
    bindings: readonly GPUGraphRecordedBinding[];
    workgroupSize: number;
    dispatch: {x: number; y: number; z: number};
    invocationCount?: number;
    datum?: GPUComputationDatumBinding;
    label?: string;
  }
): GPUComputationNodeAnnotation | undefined {
  if (!commandGraph.recording) return undefined;
  const hash = hashGPUComputationText(props.source);
  return {
    op: props.operation,
    ...(props.label !== undefined && {label: props.label}),
    ...(props.datum && {datum: props.datum}),
    source: {
      id: `wgsl:${hash}`,
      language: 'wgsl',
      moduleId: props.operation,
      hash,
      entryPoint: 'main',
      text: props.source
    },
    workgroupSize: [props.workgroupSize, 1, 1],
    dispatch: {
      workgroups: [props.dispatch.x, props.dispatch.y, props.dispatch.z],
      ...(props.invocationCount !== undefined && {invocationCount: props.invocationCount}),
      indirect: false
    },
    bindings: props.bindings.map(({name, view, usage}) => ({
      name,
      resource: view.buffer.id,
      access: usage === 'storage-read' ? 'read' : 'read_write',
      elementType: view.format.startsWith('float32')
        ? 'f32'
        : view.format.startsWith('sint32')
          ? 'i32'
          : 'u32',
      elementOffset: getViewElementOffset(view),
      length: view.length
    }))
  };
}

/** Scopes the recorded work of one contributor instance. @internal */
export function withGPUGraphRecordScope<Parameters, Result>(
  commandGraph: GPUCommandGraph<Parameters>,
  operation: string,
  id: string,
  callback: () => Result
): Result {
  return commandGraph.withRecordScope({namespace: id, op: operation}, callback);
}

/** Adds structured coordinates to one unrolled iteration's recorded nodes. @internal */
export function withGPUGraphIterationScope<Parameters, Result>(
  commandGraph: GPUCommandGraph<Parameters>,
  props: {loopId: string; iteration: number; maximumIterations: number; gated: boolean},
  callback: () => Result
): Result {
  return commandGraph.withRecordScope(
    {
      loop: {
        id: props.loopId,
        iteration: props.iteration,
        phase: props.gated ? 'gated' : 'unconditional',
        maximumIterations: props.maximumIterations
      }
    },
    callback
  );
}
