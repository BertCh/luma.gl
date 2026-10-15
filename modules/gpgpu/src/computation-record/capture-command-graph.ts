// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GraphDataView,
  type CompiledGPUCommandGraph,
  type GPUCommandGraphEncodeOptions,
  type GPUCommandGraphExecutionBudget,
  type GraphBufferHandle
} from '@luma.gl/gpgpu/gpu-core';
import type {CommandEncoder} from '@luma.gl/core';
import type {
  GPUComputationCapture,
  GPUComputationCaptureElementType
} from './computation-record-types';
import {GPUSnapshotArena, type GPUSnapshotLabel} from './gpu-snapshot-arena';

/** Options for an explicit observation run of a compiled command graph. */
export type CaptureGPUCommandGraphProps<Parameters> = {
  /** Parameters and imported resources for every step. */
  encodeOptions: GPUCommandGraphEncodeOptions<Parameters>;
  /** Named packed views or whole buffers to copy. */
  views: Readonly<Record<string, GraphDataView | GraphBufferHandle>>;
  /** Defaults to the view's 32-bit scalar type, otherwise `u32`. */
  elementTypes?: Readonly<Record<string, GPUComputationCaptureElementType>>;
  /** Snapshot boundary; defaults to `publication`. */
  at?: 'publication' | 'step' | 'end';
  /** Defaults to one node per step for `step`, otherwise unbounded. */
  budget?: GPUCommandGraphExecutionBudget;
  /** Arena capacity; defaults to the planned step count, plus one for `end`. */
  maximumSnapshots?: number;
  /** Run identity for the enclosing capture sidecar. */
  runId?: string;
  /** Graph identity used in the record; defaults to `<compiled.id>-compiled`, like describe(). */
  graphId?: string;
};

/** Readback values and execution counts for one observation run. */
export type GPUCommandGraphCapture = {
  /** One capture per named view per snapshot frame. */
  captures: GPUComputationCapture[];
  /** Number of encoded execution-plan steps. */
  stepCount: number;
  /** Number of snapshot frames, each containing every requested view. */
  snapshotCount: number;
};

/**
 * Executes and captures a graph through progressive publication points, then maps one arena.
 * Stepping splits queue submissions and limits pass coalescing: describe this observer effect
 * with the record's GPUComputationEncodingFlags. List observed transients in `recording.retain`;
 * getPhysicalBuffer rejects aliased transients that were not retained.
 */
export async function captureGPUCommandGraph<Parameters>(
  compiled: CompiledGPUCommandGraph<Parameters>,
  props: CaptureGPUCommandGraphProps<Parameters>
): Promise<GPUCommandGraphCapture> {
  const at = props.at ?? 'publication';
  const execution = compiled.createExecution(
    props.budget ?? {
      maximumInvocationCount: Number.MAX_SAFE_INTEGER,
      ...(at === 'step' && {maximumNodeCount: 1})
    },
    {publicationPolicy: 'progressive'}
  );
  const entries = Object.entries(props.views);
  const slots = entries.map(([name, view]) => {
    if (view instanceof GraphDataView && view.byteStride !== view.rowByteLength) {
      throw new Error(`Capture view "${name}" must be packed`);
    }
    const format = view instanceof GraphDataView ? view.format : '';
    const elementType =
      props.elementTypes?.[name] ??
      (format.startsWith('float32') ? 'f32' : format.startsWith('sint32') ? 'i32' : 'u32');
    const components = /^(?:float32|sint32|uint32)x([234])$/.exec(format);
    return {
      name,
      byteLength:
        view instanceof GraphDataView ? view.length * view.rowByteLength : view.byteLength,
      elementType,
      ...(components && {components: Number(components[1])})
    };
  });
  // Resolve before submitting anything, so invalid observation targets fail without a partial run.
  const sources = Object.fromEntries(
    entries.map(([name, view]) => [
      name,
      {
        buffer: compiled.getPhysicalBuffer(view, props.encodeOptions.buffers),
        byteOffset: view instanceof GraphDataView ? view.byteOffset : 0
      }
    ])
  );
  const resources = Object.fromEntries(
    entries.map(([name, view]) => [name, view instanceof GraphDataView ? view.buffer.id : view.id])
  );
  const arena = new GPUSnapshotArena(compiled.device, {
    id: `${compiled.id}-snapshots`,
    slots,
    capacity: props.maximumSnapshots ?? execution.plan.stepCount + (at === 'end' ? 1 : 0)
  });
  let stepCount = 0;
  let lastLabel: GPUSnapshotLabel = {};
  const submit = (encode: (commandEncoder: CommandEncoder) => void) => {
    const commandEncoder = compiled.device.createCommandEncoder();
    try {
      encode(commandEncoder);
      compiled.device.submit(commandEncoder.finish());
    } finally {
      commandEncoder.destroy();
    }
  };
  try {
    const requiredSnapshots =
      at === 'end'
        ? 1
        : execution.plan.steps.filter(step => at === 'step' || step.publications.length > 0).length;
    if (arena.capacity < requiredSnapshots) {
      throw new Error(`${arena.id} capacity cannot hold the planned snapshots`);
    }
    while (!execution.completed) {
      submit(commandEncoder => {
        const step = execution.encodeNext(commandEncoder, props.encodeOptions);
        stepCount++;
        lastLabel = {
          afterNode: compiled.stats.nodeOrder[step.nextNodeIndex - 1],
          ...(step.publications.length > 0 && {
            label: step.publications.map(publication => publication.id).join(', ')
          })
        };
        if (at === 'step' || (at === 'publication' && step.publications.length > 0)) {
          arena.encodeSnapshot(commandEncoder, sources, lastLabel);
        }
      });
    }
    if (at === 'end') {
      submit(commandEncoder => arena.encodeSnapshot(commandEncoder, sources, lastLabel));
    }
    const captures = await arena.read({
      graph: props.graphId ?? `${compiled.id}-compiled`,
      resources,
      ...(props.runId !== undefined && {runId: props.runId})
    });
    return {captures, stepCount, snapshotCount: arena.count};
  } finally {
    arena.destroy();
  }
}
