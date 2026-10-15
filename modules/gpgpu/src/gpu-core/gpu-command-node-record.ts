// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GraphBufferHandle, GraphDataView} from './gpu-command-graph';
import type {
  GPUComputationDatumBinding,
  GPUComputationLoopCoordinate,
  GPUComputationNodeAnnotation
} from '../computation-record/computation-record-types';
import {hashGPUComputationText} from '../computation-record/hash-computation-text';
import {getViewElementOffset} from './graph-data-view-utils';

/** Kernel facts a gpu-core contributor records for one compute node (recording graphs only). */
export type GPUComputeKernelRecord = {
  /** Exported operator name, also the pack's formula key. */
  op: string;
  /** Stable kernel module identifier. */
  moduleId: string;
  /** Full WGSL text exactly as passed to Kernel. */
  source: string;
  /** Entry point; defaults to main. */
  entryPoint?: string;
  /** Pipeline override constants. */
  constants?: Readonly<Record<string, number>>;
  /** Lanes per workgroup. */
  workgroupSize: readonly [number, number, number];
  /** Dispatched workgroups along each dimension. */
  workgroups: readonly [number, number, number];
  /** Logical invocations, excluding idle lanes. */
  invocationCount?: number;
  /** Named storage or uniform bindings. */
  bindings?: readonly {
    name: string;
    view: GraphDataView | GraphBufferHandle;
    access: 'read' | 'read_write' | 'write' | 'uniform';
  }[];
  /** Key space indexed by invocations. */
  datum?: GPUComputationDatumBinding;
  /** Coordinate of an unrolled pass. */
  loop?: GPUComputationLoopCoordinate;
  /** Shared name of unrolled copies. */
  dedupName?: string;
  /** Formula reference id. */
  formula?: string;
  /** Display label. */
  label?: string;
};

/**
 * Returns an annotation only for recording graphs. Calls `describe` only when recording, so
 * production graphs pay no hashing or annotation string construction.
 */
export function getGPUComputeKernelAnnotation(
  graph: {readonly recording?: unknown},
  describe: () => GPUComputeKernelRecord
): GPUComputationNodeAnnotation | undefined {
  if (!graph.recording) return undefined;
  const {
    source,
    moduleId,
    entryPoint,
    constants,
    workgroupSize,
    workgroups,
    invocationCount,
    bindings,
    ...annotation
  } = describe();
  const hash = hashGPUComputationText(source);
  return {
    ...Object.fromEntries(Object.entries(annotation).filter(([, value]) => value !== undefined)),
    source: {
      id: `wgsl:${hash}`,
      language: 'wgsl',
      moduleId,
      hash,
      entryPoint: entryPoint ?? 'main',
      text: source,
      ...(constants && {constants})
    },
    workgroupSize,
    dispatch: {
      workgroups,
      ...(invocationCount !== undefined && {invocationCount}),
      indirect: false
    },
    ...(bindings && {
      bindings: bindings.map(({name, view, access}) => ({
        name,
        resource: 'buffer' in view ? view.buffer.id : view.id,
        access,
        ...('buffer' in view && {
          elementType:
            view.format === 'uint32'
              ? 'u32'
              : view.format === 'sint32'
                ? 'i32'
                : view.format === 'float32'
                  ? 'f32'
                  : view.format,
          elementOffset: getViewElementOffset(view),
          length: view.length
        })
      }))
    })
  };
}
