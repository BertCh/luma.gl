// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraphTimingReport} from '@luma.gl/gpgpu/gpu-core';

/**
 * Per-stage GPU timing for the Gaussian splat command graph.
 *
 * A single end-to-end number cannot tell you what to fix. Published measurements of WebGPU splat
 * renderers disagree about where the frame goes - preprocess-bound on a weak GPU, raster-bound on a
 * strong one, and essentially never sort-bound - and which of those a given scene and device land
 * in decides whether the next thing worth doing is a better sort, a cheaper projection, or less
 * overdraw. Splitting the graph's node timings into the stages a renderer actually reasons about is
 * what makes that decidable rather than arguable.
 *
 * @remarks
 * GPU durations require the `timestamp-query` feature; without it every stage reports only its CPU
 * encoding cost, and {@link GPUSplatStageTimings.hasGPUTimings} is `false`.
 */

/** A stage of the Gaussian splat command graph. */
export type GPUSplatStage =
  /** Clearing sort keys, the identity permutation and the indirect draw count. */
  | 'initialize'
  /** Projecting, culling and publishing sort keys for every borrowed source batch. */
  | 'projection'
  /** Higher-order spherical harmonics and semantic visibility. */
  | 'features'
  /** The global back-to-front radix sort, including its histogram and scan passes. */
  | 'sort'
  /** Gathering sorted records into a vertex stream, on the compatibility path only. */
  | 'gather'
  /** Rasterizing the visible rows. */
  | 'raster';

/** CPU and optional GPU duration for one stage. */
export type GPUSplatStageTiming = {
  /** Graph nodes attributed to this stage. */
  nodeCount: number;
  /** CPU time spent recording this stage's nodes. */
  cpuEncodeTimeMilliseconds: number;
  /** GPU execution time, when timestamp queries are available. */
  gpuTimeMilliseconds?: number;
};

/** Stage-resolved timings for one encoded Gaussian splat graph. */
export type GPUSplatStageTimings = {
  /** Whether any stage carries a GPU duration. */
  hasGPUTimings: boolean;
  /** CPU time spent encoding the whole graph. */
  cpuEncodeTimeMilliseconds: number;
  /** Sum of available per-node GPU durations. */
  gpuTimeMilliseconds?: number;
  /** One entry per stage that contributed at least one node. */
  stages: Partial<Record<GPUSplatStage, GPUSplatStageTiming>>;
};

/**
 * Node-id prefixes and the stage each one marks.
 *
 * No prefix is a prefix of another, so every node id matches at most one entry and the order of this
 * list does not matter. Keep it that way when adding a stage; the tests check it.
 *
 * @internal Exported for tests only.
 */
export const GPU_SPLAT_STAGE_PREFIXES: readonly (readonly [string, GPUSplatStage])[] = [
  ['gaussian-splat-initialize', 'initialize'],
  ['gaussian-splat-project-batch', 'projection'],
  ['gaussian-splat-features-batch', 'features'],
  ['gaussian-splat-gather-sorted-records', 'gather'],
  ['gaussian-splat-global-depth-sort', 'sort'],
  ['gaussian-splat-indirect-render', 'raster']
];

/** Classifies one graph node into the stage it belongs to. */
export function getGPUSplatStage(nodeId: string): GPUSplatStage | undefined {
  for (const [prefix, stage] of GPU_SPLAT_STAGE_PREFIXES) {
    if (nodeId.startsWith(prefix)) {
      return stage;
    }
  }
  return undefined;
}

/** Groups a graph timing report into the stages a splat renderer reasons about. */
export function getGPUSplatStageTimings(report: GPUCommandGraphTimingReport): GPUSplatStageTimings {
  const stages: Partial<Record<GPUSplatStage, GPUSplatStageTiming>> = {};
  let hasGPUTimings = false;

  for (const node of report.nodes) {
    const stage = getGPUSplatStage(node.id);
    if (!stage) {
      continue;
    }
    const existing = stages[stage] ?? {nodeCount: 0, cpuEncodeTimeMilliseconds: 0};
    existing.nodeCount++;
    existing.cpuEncodeTimeMilliseconds += node.cpuEncodeTimeMilliseconds;
    if (node.gpuTimeMilliseconds !== undefined) {
      hasGPUTimings = true;
      existing.gpuTimeMilliseconds = (existing.gpuTimeMilliseconds ?? 0) + node.gpuTimeMilliseconds;
    }
    stages[stage] = existing;
  }

  return {
    hasGPUTimings,
    cpuEncodeTimeMilliseconds: report.cpuEncodeTimeMilliseconds,
    ...(report.gpuTimeMilliseconds !== undefined
      ? {gpuTimeMilliseconds: report.gpuTimeMilliseconds}
      : {}),
    stages
  };
}
