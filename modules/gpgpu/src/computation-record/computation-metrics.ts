// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPU_COMPUTATION_METRICS_SCHEMA,
  GPU_COMPUTATION_CAPTURES_SCHEMA
} from './computation-record-types';
import type {
  GPUComputationMetrics,
  GPUComputationCaptures,
  GPUComputationEncodingFlags,
  GPUComputationAdapter,
  GPUComputationSeries,
  GPUComputationNodeMetrics
} from './computation-record-types';

/** Structural subset of GPUCommandGraphTimingReport; no gpu-core dependency. */
export type GPUComputationTimingInput = {
  nodes: readonly {
    id: string;
    cpuEncodeTimeMilliseconds: number;
    gpuBeginNanoseconds?: number;
    gpuEndNanoseconds?: number;
    passIndex?: number;
    condition?: {outcome?: string};
  }[];
};

/** Builds nanosecond metrics relative to the earliest GPU begin timestamp in the report. */
export function createGPUComputationMetrics(props: {
  runId: string;
  graph: string;
  timing: GPUComputationTimingInput;
  encoding: GPUComputationEncodingFlags;
  adapter?: GPUComputationAdapter;
  series?: readonly GPUComputationSeries[];
  counters?: Readonly<Record<string, Readonly<Record<string, number>>>>;
}): GPUComputationMetrics {
  let origin = Infinity;
  for (const node of props.timing.nodes) {
    if (node.gpuBeginNanoseconds !== undefined) origin = Math.min(origin, node.gpuBeginNanoseconds);
  }
  const nodes: Record<string, GPUComputationNodeMetrics> = Object.fromEntries(
    props.timing.nodes.map(node => {
      const outcome = node.condition?.outcome;
      return [
        node.id,
        {
          cpuEncodeMs: node.cpuEncodeTimeMilliseconds,
          ...(node.gpuBeginNanoseconds !== undefined && {
            gpuBegin: node.gpuBeginNanoseconds - origin
          }),
          ...(node.gpuBeginNanoseconds !== undefined &&
            node.gpuEndNanoseconds !== undefined && {gpuEnd: node.gpuEndNanoseconds - origin}),
          ...(node.passIndex !== undefined && {passIndex: node.passIndex}),
          ...(node.condition !== undefined && {
            outcome:
              outcome === 'skipped'
                ? 'skipped'
                : outcome === 'gpu-resolved' || outcome === 'indirect'
                  ? 'indirect'
                  : 'executed'
          }),
          ...(props.counters?.[node.id] !== undefined && {counters: props.counters[node.id]})
        }
      ];
    })
  );
  return {
    schema: GPU_COMPUTATION_METRICS_SCHEMA,
    runId: props.runId,
    graph: props.graph,
    timebase: 'ns',
    encoding: props.encoding,
    nodes,
    ...(props.adapter !== undefined && {adapter: props.adapter}),
    ...(props.series !== undefined && {series: props.series})
  };
}

/** Builds a captures sidecar with empty arrays filled in. */
export function createGPUComputationCaptures(props: {
  runId: string;
  snapshots?: GPUComputationCaptures['snapshots'];
  stamps?: GPUComputationCaptures['stamps'];
  replays?: GPUComputationCaptures['replays'];
}): GPUComputationCaptures {
  return {
    schema: GPU_COMPUTATION_CAPTURES_SCHEMA,
    runId: props.runId,
    snapshots: props.snapshots ?? [],
    stamps: props.stamps ?? [],
    replays: props.replays ?? []
  };
}
