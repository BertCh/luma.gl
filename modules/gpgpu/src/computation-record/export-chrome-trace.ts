// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUComputationRecord, GPUComputationMetrics} from './computation-record-types';

/** Chrome Trace Event JSON event; timestamps and durations are in microseconds. */
export type ChromeTraceEvent = {
  name: string;
  ph: 'X' | 'C' | 'M' | 's' | 'f';
  ts: number;
  dur?: number;
  pid: number;
  tid: number;
  id?: number | string;
  cat?: string;
  bp?: 'e';
  args?: Record<string, unknown>;
};

/**
 * Exports measured nodes and RAW flows for Perfetto. Counter indices use the earliest begin
 * of a matching loop iteration when the axis is `iteration` or the loop id; otherwise ts is 0.
 * For multiple measurements, metadata.adapter and metadata.encoding are parallel arrays in
 * input order. Each measurement retains its own relative time origin.
 */
export function toChromeTraceEvents(
  record: GPUComputationRecord,
  metrics: readonly GPUComputationMetrics[],
  options: {flows?: boolean} = {}
): {
  traceEvents: ChromeTraceEvent[];
  displayTimeUnit: 'ns';
  metadata: Record<string, unknown>;
} {
  const traceEvents: ChromeTraceEvent[] = [
    {name: 'process_name', ph: 'M', ts: 0, pid: 1, tid: 0, args: {name: record.id}}
  ];
  const depth = (namespace: string): number => namespace.split('/').filter(Boolean).length;
  let maximumDepth = 0;
  for (const measurement of metrics) {
    const graph = record.graphs.find(candidate => candidate.id === measurement.graph);
    for (const node of graph?.nodes ?? [])
      maximumDepth = Math.max(maximumDepth, depth(node.namespace));
  }
  for (let level = 0; level <= maximumDepth; level++) {
    traceEvents.push({
      name: 'thread_name',
      ph: 'M',
      ts: 0,
      pid: 1,
      tid: level,
      args: {name: `level ${level}`}
    });
  }
  let flowId = 0;
  for (const measurement of metrics) {
    const graph = record.graphs.find(candidate => candidate.id === measurement.graph);
    if (!graph) continue;
    const scale = measurement.timebase === 'ns' ? 1 / 1000 : 1000;
    const timed = new Map<string, ChromeTraceEvent>();
    for (const node of graph.nodes) {
      const timing = measurement.nodes[node.id];
      if (timing?.gpuBegin === undefined || timing.gpuEnd === undefined) continue;
      const event: ChromeTraceEvent = {
        name: node.label,
        ph: 'X',
        ts: timing.gpuBegin * scale,
        dur: (timing.gpuEnd - timing.gpuBegin) * scale,
        pid: 1,
        tid: depth(node.namespace),
        cat: node.kind,
        args: {
          nodeId: node.id,
          graph: graph.id,
          namespace: node.namespace,
          ...(node.op !== undefined && {op: node.op}),
          ...(timing.passIndex !== undefined && {passIndex: timing.passIndex})
        }
      };
      traceEvents.push(event);
      timed.set(node.id, event);
    }
    for (const series of measurement.series ?? []) {
      series.values.forEach((value, iteration) => {
        let begin = Infinity;
        for (const node of graph.nodes) {
          if (
            node.loop?.iteration === iteration &&
            (series.axis === 'iteration' || series.axis === node.loop.id)
          ) {
            const timestamp = measurement.nodes[node.id]?.gpuBegin;
            if (timestamp !== undefined) begin = Math.min(begin, timestamp * scale);
          }
        }
        traceEvents.push({
          name: series.name,
          ph: 'C',
          ts: Number.isFinite(begin) ? begin : 0,
          pid: 1,
          tid: 0,
          args: {[series.name]: value}
        });
      });
    }
    if (options.flows !== false) {
      for (const edge of graph.edges) {
        const from = timed.get(edge.from);
        const to = timed.get(edge.to);
        if (edge.kind !== 'raw' || !from || !to) continue;
        const id = flowId++;
        traceEvents.push({
          name: 'raw',
          ph: 's',
          ts: from.ts + (from.dur ?? 0),
          pid: 1,
          tid: from.tid,
          id,
          cat: 'raw'
        });
        traceEvents.push({
          name: 'raw',
          ph: 'f',
          ts: to.ts,
          pid: 1,
          tid: to.tid,
          id,
          cat: 'raw',
          bp: 'e'
        });
      }
    }
  }
  return {
    traceEvents,
    displayTimeUnit: 'ns',
    metadata: {
      schema: record.schema,
      runId: record.runId,
      adapter:
        metrics.length > 1
          ? metrics.map(measurement => measurement.adapter ?? record.adapter)
          : (metrics[0]?.adapter ?? record.adapter),
      encoding:
        metrics.length === 1
          ? metrics[0].encoding
          : metrics.map(measurement => measurement.encoding)
    }
  };
}
