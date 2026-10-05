// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * On-demand timing of a compiled command graph, used by the join, lasso, zonal and buffer modes to
 * show what a compile-time option (`spatialSort`, `spatialIndex`) buys.
 *
 * Measurement never touches Deck's frame encoder: each run encodes the graph into its own command
 * encoder and submits it outside the frame. When the device was created with the
 * `timestamp-query` feature, every compute pass gets a timestamp pair and the result is the median
 * summed GPU pass time. Otherwise the graph is encoded `repetitions` times into one encoder, and
 * the wall time from submit to a 4-byte readback is divided by `repetitions` (GPU time plus an
 * amortized share of one queue round trip, so it is an upper bound).
 */

import type {Buffer, Device} from '@luma.gl/core';
import type {CompiledGPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';

/** How {@link measureCompiledGraph} obtained its number. */
export type CompiledGraphTimingMethod = 'gpu-timestamps' | 'wall-clock';

/** Result of {@link measureCompiledGraph}. */
export type CompiledGraphTiming = {
  /** Median milliseconds per graph execution. */
  milliseconds: number;
  /** Median CPU milliseconds spent in `compiled.encode()`. */
  cpuEncodeMilliseconds: number;
  /** Measurement method; wall-clock includes an amortized queue round trip. */
  method: CompiledGraphTimingMethod;
};

/** Options for {@link measureCompiledGraph}. */
export type MeasureCompiledGraphOptions<Parameters> = {
  /** Encode parameters passed to `compiled.encode`. */
  parameters: Parameters;
  /** Any small buffer the graph writes; read back to wait for completion in wall-clock mode. */
  completionBuffer: Buffer;
  /** Timed runs (median reported). Defaults to 7. */
  runs?: number;
  /** Untimed warm-up runs. Defaults to 2. */
  warmUpRuns?: number;
  /** Graph executions per wall-clock run. Defaults to 8. */
  repetitions?: number;
  /** Stops early (rejects) when aborted. */
  signal?: AbortSignal;
};

/** Largest timestamp query set WebGPU allows. */
const MAXIMUM_QUERY_COUNT = 4096;

/** Returns true when GPU timestamp queries are available on `device`. */
export function hasGPUTimestamps(device: Device): boolean {
  return device.features.has('timestamp-query');
}

/**
 * Measures one compiled graph outside Deck's frame. Results the graph writes are identical to a
 * normal frame encoding, so running it between frames is invisible.
 */
export async function measureCompiledGraph<Parameters>(
  device: Device,
  compiled: CompiledGPUCommandGraph<Parameters>,
  options: MeasureCompiledGraphOptions<Parameters>
): Promise<CompiledGraphTiming> {
  const {parameters, completionBuffer, runs = 7, warmUpRuns = 2, repetitions = 8, signal} = options;
  const useTimestamps = hasGPUTimestamps(device);
  const timings: number[] = [];
  const cpuTimings: number[] = [];
  for (let run = 0; run < warmUpRuns + runs; run++) {
    signal?.throwIfAborted();
    const timing = useTimestamps
      ? await measureWithTimestamps(device, compiled, parameters)
      : await measureWithWallClock(device, compiled, parameters, completionBuffer, repetitions);
    if (run >= warmUpRuns) {
      timings.push(timing.milliseconds);
      cpuTimings.push(timing.cpuEncodeMilliseconds);
    }
  }
  return {
    milliseconds: getMedian(timings),
    cpuEncodeMilliseconds: getMedian(cpuTimings),
    method: useTimestamps ? 'gpu-timestamps' : 'wall-clock'
  };
}

/** Formats a timing for a panel readout, for example `"0.42 ms GPU"`. */
export function formatCompiledGraphTiming(timing: CompiledGraphTiming | null): string {
  if (!timing) return '...';
  const label = timing.method === 'gpu-timestamps' ? 'GPU' : 'GPU+sync (wall)';
  return `${timing.milliseconds.toFixed(2)} ms ${label} · ${timing.cpuEncodeMilliseconds.toFixed(2)} ms CPU encode`;
}

/** Formats `baseline / optimized` as a speedup such as `"3.4x faster"`. */
export function formatSpeedup(baselineMilliseconds: number, optimizedMilliseconds: number): string {
  if (!(baselineMilliseconds > 0) || !(optimizedMilliseconds > 0)) return 'n/a';
  const ratio = baselineMilliseconds / optimizedMilliseconds;
  return ratio >= 1 ? `${ratio.toFixed(2)}x faster` : `${(1 / ratio).toFixed(2)}x slower`;
}

async function measureWithTimestamps<Parameters>(
  device: Device,
  compiled: CompiledGPUCommandGraph<Parameters>,
  parameters: Parameters
): Promise<{milliseconds: number; cpuEncodeMilliseconds: number}> {
  const queryCount = Math.min(MAXIMUM_QUERY_COUNT, compiled.stats.nodeOrder.length * 2 + 2);
  const querySet = device.createQuerySet({
    id: `${compiled.id}-timing`,
    type: 'timestamp',
    count: queryCount
  });
  try {
    const commandEncoder = device.createCommandEncoder({
      id: `${compiled.id}-timing-encoder`,
      timeProfilingQuerySet: querySet
    });
    const encoding = compiled.encode(commandEncoder, {parameters});
    device.submit(commandEncoder.finish());
    const report = await encoding.readTimings();
    return {
      milliseconds: report.gpuTimeMilliseconds ?? 0,
      cpuEncodeMilliseconds: encoding.stats.cpuEncodeTimeMilliseconds
    };
  } finally {
    querySet.destroy();
  }
}

async function measureWithWallClock<Parameters>(
  device: Device,
  compiled: CompiledGPUCommandGraph<Parameters>,
  parameters: Parameters,
  completionBuffer: Buffer,
  repetitions: number
): Promise<{milliseconds: number; cpuEncodeMilliseconds: number}> {
  // Drain earlier work so the timed window covers only this submission.
  await completionBuffer.readAsync(0, 4);
  const commandEncoder = device.createCommandEncoder({id: `${compiled.id}-timing-encoder`});
  let cpuEncodeMilliseconds = 0;
  for (let repetition = 0; repetition < repetitions; repetition++) {
    cpuEncodeMilliseconds += compiled.encode(commandEncoder, {parameters}).stats
      .cpuEncodeTimeMilliseconds;
  }
  const start = performance.now();
  device.submit(commandEncoder.finish());
  await completionBuffer.readAsync(0, 4);
  return {
    milliseconds: (performance.now() - start) / repetitions,
    cpuEncodeMilliseconds: cpuEncodeMilliseconds / repetitions
  };
}

function getMedian(values: number[]): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((left, right) => left - right);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}
