// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import type {DeckHost} from '../engine/deck-host';
import {
  formatCompiledGraphTiming,
  hasGPUTimestamps,
  measureCompiledGraph
} from '../engine/vector-timing';
import {h} from './dom';

/** GPU time of every compiled graph of the active scene, run once. */
export type GpuMeasurement = {
  /** Sum over graphs, milliseconds per run. */
  milliseconds: number;
  /** One-line description with the measuring method. */
  summary: string;
};

/**
 * Measures the active scene's compiled graphs (five runs each, GPU timestamps when available).
 * Resolves `null` when there is no device or scene, or the measurement was interrupted.
 */
export async function measureActiveScene(host: DeckHost): Promise<GpuMeasurement | null> {
  const device = host.gpuDevice;
  const instance = host.activeInstance;
  if (!device || !instance) return null;
  const fence = device.createBuffer({
    byteLength: 4,
    usage: Buffer.COPY_SRC | Buffer.COPY_DST | Buffer.STORAGE
  });
  try {
    let total = 0;
    let cpu = 0;
    let method = 'wall-clock' as 'wall-clock' | 'gpu-timestamps';
    for (const graph of instance.getCompiledGraphs()) {
      const timing = await measureCompiledGraph(device, graph as never, {
        parameters: undefined as never,
        completionBuffer: fence,
        runs: 5
      });
      total += timing.milliseconds;
      cpu += timing.cpuEncodeMilliseconds;
      method = timing.method;
    }
    const summary = formatCompiledGraphTiming({
      milliseconds: total,
      cpuEncodeMilliseconds: cpu,
      method
    });
    return {milliseconds: total, summary: `All graphs once: ${summary}`};
  } catch {
    return null;
  } finally {
    fence.destroy();
  }
}

/** The "Under the hood" group: engine statistics, the measure button, scene hood readouts. */
export type UnderTheHood = {
  /** The collapsible group. */
  element: HTMLElement;
  /** Refreshes the statistics (called every 500 ms by the view). */
  update: () => void;
  /** Shows a finished measurement. */
  showMeasurement: (measurement: GpuMeasurement | null) => void;
  /** Disables the button while a measurement runs. */
  setMeasuring: (measuring: boolean) => void;
};

const STAT_LABELS = [
  ['graphs', 'Compiled graphs'],
  ['nodes', 'Graph nodes'],
  ['rebuilds', 'Rebuilds'],
  ['frames', 'Frames encoded'],
  ['fps', 'Frame rate'],
  ['encode', 'CPU encode'],
  ['gpu', 'GPU time']
] as const;

/**
 * Builds the "Under the hood" group. `wrap` turns the content into the collapsible card (the view
 * owns that style); `hoodReadouts` is the list of `hood: true` readouts, when the scene has any.
 */
export function createUnderTheHood(options: {
  getHost: () => DeckHost;
  onMeasure: () => void;
  hoodReadouts: HTMLElement | null;
  wrap: (content: HTMLElement) => HTMLElement;
}): UnderTheHood {
  const {getHost, onMeasure, hoodReadouts, wrap} = options;
  const lines = h('dl', {class: 'hood-grid'});
  const cells = new Map<string, HTMLElement>();
  for (const [key, label] of STAT_LABELS) {
    const value = h('dd', {}, '—');
    cells.set(key, value);
    lines.append(h('div', {}, h('dt', {}, label), value));
  }
  const measureButton = h('button', {class: 'btn btn-small', type: 'button'}, 'Measure GPU time');
  measureButton.addEventListener('click', onMeasure);
  const measureResult = h('div', {class: 'muted small'});
  const note = h(
    'p',
    {class: 'muted small'},
    'Compile once, rewrite parameters each frame: the rebuild counter only moves when a compile-time option changes. Compute and drawing share one command buffer per frame.'
  );
  const element = wrap(
    h(
      'div',
      {},
      lines,
      h('div', {class: 'hood-actions'}, measureButton, measureResult),
      hoodReadouts,
      note
    )
  );
  const showHint = () => {
    const device = getHost().gpuDevice;
    if (device && !cells.get('gpu')?.dataset['measured']) {
      measureResult.textContent = hasGPUTimestamps(device)
        ? 'GPU timestamp queries available.'
        : 'timestamp-query unavailable: timings use wall clock.';
    }
  };
  return {
    element,
    update: () => {
      const host = getHost();
      const {graphCount, nodeCount, rebuildCount} = host.pollRebuilds();
      const {frameCount, framesPerSecond, encodeMilliseconds} = host.stats;
      cells.get('graphs')!.textContent = String(graphCount);
      cells.get('nodes')!.textContent = String(nodeCount);
      cells.get('rebuilds')!.textContent = String(rebuildCount);
      cells.get('frames')!.textContent = String(frameCount);
      cells.get('fps')!.textContent = `${framesPerSecond.toFixed(0)} fps`;
      cells.get('encode')!.textContent = `${encodeMilliseconds.toFixed(2)} ms`;
      if (!measureButton.disabled) showHint();
    },
    showMeasurement: measurement => {
      const gpu = cells.get('gpu')!;
      if (measurement) {
        gpu.textContent = `${measurement.milliseconds.toFixed(2)} ms`;
        gpu.dataset['measured'] = 'true';
        measureResult.textContent = measurement.summary;
      } else {
        measureResult.textContent = 'Measurement was interrupted.';
      }
    },
    setMeasuring: measuring => {
      measureButton.disabled = measuring;
      if (measuring) measureResult.textContent = 'Measuring…';
    }
  };
}
