// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUMapGraphParameterBuffer, importGraphBuffer, submitGraph} from '../../../src/map-graphs';
import {
  getGPUTimeWindowParameterValues,
  getGPUTimeWindowWordParameterValues,
  getInt64TimeWords,
  GPUTimeWindowFilter,
  GPU_TIME_WORD_WINDOW_PARAMETER_LENGTH,
  splitTimeWords,
  type GPUTimeWordWindow
} from '../../../src/map-graphs/time-window-filter';
import {
  createInputBuffer,
  createOutputBuffer,
  createVectorView,
  readCompactIds,
  readFloat32
} from '../map-graph-test-utils';

const NOW = 1_700_000_000_000n;
const YEAR = 365n * 24n * 3600n * 1000n;

/** Exact instant as integer plus f32 fraction, from the same split the GPU window uses. */
function getInstant(value: number | bigint): {
  integer: bigint;
  fraction: number;
} {
  const {low, high, fraction} = splitTimeWords(value);
  return {
    integer: BigInt.asIntN(64, (BigInt(high) << 32n) | BigInt(low)),
    fraction
  };
}

type WordOracle = {mask: number[]; weights: number[]; clip: number[]};

/** BigInt CPU reference: exact acceptance, Number differences for fade and clip. */
function filterWordsOracle(
  starts: readonly bigint[],
  ends: readonly bigint[] | undefined,
  window: GPUTimeWordWindow
): WordOracle {
  const windowStart = getInstant(window.start);
  const windowEnd = getInstant(window.end);
  const startFade = window.startFadeDuration ?? 0;
  const endFade = window.endFadeDuration ?? 0;
  const result: WordOracle = {mask: [], weights: [], clip: []};
  const clamp01 = (value: number) => Math.min(Math.max(value, 0), 1);
  for (const [row, rowStart] of starts.entries()) {
    const rowEnd = ends?.[row] ?? rowStart;
    const endAfterStart = Number(rowEnd - windowStart.integer) - windowStart.fraction;
    const startBeforeEnd = Number(windowEnd.integer - rowStart) + windowEnd.fraction;
    const accepted = endAfterStart >= 0 && startBeforeEnd >= 0;
    result.mask.push(accepted ? 1 : 0);
    const startRamp = startFade > 0 ? clamp01(endAfterStart / startFade) : 1;
    const endRamp = endFade > 0 ? clamp01(startBeforeEnd / endFade) : 1;
    result.weights.push(accepted ? Math.min(startRamp, endRamp) : 0);
    const duration = Number(rowEnd - rowStart);
    const clipStart =
      duration > 0
        ? clamp01((Number(windowStart.integer - rowStart) + windowStart.fraction) / duration)
        : 0;
    const clipEnd = duration > 0 ? clamp01(startBeforeEnd / duration) : 1;
    result.clip.push(accepted ? clipStart : 0, accepted ? clipEnd : 0);
  }
  return result;
}

function createWordBuffer(device: Device, values: readonly bigint[]): Buffer {
  return createInputBuffer(device, getInt64TimeWords(BigInt64Array.from(values)));
}

/** Runs one compiled word filter over several windows and returns per-window GPU results. */
async function runWordFilter(
  device: Device,
  props: {
    starts: readonly bigint[];
    ends?: readonly bigint[];
    windows: readonly GPUTimeWordWindow[];
    withFade?: boolean;
    withClip?: boolean;
    chunkSize?: number;
  }
): Promise<{mask: number[]; ids: number[]; weights: number[]; clip: number[]}[]> {
  const rows = props.starts.length;
  const buffers: Buffer[] = [];
  const track = (buffer: Buffer): Buffer => {
    buffers.push(buffer);
    return buffer;
  };
  const graph = new GPUCommandGraph(device, {id: 'time-window-words'});
  const importWords = (id: string, values: readonly bigint[]) => {
    const chunkSize = props.chunkSize ?? values.length;
    const chunks = [];
    for (let first = 0; first < values.length; first += chunkSize) {
      const part = values.slice(first, first + chunkSize);
      chunks.push(
        importGraphBuffer(
          graph,
          `${id}-${first}`,
          track(createWordBuffer(device, part)),
          'uint32x2',
          part.length
        )
      );
    }
    return props.chunkSize ? createVectorView(id, 'uint32x2', chunks) : chunks[0];
  };
  /** Creates output buffers, one per input chunk, and imports them as a view or vector view. */
  const createOutputRows = <Format extends 'float32' | 'float32x2'>(
    id: string,
    format: Format,
    components: number
  ) => {
    const chunkSize = props.chunkSize ?? rows;
    const outputs: Buffer[] = [];
    const views = [];
    for (let first = 0; first < rows; first += chunkSize) {
      const length = Math.min(chunkSize, rows - first);
      const buffer = track(createOutputBuffer(device, length * components));
      outputs.push(buffer);
      views.push(importGraphBuffer(graph, `${id}-${first}`, buffer, format, length));
    }
    return {
      buffers: outputs,
      view: props.chunkSize ? createVectorView(id, format, views) : views[0]
    };
  };
  const readChunks = async (buffers: Buffer[], lengths: number) => {
    const values: number[] = [];
    for (const buffer of buffers) {
      values.push(...(await readFloat32(buffer, Math.min(lengths, buffer.byteLength / 4))));
    }
    return values;
  };
  const weightOutput = props.withFade ? createOutputRows('weights', 'float32', 1) : undefined;
  const clipOutput = props.withClip ? createOutputRows('clip', 'float32x2', 2) : undefined;
  const window = new GPUMapGraphParameterBuffer(device, {
    id: 'word-window',
    format: 'uint32',
    length: GPU_TIME_WORD_WINDOW_PARAMETER_LENGTH,
    values: getGPUTimeWindowWordParameterValues(props.windows[0])
  });
  const maskBuffer = track(createOutputBuffer(device, rows));
  const idsBuffer = track(createOutputBuffer(device, rows));
  const countBuffer = track(createOutputBuffer(device, 1));
  const overflowBuffer = track(createOutputBuffer(device, 1));
  const maskView = props.chunkSize
    ? undefined
    : importGraphBuffer(graph, 'mask', maskBuffer, 'uint32', rows);
  graph.add(
    new GPUTimeWindowFilter({
      timestamps: importWords('starts', props.starts),
      endTimestamps: props.ends ? importWords('ends', props.ends) : undefined,
      window: window.importToGraph(graph),
      outputMask: maskView,
      fadeWeights: weightOutput?.view,
      clipFractions: clipOutput?.view,
      output: {
        ids: importGraphBuffer(graph, 'ids', idsBuffer, 'uint32', rows),
        count: importGraphBuffer(graph, 'count', countBuffer, 'uint32', 1),
        overflow: importGraphBuffer(graph, 'overflow', overflowBuffer, 'uint32', 1)
      }
    })
  );
  const compiled = graph.compile();
  const results = [];
  for (const timeWindow of props.windows) {
    window.write(getGPUTimeWindowWordParameterValues(timeWindow));
    submitGraph(device, compiled, undefined);
    const ids = await readCompactIds(idsBuffer, countBuffer);
    const mask = new Array<number>(rows).fill(0);
    for (const id of ids) mask[id] = 1;
    results.push({
      mask,
      ids,
      weights: weightOutput ? await readChunks(weightOutput.buffers, rows) : [],
      clip: clipOutput ? await readChunks(clipOutput.buffers, rows * 2) : []
    });
  }
  compiled.destroy();
  window.destroy();
  for (const buffer of buffers) buffer.destroy();
  return results;
}

function expectClose(actual: number[], expected: number[], relative = 1e-5): void {
  expect(actual.length).toBe(expected.length);
  for (const [index, value] of expected.entries()) {
    expect(Math.abs(actual[index] - value)).toBeLessThanOrEqual(
      relative * Math.max(1, Math.abs(value))
    );
  }
}

it('GPUTimeWindowFilter word times are exact across a low-word wrap', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const starts: bigint[] = [];
  for (let value = 0x1_ffff_fff0n; value <= 0x2_0000_0010n; value++) starts.push(value);
  const windows: GPUTimeWordWindow[] = [
    {start: 0x1_ffff_ffffn, end: 0x2_0000_0001n},
    {start: 0x2_0000_0000n, end: 0x2_0000_0000n},
    {start: 0x1_ffff_fffen, end: 0x1_ffff_ffffn},
    {start: 0x2_0000_0001n, end: 0x2_0000_0000n},
    {start: 0x1_ffff_fff0n, end: 0x2_0000_0010n}
  ];
  const results = await runWordFilter(device, {starts, windows});
  for (const [index, timeWindow] of windows.entries()) {
    const oracle = filterWordsOracle(starts, undefined, timeWindow);
    expect(results[index].mask).toEqual(oracle.mask);
  }
  expect(results[0].mask.reduce((sum, value) => sum + value, 0)).toBe(3);
  expect(results[3].ids).toEqual([]);
});

it('GPUTimeWindowFilter word times resolve 1 ms edges where float32 collapses', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const edges = [NOW, NOW + YEAR, -NOW];
  const starts = edges.flatMap(edge => [edge - 1n, edge, edge + 1n]);
  const windows: GPUTimeWordWindow[] = [
    {start: NOW, end: NOW + YEAR},
    {start: NOW + YEAR, end: NOW + YEAR},
    {start: -NOW, end: -NOW + 1n},
    {start: -NOW - 1n, end: -NOW}
  ];
  const results = await runWordFilter(device, {starts, windows});
  for (const [index, timeWindow] of windows.entries()) {
    expect(results[index].mask).toEqual(filterWordsOracle(starts, undefined, timeWindow).mask);
  }

  // Control: float32 absolute values cannot separate these rows, so the cheap path gets one wrong.
  let float32Mistakes = 0;
  for (const timeWindow of windows) {
    const exact = filterWordsOracle(starts, undefined, timeWindow).mask;
    const windowStart = Math.fround(Number(timeWindow.start));
    const windowEnd = Math.fround(Number(timeWindow.end));
    starts.forEach((start, row) => {
      const value = Math.fround(Number(start));
      const accepted = value >= windowStart && value <= windowEnd ? 1 : 0;
      if (accepted !== exact[row]) float32Mistakes++;
    });
  }
  expect(float32Mistakes).toBeGreaterThan(0);
});

it('GPUTimeWindowFilter word intervals match fade weights and clip fractions', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const starts = [-100n, -20n, 0n, 40n, 90n, 150n].map(offset => NOW + offset);
  const ends = [-50n, 30n, 80n, 100n, 200n, 160n].map(offset => NOW + offset);
  const windows: GPUTimeWordWindow[] = [
    {start: NOW, end: NOW + 100n, startFadeDuration: 20, endFadeDuration: 30},
    {start: Number(NOW) + 10.5, end: Number(NOW) + 95.25, startFadeDuration: 40}
  ];
  const results = await runWordFilter(device, {
    starts,
    ends,
    windows,
    withFade: true,
    withClip: true
  });
  for (const [index, timeWindow] of windows.entries()) {
    const oracle = filterWordsOracle(starts, ends, timeWindow);
    expect(results[index].mask).toEqual(oracle.mask);
    expectClose(results[index].weights, oracle.weights);
    expectClose(results[index].clip, oracle.clip);
  }
  expect(results[0].mask.some(value => value === 1)).toBe(true);
});

it('GPUTimeWindowFilter word windows honor a fractional playhead', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const starts = Array.from({length: 13}, (_, offset) => NOW + BigInt(offset));
  const timeWindow = {start: Number(NOW) + 0.5, end: Number(NOW) + 10.5};
  const [result] = await runWordFilter(device, {
    starts,
    windows: [timeWindow]
  });
  expect(result.mask).toEqual([0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]);
  expect(result.mask).toEqual(filterWordsOracle(starts, undefined, timeWindow).mask);
});

it('GPUTimeWindowFilter word windows update per frame without recompiling', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const starts = [0n, 1n, 2n, 3n, 4n, 5n].map(offset => NOW + offset * 1000n);
  const windows: GPUTimeWordWindow[] = [0, 2, 4, 5].map(index => ({
    start: starts[index],
    end: starts[index] + 1500n
  }));
  const results = await runWordFilter(device, {starts, windows});
  expect(results.map(result => result.ids)).toEqual([[0, 1], [2, 3], [4, 5], [5]]);
});

it('GPUTimeWindowFilter word timestamps work across vector chunks', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const starts = [0x1_ffff_fffen, 0x1_ffff_ffffn, 0x2_0000_0000n, 0x2_0000_0001n, 0x2_0000_0002n];
  const ends = starts.map(start => start + 1n);
  const windows: GPUTimeWordWindow[] = [
    {start: 0x2_0000_0000n, end: 0x2_0000_0001n, endFadeDuration: 2}
  ];
  const results = await runWordFilter(device, {
    starts,
    ends,
    windows,
    chunkSize: 2,
    withFade: true
  });
  const oracle = filterWordsOracle(starts, ends, windows[0]);
  expect(results[0].mask).toEqual(oracle.mask);
  expect(results[0].mask).toEqual([0, 1, 1, 1, 0]);
  expectClose(results[0].weights, oracle.weights);
});

it('GPUTimeWindowFilter validates word and float inputs', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const graph = new GPUCommandGraph(device, {id: 'time-window-validation'});
  const wordsBuffer = createWordBuffer(device, [1n, 2n]);
  const floatsBuffer = createInputBuffer(device, Float32Array.from([1, 2]));
  const idsBuffer = createOutputBuffer(device, 2);
  const countBuffer = createOutputBuffer(device, 1);
  const overflowBuffer = createOutputBuffer(device, 1);
  const wordWindow = new GPUMapGraphParameterBuffer(device, {
    id: 'word-window',
    format: 'uint32',
    length: GPU_TIME_WORD_WINDOW_PARAMETER_LENGTH
  });
  const shortWordWindow = new GPUMapGraphParameterBuffer(device, {
    id: 'short-word-window',
    format: 'uint32',
    length: 4
  });
  const floatWindow = new GPUMapGraphParameterBuffer(device, {
    id: 'float-window',
    format: 'float32',
    length: 8,
    values: getGPUTimeWindowParameterValues({start: 0, end: 1})
  });
  const words = importGraphBuffer(graph, 'words', wordsBuffer, 'uint32x2', 2);
  const floats = importGraphBuffer(graph, 'floats', floatsBuffer, 'float32', 2);
  const output = {
    ids: importGraphBuffer(graph, 'ids', idsBuffer, 'uint32', 2),
    count: importGraphBuffer(graph, 'count', countBuffer, 'uint32', 1),
    overflow: importGraphBuffer(graph, 'overflow', overflowBuffer, 'uint32', 1)
  };
  const base = {output};
  expect(
    () =>
      new GPUTimeWindowFilter({
        ...base,
        timestamps: words,
        window: floatWindow.importToGraph(graph, 'fw')
      })
  ).toThrow(/word window/);
  expect(
    () =>
      new GPUTimeWindowFilter({
        ...base,
        timestamps: words,
        window: shortWordWindow.importToGraph(graph, 'sw')
      })
  ).toThrow(/must hold 8 uint32/);
  expect(
    () =>
      new GPUTimeWindowFilter({
        ...base,
        timestamps: floats,
        window: wordWindow.importToGraph(graph, 'ww')
      })
  ).toThrow(/float32 window/);
  expect(
    () =>
      new GPUTimeWindowFilter({
        ...base,
        timestamps: words,
        timestampsLow: floats,
        window: wordWindow.importToGraph(graph, 'ww2')
      })
  ).toThrow(/require float32 timestamps/);
  expect(
    () =>
      new GPUTimeWindowFilter({
        ...base,
        timestamps: words,
        endTimestamps: floats,
        window: wordWindow.importToGraph(graph, 'ww3')
      })
  ).toThrow(/same format/);
  expect(
    () =>
      new GPUTimeWindowFilter({
        ...base,
        timestamps: floats,
        endTimestamps: words,
        window: floatWindow.importToGraph(graph, 'fw2')
      })
  ).toThrow(/same format/);

  for (const buffer of [wordsBuffer, floatsBuffer, idsBuffer, countBuffer, overflowBuffer]) {
    buffer.destroy();
  }
  for (const buffer of [wordWindow, shortWordWindow, floatWindow]) buffer.destroy();
});
