// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';
import {DrawCommandBuffer, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it, vi} from 'vitest';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {GPUResidentRowSelection} from '../../../src/gpu-tables/residency-arena/gpu-resident-row-selection';
import {GPU_RESIDENCY_ARENA_DEAD_SLOT} from '../../../src/gpu-tables/residency-arena/residency-arena-types';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {selectResidentRowsOracle} from './resident-row-selection-oracle';

const ROW_CAPACITY = 4096;
const PAGE_ROWS = 64;
const TILE_MASK_LENGTH = 32;
const MAXIMUM_SLOT = 40;

/** Seeded mulberry32 generator. */
function createRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

type ArenaState = {
  liveMask: Uint32Array;
  rowTileSlots: Uint32Array;
  tileMask: Uint32Array;
  predicateMask: Uint32Array;
};

/** Page-like live runs: whole pages live or dead, with a partially live tail on some pages. */
function createArenaState(seed: number, pageLiveProbability: number): ArenaState {
  const random = createRandom(seed);
  const liveMask = new Uint32Array(ROW_CAPACITY);
  const rowTileSlots = new Uint32Array(ROW_CAPACITY).fill(GPU_RESIDENCY_ARENA_DEAD_SLOT);
  const predicateMask = new Uint32Array(ROW_CAPACITY);
  for (let page = 0; page < ROW_CAPACITY / PAGE_ROWS; page++) {
    const slot = Math.floor(random() * MAXIMUM_SLOT);
    const liveRows =
      random() < pageLiveProbability
        ? random() < 0.3
          ? Math.floor(random() * PAGE_ROWS)
          : PAGE_ROWS
        : 0;
    for (let row = page * PAGE_ROWS; row < page * PAGE_ROWS + liveRows; row++) {
      liveMask[row] = 1;
      rowTileSlots[row] = slot;
    }
  }
  for (let row = 0; row < ROW_CAPACITY; row++) {
    predicateMask[row] = random() < 0.7 ? Math.floor(random() * 5) + 1 : 0;
  }
  const tileMask = new Uint32Array(TILE_MASK_LENGTH);
  for (let slot = 0; slot < TILE_MASK_LENGTH; slot++) {
    tileMask[slot] = random() < 0.6 ? Math.floor(random() * 3) + 1 : 0;
  }
  return {liveMask, rowTileSlots, tileMask, predicateMask};
}

function destroyAll(buffers: Buffer[]): void {
  for (const buffer of buffers) {
    buffer.destroy();
  }
}

type FullHarness = {
  state: ArenaState;
  device: NonNullable<Awaited<ReturnType<typeof getWebGPUTestDevice>>>;
  compiled: ReturnType<GPUCommandGraph['compile']>;
  compileSpy: ReturnType<typeof vi.fn>;
  buffers: Record<string, Buffer>;
  drawCommands: DrawCommandBuffer;
  readDrawCount: () => Promise<number>;
  destroy: () => void;
};

/** Compiles ONE graph with every optional output over imported arena buffers. */
async function createFullHarness(
  capacity: number,
  options: {sourceIds: boolean; tiles: boolean}
): Promise<FullHarness | undefined> {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return undefined;
  }
  const sourceIds = Uint32Array.from({length: ROW_CAPACITY}, (_, row) => 5000 + row * 3);
  const buffers: Record<string, Buffer> = {
    live: createInputBuffer(device, new Uint32Array(ROW_CAPACITY)),
    slots: createInputBuffer(device, new Uint32Array(ROW_CAPACITY)),
    tileMask: createInputBuffer(device, new Uint32Array(TILE_MASK_LENGTH)),
    predicate: createInputBuffer(device, new Uint32Array(ROW_CAPACITY)),
    sourceIds: createInputBuffer(device, sourceIds),
    ids: createOutputBuffer(device, capacity),
    count: createOutputBuffer(device, 1),
    overflow: createOutputBuffer(device, 1),
    total: createOutputBuffer(device, 1),
    mask: createOutputBuffer(device, ROW_CAPACITY)
  };
  const drawCommands = new DrawCommandBuffer(device, {
    id: 'resident-draw',
    type: 'draw',
    commands: [{vertexCount: 6, instanceCount: 0}]
  });
  const graph = new GPUCommandGraph(device, {
    id: `resident-row-selection-${capacity}`
  });
  const view = <Format extends 'uint32'>(name: string, length: number) =>
    importGraphBuffer(graph, name, buffers[name], 'uint32' as Format, length);
  graph.add(
    new GPUResidentRowSelection({
      liveMask: view('live', ROW_CAPACITY),
      tileVisibility: options.tiles
        ? {
            rowTileSlots: view('slots', ROW_CAPACITY),
            tileMask: view('tileMask', TILE_MASK_LENGTH)
          }
        : undefined,
      additionalPredicates: [{kind: 'time-range', mask: view('predicate', ROW_CAPACITY)}],
      sourceIds: options.sourceIds ? view('sourceIds', ROW_CAPACITY) : undefined,
      output: {
        ids: view('ids', capacity),
        count: view('count', 1),
        overflow: view('overflow', 1),
        totalCount: view('total', 1)
      },
      outputMask: view('mask', ROW_CAPACITY),
      drawInstanceCount: graph.importGPUData('draw-count', drawCommands.getInstanceCountData(0))
    })
  );
  const compileSpy = vi.spyOn(graph, 'compile');
  const compiled = graph.compile();
  expect(compileSpy).toHaveBeenCalledTimes(1);
  return {
    state: createArenaState(0, 0),
    device,
    compiled,
    compileSpy,
    buffers,
    drawCommands,
    readDrawCount: async () => {
      const bytes = await drawCommands.buffer.readAsync(
        drawCommands.getInstanceCountByteOffset(0),
        4
      );
      return new Uint32Array(bytes.buffer, bytes.byteOffset, 1)[0];
    },
    destroy: () => {
      compiled.destroy();
      drawCommands.destroy();
      destroyAll(Object.values(buffers));
    }
  };
}

/** Rewrites the input buffers, re-encodes the same compiled graph, and checks oracle parity. */
async function expectStateMatchesOracle(
  harness: FullHarness,
  state: ArenaState,
  capacity: number,
  options: {sourceIds: boolean; tiles: boolean}
): Promise<ReturnType<typeof selectResidentRowsOracle>> {
  const {buffers, device, compiled} = harness;
  buffers['live'].write(state.liveMask);
  buffers['slots'].write(state.rowTileSlots);
  buffers['tileMask'].write(state.tileMask);
  buffers['predicate'].write(state.predicateMask);
  submitGraph(device, compiled, undefined);
  const expected = selectResidentRowsOracle({
    liveMask: state.liveMask,
    rowTileSlots: options.tiles ? state.rowTileSlots : undefined,
    tileMask: options.tiles ? state.tileMask : undefined,
    predicateMasks: [state.predicateMask],
    sourceIds: options.sourceIds
      ? Uint32Array.from({length: ROW_CAPACITY}, (_, row) => 5000 + row * 3)
      : undefined,
    capacity
  });
  const [count] = await readUint32(buffers['count'], 1);
  expect(count).toBe(expected.count);
  expect(await readUint32(buffers['ids'], count)).toEqual(expected.ids);
  expect(await readUint32(buffers['total'], 1)).toEqual([expected.total]);
  expect(await readUint32(buffers['overflow'], 1)).toEqual([expected.overflow]);
  expect(await readUint32(buffers['mask'], ROW_CAPACITY)).toEqual(expected.mask);
  expect(await harness.readDrawCount()).toBe(expected.count);
  return expected;
}

it('GPUResidentRowSelection matches the oracle over 24 seeded states on one compiled graph', async () => {
  const options = {sourceIds: true, tiles: true};
  const capacity = 1024;
  const harness = await createFullHarness(capacity, options);
  if (!harness) {
    return;
  }
  const compiledGraph = harness.compiled;
  let sawOverflow = false;
  let sawPartial = false;
  for (let stateIndex = 0; stateIndex < 24; stateIndex++) {
    const state =
      stateIndex === 0
        ? createArenaState(1, 0)
        : createArenaState(1000 + stateIndex, 0.1 + 0.9 * ((stateIndex % 8) / 7));
    const expected = await expectStateMatchesOracle(harness, state, capacity, options);
    sawOverflow ||= expected.overflow === 1;
    sawPartial ||= expected.total > 0 && expected.overflow === 0;
    if (stateIndex === 0) {
      // All-dead arena: nothing selected.
      expect(expected.count).toBe(0);
      expect(expected.total).toBe(0);
    }
  }
  expect(sawOverflow).toBe(true);
  expect(sawPartial).toBe(true);
  // Same compiled graph object, and graph.compile was never called again.
  expect(harness.compiled).toBe(compiledGraph);
  expect(harness.compileSpy).toHaveBeenCalledTimes(1);
  harness.destroy();
});

it('GPUResidentRowSelection reports overflow for capacity below the accepted count', async () => {
  const options = {sourceIds: false, tiles: true};
  const capacity = 50;
  const harness = await createFullHarness(capacity, options);
  if (!harness) {
    return;
  }
  const state = createArenaState(77, 1);
  state.tileMask.fill(1);
  const expected = await expectStateMatchesOracle(harness, state, capacity, options);
  expect(expected.total).toBeGreaterThan(capacity);
  expect(expected.overflow).toBe(1);
  expect(expected.count).toBe(capacity);
  // Changing only buffer contents recovers from overflow without recompiling.
  state.tileMask.fill(0);
  const cleared = await expectStateMatchesOracle(harness, state, capacity, options);
  expect(cleared.overflow).toBe(0);
  expect(cleared.count).toBe(0);
  expect(harness.compileSpy).toHaveBeenCalledTimes(1);
  harness.destroy();
});

it('GPUResidentRowSelection works with only a live mask and full-size output', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const buffers = {
    live: createInputBuffer(device, new Uint32Array(ROW_CAPACITY)),
    ids: createOutputBuffer(device, ROW_CAPACITY),
    count: createOutputBuffer(device, 1),
    overflow: createOutputBuffer(device, 1)
  };
  const graph = new GPUCommandGraph(device, {id: 'resident-live-only'});
  graph.add(
    new GPUResidentRowSelection({
      liveMask: importGraphBuffer(graph, 'live', buffers.live, 'uint32', ROW_CAPACITY),
      output: {
        ids: importGraphBuffer(graph, 'ids', buffers.ids, 'uint32', ROW_CAPACITY),
        count: importGraphBuffer(graph, 'count', buffers.count, 'uint32', 1),
        overflow: importGraphBuffer(graph, 'overflow', buffers.overflow, 'uint32', 1)
      }
    })
  );
  const compiled = graph.compile();
  for (let stateIndex = 0; stateIndex < 4; stateIndex++) {
    const {liveMask} = createArenaState(500 + stateIndex, 0.3 * stateIndex);
    buffers.live.write(liveMask);
    submitGraph(device, compiled, undefined);
    const expected = selectResidentRowsOracle({
      liveMask,
      capacity: ROW_CAPACITY
    });
    const [count] = await readUint32(buffers.count, 1);
    expect(count).toBe(expected.count);
    expect(await readUint32(buffers.ids, count)).toEqual(expected.ids);
    expect(await readUint32(buffers.overflow, 1)).toEqual([0]);
  }
  compiled.destroy();
  destroyAll(Object.values(buffers));
});
