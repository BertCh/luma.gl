// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it, vi} from 'vitest';
import {GPUParameterBuffer, importGraphBuffer, submitGraph} from '../../../src/utils/gpu-contributor-utils';
import {
  getGPULineSimplificationParameterValues,
  GPULineSimplification
} from '../../../src/geospatial/line-simplification';
import {createInputBuffer, createOutputBuffer, readUint32} from '../../utils/gpu-contributor-test-utils';
import {
  computeParallelImportance,
  getKeptLineRanges,
  getKeptRowsFromImportance,
  simplifyDouglasPeucker,
  type LineSimplificationScene
} from './line-simplification-oracle';
import {
  createCornerCaseScene,
  createRandom,
  createRandomTracksScene,
  createScene,
  createSpiralScene
} from './line-simplification-scenes';

type ImportanceResult = {
  importanceBits: Uint32Array;
  converged: number;
  roundCount: number;
};

type SelectionResult = {
  ids: number[];
  count: number;
  overflow: number;
  totalCount: number;
  keepMask: number[];
  lineCounts: number[];
  lineStarts: number[];
};

type Fixture = {
  /** Runs the one-shot importance graph. */
  computeImportance(): Promise<ImportanceResult>;
  /** Runs the per-frame selection graph at `tolerance`. */
  select(tolerance: number): Promise<SelectionResult>;
  /** Number of `compile()` calls on either graph after setup. */
  getCompileCount(): number;
  destroy(): void;
};

/**
 * Two graphs, as an application would use them: a one-shot importance graph and a per-frame
 * selection graph that reads the importance buffer and a tolerance parameter buffer.
 */
function createFixture(
  device: Device,
  scene: LineSimplificationScene,
  options: {maximumRounds?: number; capacity?: number} = {}
): Fixture {
  const rowCount = scene.positions.length / 2;
  const lineCount = scene.trackOffsets.length - 1;
  const capacity = options.capacity ?? rowCount;
  const buffers: Buffer[] = [];
  const input = (values: Float32Array | Uint32Array) => {
    const buffer = createInputBuffer(device, values);
    buffers.push(buffer);
    return buffer;
  };
  const output = (length: number) => {
    const buffer = createOutputBuffer(device, length);
    buffers.push(buffer);
    return buffer;
  };
  const positions = input(scene.positions);
  const trackOffsets = input(scene.trackOffsets);
  const timestamps = input(scene.timestamps ?? new Float32Array(rowCount));
  const importance = output(rowCount);
  const converged = output(1);
  const roundCount = output(1);
  const outputs = {
    ids: output(capacity),
    count: output(1),
    overflow: output(1),
    totalCount: output(1),
    keepMask: output(rowCount),
    lineCounts: output(lineCount),
    lineStarts: output(lineCount)
  };

  const importanceGraph = new GPUCommandGraph(device, {
    id: 'line-importance-graph'
  });
  importanceGraph.add(
    new GPULineSimplification({
      id: 'importance',
      positions: importGraphBuffer(importanceGraph, 'positions', positions, 'float32x2', rowCount),
      trackOffsets: importGraphBuffer(
        importanceGraph,
        'offsets',
        trackOffsets,
        'uint32',
        lineCount + 1
      ),
      timestamps: importGraphBuffer(importanceGraph, 'times', timestamps, 'float32', rowCount),
      metric: scene.metric,
      maximumRounds: options.maximumRounds,
      importance: importGraphBuffer(importanceGraph, 'importance', importance, 'float32', rowCount),
      status: {
        converged: importGraphBuffer(importanceGraph, 'converged', converged, 'uint32', 1),
        roundCount: importGraphBuffer(importanceGraph, 'round-count', roundCount, 'uint32', 1)
      }
    })
  );
  const importanceCompiled = importanceGraph.compile();

  const parameterBuffer = new GPUParameterBuffer(device, {
    id: 'tolerance',
    format: 'float32',
    length: 4
  });
  const selectionGraph = new GPUCommandGraph(device, {
    id: 'line-selection-graph'
  });
  selectionGraph.add(
    new GPULineSimplification({
      id: 'selection',
      positions: importGraphBuffer(selectionGraph, 'positions', positions, 'float32x2', rowCount),
      trackOffsets: importGraphBuffer(
        selectionGraph,
        'offsets',
        trackOffsets,
        'uint32',
        lineCount + 1
      ),
      importance: importGraphBuffer(selectionGraph, 'importance', importance, 'float32', rowCount),
      computeImportance: false,
      parameters: parameterBuffer.importToGraph(selectionGraph),
      selection: {
        output: {
          ids: importGraphBuffer(selectionGraph, 'o-ids', outputs.ids, 'uint32', capacity),
          count: importGraphBuffer(selectionGraph, 'o-count', outputs.count, 'uint32', 1),
          overflow: importGraphBuffer(selectionGraph, 'o-overflow', outputs.overflow, 'uint32', 1),
          totalCount: importGraphBuffer(selectionGraph, 'o-total', outputs.totalCount, 'uint32', 1)
        },
        keepMask: importGraphBuffer(selectionGraph, 'o-mask', outputs.keepMask, 'uint32', rowCount),
        lineCounts: importGraphBuffer(
          selectionGraph,
          'o-line-counts',
          outputs.lineCounts,
          'uint32',
          lineCount
        ),
        lineStarts: importGraphBuffer(
          selectionGraph,
          'o-line-starts',
          outputs.lineStarts,
          'uint32',
          lineCount
        )
      }
    })
  );
  const selectionCompiled = selectionGraph.compile();
  const compileSpies = [vi.spyOn(importanceGraph, 'compile'), vi.spyOn(selectionGraph, 'compile')];

  return {
    async computeImportance() {
      submitGraph(device, importanceCompiled, undefined);
      return {
        importanceBits: Uint32Array.from(await readUint32(importance, rowCount)),
        converged: (await readUint32(converged, 1))[0],
        roundCount: (await readUint32(roundCount, 1))[0]
      };
    },
    async select(tolerance) {
      parameterBuffer.write(getGPULineSimplificationParameterValues({tolerance}));
      submitGraph(device, selectionCompiled, undefined);
      const [count] = await readUint32(outputs.count, 1);
      return {
        ids: await readUint32(outputs.ids, count),
        count,
        overflow: (await readUint32(outputs.overflow, 1))[0],
        totalCount: (await readUint32(outputs.totalCount, 1))[0],
        keepMask: await readUint32(outputs.keepMask, rowCount),
        lineCounts: await readUint32(outputs.lineCounts, lineCount),
        lineStarts: await readUint32(outputs.lineStarts, lineCount)
      };
    },
    getCompileCount() {
      return compileSpies.reduce((sum, spy) => sum + spy.mock.calls.length, 0);
    },
    destroy() {
      importanceCompiled.destroy();
      selectionCompiled.destroy();
      parameterBuffer.destroy();
      for (const buffer of buffers) {
        buffer.destroy();
      }
    }
  };
}

function expectSelectionParity(
  actual: SelectionResult,
  scene: LineSimplificationScene,
  importanceBits: Uint32Array,
  tolerance: number,
  capacity: number
): number[] {
  const kept = getKeptRowsFromImportance(scene, importanceBits, tolerance);
  expect(actual.totalCount).toBe(kept.length);
  expect(actual.count).toBe(Math.min(kept.length, capacity));
  expect(actual.overflow).toBe(kept.length > capacity ? 1 : 0);
  expect(actual.ids).toEqual(kept.slice(0, capacity));
  const keptSet = new Set(kept);
  expect(actual.keepMask).toEqual(actual.keepMask.map((_, row) => (keptSet.has(row) ? 1 : 0)));
  const ranges = getKeptLineRanges(scene, kept);
  expect(actual.lineCounts).toEqual(ranges.lineCounts);
  expect(actual.lineStarts).toEqual(ranges.lineStarts);
  return kept;
}

async function expectSceneParity(
  device: Device,
  scene: LineSimplificationScene,
  tolerances: readonly number[]
): Promise<void> {
  const fixture = createFixture(device, scene);
  const actual = await fixture.computeImportance();
  const expected = computeParallelImportance(scene, 64);
  expect(actual.converged).toBe(1);
  expect(actual.roundCount).toBe(expected.roundCount);
  expect(Array.from(actual.importanceBits)).toEqual(Array.from(expected.importanceBits));
  for (const tolerance of tolerances) {
    const kept = expectSelectionParity(
      await fixture.select(tolerance),
      scene,
      actual.importanceBits,
      tolerance,
      scene.positions.length / 2
    );
    expect(kept, `tolerance ${tolerance}`).toEqual(simplifyDouglasPeucker(scene, tolerance));
  }
  expect(fixture.getCompileCount()).toBe(0);
  fixture.destroy();
}

it('GPULineSimplification matches the oracle and Douglas-Peucker on corner cases', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  await expectSceneParity(device, createCornerCaseScene(), [0, 0.5, 1, 2 / Math.sqrt(10), 3]);
});

it('GPULineSimplification matches the oracle on many random tracks with varied lengths', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  for (const seed of [1, 2]) {
    await expectSceneParity(
      device,
      createRandomTracksScene(seed, 300, 120),
      [0, 0.25, 0.5, 1, 1.75, 3, 8, 40]
    );
  }
});

it('GPULineSimplification is bit-exact on unquantized float coordinates', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  // Inexact products and differences: a fused multiply-add anywhere would change the bits.
  const random = createRandom(9);
  const lines = Array.from({length: 1500}, () =>
    Array.from({length: 3 + Math.floor(random() * 6)}, () => [random() * 20, random() * 20])
  );
  await expectSceneParity(device, createScene(lines), [0, 0.5, 1, 3, 8]);
});

it('GPULineSimplification matches the oracle with the TD-TR time-ratio metric', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  await expectSceneParity(
    device,
    createRandomTracksScene(7, 200, 100, 'time-ratio'),
    [0, 0.5, 1, 2, 6, 20]
  );
});

it('GPULineSimplification changes tolerance per frame without recompiling', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const scene = createRandomTracksScene(11, 100, 80);
  const capacity = 150;
  const fixture = createFixture(device, scene, {capacity});
  const {importanceBits} = await fixture.computeImportance();
  let sawOverflow = false;
  for (const tolerance of [40, 2, 0, 0.5, 6, 40]) {
    const actual = await fixture.select(tolerance);
    expectSelectionParity(actual, scene, importanceBits, tolerance, capacity);
    sawOverflow ||= actual.overflow === 1;
  }
  expect(sawOverflow).toBe(true);
  expect(fixture.getCompileCount()).toBe(0);
  fixture.destroy();
});

it('GPULineSimplification reports a deep spiral that hits the round cap', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const scene = createSpiralScene(300);
  const full = computeParallelImportance(scene, 1024);
  const maximumRounds = 6;
  expect(full.roundCount).toBeGreaterThan(maximumRounds);

  const capped = createFixture(device, scene, {maximumRounds});
  const actual = await capped.computeImportance();
  const expected = computeParallelImportance(scene, maximumRounds);
  expect(actual.converged).toBe(0);
  expect(actual.roundCount).toBe(maximumRounds);
  expect(Array.from(actual.importanceBits)).toEqual(Array.from(expected.importanceBits));
  for (const tolerance of [0.5, 2, 5]) {
    const kept = new Set((await capped.select(tolerance)).ids);
    for (const row of simplifyDouglasPeucker(scene, tolerance)) {
      expect(kept.has(row)).toBe(true);
    }
  }
  capped.destroy();

  const uncapped = createFixture(device, scene, {
    maximumRounds: full.roundCount
  });
  const converged = await uncapped.computeImportance();
  expect(converged.converged).toBe(1);
  expect(converged.roundCount).toBe(full.roundCount);
  expect(Array.from(converged.importanceBits)).toEqual(Array.from(full.importanceBits));
  for (const tolerance of [0.5, 2, 5]) {
    expect((await uncapped.select(tolerance)).ids).toEqual(
      simplifyDouglasPeucker(scene, tolerance)
    );
  }
  uncapped.destroy();
});

it('GPULineSimplification computes importance and selection in one graph', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const scene = createRandomTracksScene(5, 20, 40);
  const rowCount = scene.positions.length / 2;
  const lineCount = scene.trackOffsets.length - 1;
  const graph = new GPUCommandGraph(device, {id: 'line-both-graph'});
  const buffers = {
    positions: createInputBuffer(device, scene.positions),
    offsets: createInputBuffer(device, scene.trackOffsets),
    importance: createOutputBuffer(device, rowCount),
    ids: createOutputBuffer(device, rowCount),
    count: createOutputBuffer(device, 1),
    overflow: createOutputBuffer(device, 1)
  };
  const parameterBuffer = new GPUParameterBuffer(device, {
    id: 'tolerance',
    format: 'float32',
    length: 4,
    values: getGPULineSimplificationParameterValues({tolerance: 1.5})
  });
  graph.add(
    new GPULineSimplification({
      positions: importGraphBuffer(graph, 'positions', buffers.positions, 'float32x2', rowCount),
      trackOffsets: importGraphBuffer(graph, 'offsets', buffers.offsets, 'uint32', lineCount + 1),
      importance: importGraphBuffer(graph, 'importance', buffers.importance, 'float32', rowCount),
      parameters: parameterBuffer.importToGraph(graph),
      selection: {
        output: {
          ids: importGraphBuffer(graph, 'ids', buffers.ids, 'uint32', rowCount),
          count: importGraphBuffer(graph, 'count', buffers.count, 'uint32', 1),
          overflow: importGraphBuffer(graph, 'overflow', buffers.overflow, 'uint32', 1)
        }
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const [count] = await readUint32(buffers.count, 1);
  expect(await readUint32(buffers.ids, count)).toEqual(simplifyDouglasPeucker(scene, 1.5));
  compiled.destroy();
  parameterBuffer.destroy();
  for (const buffer of Object.values(buffers)) {
    buffer.destroy();
  }
});
