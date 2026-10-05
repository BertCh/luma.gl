// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUCommandGraph,
  GPUVisibilityWorkflow,
  type GraphDataView,
  type GraphVectorView
} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {GPUTimeWindowFilter} from '../../../src/gpu-dataframe/time-window-filter';
import {GPUResidentRowSelection} from '../../../src/gpu-tables/residency-arena/gpu-resident-row-selection';
import {createNullWebGPUDevice, createVectorView} from '../../utils/gpu-contributor-test-utils';

const ROW_COUNT = 1_048_576;
const CHUNK_COUNTS = [1, 8, 32, 128];

/** A chunked uint32 or float32 vector of `ROW_COUNT` rows split into `chunkCount` equal chunks. */
function createChunkedView<Format extends 'uint32' | 'float32'>(
  graph: GPUCommandGraph,
  id: string,
  format: Format,
  chunkCount: number
): GraphVectorView<Format> {
  const chunkLength = ROW_COUNT / chunkCount;
  const chunks: GraphDataView<Format>[] = [];
  for (let chunk = 0; chunk < chunkCount; chunk++) {
    chunks.push(createTransientView(graph, `${id}-chunk-${chunk}`, format, chunkLength));
  }
  return createVectorView(id, format, chunks);
}

/** Node counts of the four shapes at one chunk count. */
function countNodes(chunkCount: number, tileCount: number) {
  const device = createNullWebGPUDevice();
  const count = (build: (graph: GPUCommandGraph) => {getCommandNodes: unknown}): number => {
    const graph = new GPUCommandGraph(device);
    const recipe = build(graph) as {
      getCommandNodes: (graph: GPUCommandGraph) => readonly unknown[];
    };
    return recipe.getCommandNodes(graph).length;
  };
  const scalarOutputs = (graph: GPUCommandGraph) => ({
    count: createTransientView(graph, 'count', 'uint32', 1),
    overflow: createTransientView(graph, 'overflow', 'uint32', 1)
  });

  // (a) chunked masks (time-style + live), chunked ids output: the V3 shape.
  const chunkedVector = count(graph => {
    return new GPUVisibilityWorkflow({
      predicates: [
        {kind: 'time-range', mask: createChunkedView(graph, 'time-mask', 'uint32', chunkCount)},
        {kind: 'selection', mask: createChunkedView(graph, 'live', 'uint32', chunkCount)}
      ],
      output: createChunkedView(graph, 'ids', 'uint32', chunkCount),
      count: createTransientView(graph, 'total', 'uint32', 1)
    });
  });
  // (b) same inputs, one packed ids view.
  const chunkedInputPackedOutput = count(graph => {
    return new GPUVisibilityWorkflow({
      predicates: [
        {kind: 'time-range', mask: createChunkedView(graph, 'time-mask', 'uint32', chunkCount)},
        {kind: 'selection', mask: createChunkedView(graph, 'live', 'uint32', chunkCount)}
      ],
      output: createTransientView(graph, 'ids', 'uint32', ROW_COUNT),
      count: createTransientView(graph, 'total', 'uint32', 1)
    });
  });
  // (c) GPUTimeWindowFilter over chunked timestamps and a chunked live predicate.
  const chunkedTimeFilter = count(graph => {
    return new GPUTimeWindowFilter({
      timestamps: createChunkedView(graph, 'times', 'float32', chunkCount),
      window: createTransientView(graph, 'window', 'float32', 8),
      additionalPredicates: [
        {kind: 'selection', mask: createChunkedView(graph, 'live', 'uint32', chunkCount)}
      ],
      output: {ids: createTransientView(graph, 'ids', 'uint32', ROW_COUNT), ...scalarOutputs(graph)}
    });
  });
  // (d) arena shape: single views over the whole capacity, whatever the chunk or tile count.
  const arenaTimeFilter = count(graph => {
    return new GPUTimeWindowFilter({
      timestamps: createTransientView(graph, 'times', 'float32', ROW_COUNT),
      window: createTransientView(graph, 'window', 'float32', 8),
      additionalPredicates: [
        {kind: 'selection', mask: createTransientView(graph, 'live', 'uint32', ROW_COUNT)}
      ],
      output: {
        ids: createTransientView(graph, 'ids', 'uint32', ROW_COUNT),
        ...scalarOutputs(graph)
      },
      drawInstanceCount: createTransientView(graph, 'draw', 'uint32', 1)
    });
  });
  const arenaRowSelection = count(graph => {
    return new GPUResidentRowSelection({
      liveMask: createTransientView(graph, 'live', 'uint32', ROW_COUNT),
      tileVisibility: {
        rowTileSlots: createTransientView(graph, 'slots', 'uint32', ROW_COUNT),
        tileMask: createTransientView(graph, 'tile-mask', 'uint32', tileCount)
      },
      additionalPredicates: [
        {kind: 'time-range', mask: createTransientView(graph, 'time-mask', 'uint32', ROW_COUNT)}
      ],
      output: {
        ids: createTransientView(graph, 'ids', 'uint32', ROW_COUNT),
        ...scalarOutputs(graph)
      },
      drawInstanceCount: createTransientView(graph, 'draw', 'uint32', 1)
    });
  });
  device.destroy();
  return {
    chunkedVector,
    chunkedInputPackedOutput,
    chunkedTimeFilter,
    arenaTimeFilter,
    arenaRowSelection
  };
}

it('arena graphs have a constant node count while chunked graphs grow with the chunk count', () => {
  const rows = CHUNK_COUNTS.map(chunkCount => ({
    chunkCount,
    tiles: chunkCount === 1 ? 1 : 1000,
    ...countNodes(chunkCount, chunkCount === 1 ? 1 : 1000)
  }));
  // eslint-disable-next-line no-console
  console.log(
    [
      `residency arena node counts, ${ROW_COUNT} rows, getCommandNodes(graph).length`,
      'chunks  tiles  (a) chunked-vector  (b) chunked-in/packed-out  (c) chunked time filter  (d1) arena time filter  (d2) arena row selection',
      ...rows.map(
        row =>
          `${String(row.chunkCount).padStart(6)} ${String(row.tiles).padStart(6)} ` +
          `${String(row.chunkedVector).padStart(18)} ${String(row.chunkedInputPackedOutput).padStart(23)} ` +
          `${String(row.chunkedTimeFilter).padStart(23)} ${String(row.arenaTimeFilter).padStart(21)} ` +
          `${String(row.arenaRowSelection).padStart(25)}`
      )
    ].join('\n')
  );

  // (d) does not depend on the chunk count or the number of resident tiles.
  const arenaTimeFilterCounts = new Set(rows.map(row => row.arenaTimeFilter));
  const arenaRowSelectionCounts = new Set(rows.map(row => row.arenaRowSelection));
  expect(arenaTimeFilterCounts.size).toBe(1);
  expect(arenaRowSelectionCounts.size).toBe(1);
  expect(countNodes(1, 1000).arenaRowSelection).toBe(rows[0].arenaRowSelection);

  const at = (chunkCount: number) => rows.find(row => row.chunkCount === chunkCount)!;
  // (a) grows with chunks^2: quadrupling the chunk count multiplies the count by 9 to 16 (linear
  // terms dominate at small counts); at 128 chunks the quadratic term is visible.
  for (const [small, large] of [
    [8, 32],
    [32, 128]
  ]) {
    const ratio = at(large).chunkedVector / at(small).chunkedVector;
    expect(ratio).toBeGreaterThan(8);
    expect(ratio).toBeLessThan(18);
  }
  expect(at(128).chunkedVector).toBeGreaterThanOrEqual(128 * 128);
  // (b) and (c) grow with chunks: quadrupling the chunk count multiplies the count by about 4.
  for (const key of ['chunkedInputPackedOutput', 'chunkedTimeFilter'] as const) {
    for (const [small, large] of [
      [8, 32],
      [32, 128]
    ]) {
      const ratio = at(large)[key] / at(small)[key];
      expect(ratio).toBeGreaterThan(2.5);
      expect(ratio).toBeLessThan(5);
    }
  }
  // The arena shape is smaller than every chunked shape beyond one chunk.
  expect(at(8).arenaTimeFilter).toBeLessThan(at(8).chunkedInputPackedOutput);
  expect(at(8).arenaTimeFilter).toBeLessThan(at(8).chunkedVector);
});
