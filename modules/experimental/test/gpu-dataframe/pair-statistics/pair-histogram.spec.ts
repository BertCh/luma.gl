// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';
import {GPUCommandGraph, type GPUCommandNode} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUParameterBuffer, importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {createWGSLKernelNode} from '../../../src/utils/wgsl-kernel-nodes';
import {
  getPairHistogramNodes,
  getPairHistogramReadWGSL
} from '../../../src/gpu-dataframe/pair-statistics/pair-histogram';
import {getPairStatisticsInputNodes} from '../../../src/gpu-dataframe/pair-statistics/pair-statistics-grid';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';

const BIN_COUNT = 8;
const DISTANCE_SCALE = 2 ** 20;

function createRandom(seed: number): () => number {
  let state = seed >>> 0 || 1;
  return () => {
    state ^= state << 13;
    state >>>= 0;
    state ^= state >>> 17;
    state ^= state << 5;
    state >>>= 0;
    return state / 4294967296;
  };
}

it('pair histogram counts and fixed-point distance sums match a CPU pair loop', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const rows = 1500;
  const random = createRandom(7);
  const positions = new Float32Array(rows * 2);
  const mask = new Uint32Array(rows);
  for (let row = 0; row < rows; row++) {
    positions[row * 2] = random() * 100;
    positions[row * 2 + 1] = random() * 60;
    mask[row] = random() < 0.9 ? 1 : 0;
  }
  const buffers: Buffer[] = [];
  const positionsBuffer = createInputBuffer(device, positions);
  const maskBuffer = createInputBuffer(device, mask);
  const countsBuffer = createOutputBuffer(device, BIN_COUNT);
  const sumsBuffer = createOutputBuffer(device, BIN_COUNT);
  buffers.push(positionsBuffer, maskBuffer, countsBuffer, sumsBuffer);
  const parameterBuffer = new GPUParameterBuffer(device, {
    id: 'pair-parameters',
    format: 'float32',
    length: 5
  });
  const graph = new GPUCommandGraph(device, {id: 'pair-histogram-graph'});
  const positionsView = importGraphBuffer(graph, 'positions', positionsBuffer, 'float32x2', rows);
  const maskView = importGraphBuffer(graph, 'mask', maskBuffer, 'uint32', rows);
  const countsView = importGraphBuffer(graph, 'counts', countsBuffer, 'uint32', BIN_COUNT);
  const sumsView = importGraphBuffer(graph, 'sums', sumsBuffer, 'float32', BIN_COUNT);
  const parameters = parameterBuffer.importToGraph(graph);
  const gridSize = [16, 16] as const;
  graph.add({
    getCommandNodes(target: GPUCommandGraph): readonly GPUCommandNode[] {
      const inputs = getPairStatisticsInputNodes(target, {
        id: 'smoke',
        operation: 'PairHistogramSmoke',
        positions: positionsView,
        parameters,
        gridSize,
        mask: maskView
      });
      const histogram = getPairHistogramNodes(target, {
        id: 'smoke-histogram',
        operation: 'PairHistogramSmoke',
        positions: positionsView,
        parameters,
        gridSize,
        sortedRows: inputs.sortedRows,
        cellOffsets: inputs.cellOffsets,
        slotCount: BIN_COUNT,
        channelCount: 2,
        pairOrder: 'unordered',
        pairAction: `let bin = min(u32(pairDistance / lattice.maximumDistance * ${BIN_COUNT}.0), ${BIN_COUNT - 1}u);
            accumulate(bin, 0u, 1u);
            accumulate(bin, 1u, quantizePairAmount(pairDistance / lattice.maximumDistance * ${DISTANCE_SCALE}.0));`
      });
      return [
        ...inputs.nodes,
        ...histogram.nodes,
        createWGSLKernelNode(target, {
          id: 'smoke-finish',
          operation: 'PairHistogramSmoke',
          bindings: [
            {
              name: 'accumulators',
              view: histogram.accumulators,
              type: 'u32',
              access: 'read'
            },
            {
              name: 'parameters',
              view: parameters,
              type: 'f32',
              access: 'read'
            },
            {
              name: 'counts',
              view: countsView,
              type: 'u32',
              access: 'read_write'
            },
            {name: 'sums', view: sumsView, type: 'f32', access: 'read_write'}
          ],
          invocationCount: BIN_COUNT,
          declarations: getPairHistogramReadWGSL('accumulators', 2),
          body: `counts[countsOffset + index] = readPairCount(index, 0u);
  sums[sumsOffset + index] = readPairAccumulator(index, 1u) / ${DISTANCE_SCALE}.0 * parameters[parametersOffset + 4u];`
        })
      ];
    }
  });
  const compiled = graph.compile();
  try {
    for (const maximumDistance of [7, 250]) {
      parameterBuffer.write(new Float32Array([0, 0, 100, 60, maximumDistance]));
      submitGraph(device, compiled, undefined);
      const counts = await readUint32(countsBuffer, BIN_COUNT);
      const sums = await readFloat32(sumsBuffer, BIN_COUNT);
      const expectedCounts = new Array(BIN_COUNT).fill(0);
      const expectedSums = new Array(BIN_COUNT).fill(0);
      const maximumSquared = Math.fround(maximumDistance * maximumDistance);
      for (let i = 0; i < rows; i++) {
        for (let j = i + 1; j < rows; j++) {
          if (!mask[i] || !mask[j]) {
            continue;
          }
          const deltaX = Math.fround(positions[j * 2] - positions[i * 2]);
          const deltaY = Math.fround(positions[j * 2 + 1] - positions[i * 2 + 1]);
          const squared = Math.fround(Math.fround(deltaX * deltaX) + Math.fround(deltaY * deltaY));
          if (squared > maximumSquared) {
            continue;
          }
          const pairDistance = Math.sqrt(squared);
          const bin = Math.min(
            Math.floor((pairDistance / maximumDistance) * BIN_COUNT),
            BIN_COUNT - 1
          );
          expectedCounts[bin]++;
          expectedSums[bin] += pairDistance;
        }
      }
      expect(counts).toEqual(expectedCounts);
      for (let bin = 0; bin < BIN_COUNT; bin++) {
        expect(sums[bin]).toBeCloseTo(
          expectedSums[bin],
          -Math.log10(expectedSums[bin] * 1e-5 + 1e-6)
        );
      }
      submitGraph(device, compiled, undefined);
      expect(await readFloat32(sumsBuffer, BIN_COUNT)).toEqual(sums);
    }
  } finally {
    compiled.destroy();
    parameterBuffer.destroy();
    for (const buffer of buffers) {
      buffer.destroy();
    }
  }
});

/**
 * Runs a pair histogram over four mutually close points (six unordered pairs, all in slot 0) with
 * the given `pairAction` and returns the slot's low/high words read three ways.
 */
async function runForcedAccumulation(pairAction: string, slotCount = 1) {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return null;
  }
  const positions = new Float32Array([1, 1, 2, 1, 1, 2, 2, 2]);
  const buffers: Buffer[] = [];
  const positionsBuffer = createInputBuffer(device, positions);
  const countBuffer = createOutputBuffer(device, 1);
  const sumBuffer = createOutputBuffer(device, 1);
  const overflowBuffer = createOutputBuffer(device, 1);
  buffers.push(positionsBuffer, countBuffer, sumBuffer, overflowBuffer);
  const parameterBuffer = new GPUParameterBuffer(device, {
    id: 'forced-parameters',
    format: 'float32',
    length: 5
  });
  const graph = new GPUCommandGraph(device, {id: 'pair-histogram-forced'});
  const positionsView = importGraphBuffer(graph, 'positions', positionsBuffer, 'float32x2', 4);
  const countView = importGraphBuffer(graph, 'count', countBuffer, 'uint32', 1);
  const sumView = importGraphBuffer(graph, 'sum', sumBuffer, 'float32', 1);
  const overflowView = importGraphBuffer(graph, 'overflow', overflowBuffer, 'uint32', 1);
  const parameters = parameterBuffer.importToGraph(graph);
  graph.add({
    getCommandNodes(target: GPUCommandGraph): readonly GPUCommandNode[] {
      const inputs = getPairStatisticsInputNodes(target, {
        id: 'forced',
        operation: 'PairHistogramForced',
        positions: positionsView,
        parameters,
        gridSize: [2, 2],
        mask: undefined
      });
      const histogram = getPairHistogramNodes(target, {
        id: 'forced-histogram',
        operation: 'PairHistogramForced',
        positions: positionsView,
        parameters,
        gridSize: [2, 2],
        sortedRows: inputs.sortedRows,
        cellOffsets: inputs.cellOffsets,
        slotCount,
        channelCount: 1,
        pairOrder: 'unordered',
        pairAction
      });
      return [
        ...inputs.nodes,
        ...histogram.nodes,
        createWGSLKernelNode(target, {
          id: 'forced-finish',
          operation: 'PairHistogramForced',
          bindings: [
            {name: 'accumulators', view: histogram.accumulators, type: 'u32', access: 'read'},
            {name: 'count', view: countView, type: 'u32', access: 'read_write'},
            {name: 'sum', view: sumView, type: 'f32', access: 'read_write'},
            {name: 'overflow', view: overflowView, type: 'u32', access: 'read_write'}
          ],
          invocationCount: 1,
          declarations: getPairHistogramReadWGSL('accumulators', 1, '', slotCount),
          body: `count[countOffset] = readPairCount(0u, 0u);
  sum[sumOffset] = readPairAccumulator(0u, 0u);
  overflow[overflowOffset] = select(0u, 1u, readPairOverflow());`
        })
      ];
    }
  });
  const compiled = graph.compile();
  try {
    parameterBuffer.write(new Float32Array([0, 0, 4, 4, 10]));
    submitGraph(device, compiled, undefined);
    return {
      count: (await readUint32(countBuffer, 1))[0],
      sum: (await readFloat32(sumBuffer, 1))[0],
      overflow: (await readUint32(overflowBuffer, 1))[0]
    };
  } finally {
    compiled.destroy();
    parameterBuffer.destroy();
    for (const buffer of buffers) {
      buffer.destroy();
    }
  }
}

it('pair histogram pair counts saturate at 0xffffffff when the 64-bit sum passes 2^32 (B-06)', async () => {
  // Six pairs each add 7 to the low and 3 to the high word: sum = 18 * 2^32 + 42.
  const result = await runForcedAccumulation('accumulateWide(0u, 0u, 7u, 3u);');
  if (!result) {
    return;
  }
  expect(result.count).toBe(0xffffffff);
  expect(result.sum).toBeCloseTo(18 * 2 ** 32 + 42, -5);
  expect(result.overflow).toBe(0);
  // Below 2^32 the count is exact.
  const exact = await runForcedAccumulation('accumulate(0u, 0u, 1u);');
  expect(exact?.count).toBe(6);
  expect(exact?.overflow).toBe(0);
});

it('pair histogram raises the overflow word when a 64-bit sum or a term exceeds its range', async () => {
  // Six pairs of 2^31 high-word units reach 3 * 2^32 words: the high word carries out of 64 bits.
  const wrapped = await runForcedAccumulation('accumulateWide(0u, 0u, 0u, 0x80000000u);');
  if (!wrapped) {
    return;
  }
  expect(wrapped.overflow).toBe(1);
  // A term beyond 2^75 fixed-point units is clamped and flagged.
  const clamped = await runForcedAccumulation('accumulateScaled(0u, 0u, 0u, 1.0e30);');
  expect(clamped?.overflow).toBe(1);
  // A large but representable term is added exactly in the coarse channel.
  const large = await runForcedAccumulation('accumulateScaled(0u, 0u, 0u, 4294967296.0 * 1024.0);');
  expect(large?.overflow).toBe(0);
  expect(large?.sum).toBeCloseTo(6 * 2 ** 30, -5);
});
