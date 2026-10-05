// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  GPUGroupStatistics,
  type GPUGroupStatistic,
  type GPUGroupStatisticsProps
} from '../../../src/gpu-dataframe/group-statistics';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  computeColumnStatistics,
  computeGroupStatisticsOnCPU,
  interpolateQuantile
} from './group-statistics-oracle';

let serial = 0;

function createProps(
  graph: GPUCommandGraph,
  overrides: Partial<GPUGroupStatisticsProps> = {},
  statistics: readonly GPUGroupStatistic[] = ['count', 'mean']
): GPUGroupStatisticsProps {
  const view = <Format extends 'uint32' | 'float32' | 'uint32x2'>(format: Format, length: number) =>
    createTransientView(graph, `view-${serial++}`, format, length);
  return {
    keys: view('uint32', 10),
    columns: [
      {
        values: view('float32', 10),
        statistics,
        output: {counts: view('uint32', 4), means: view('float32', 4)}
      }
    ],
    output: {
      keys: view('uint32', 4),
      counts: view('uint32', 4),
      count: view('uint32', 1),
      overflow: view('uint32', 1)
    },
    ...overrides
  };
}

function expectThrows(
  overrides: (graph: GPUCommandGraph) => Partial<GPUGroupStatisticsProps>,
  message: RegExp,
  statistics?: readonly GPUGroupStatistic[]
): void {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  expect(() => new GPUGroupStatistics(createProps(graph, overrides(graph), statistics))).toThrow(
    message
  );
  device.destroy();
}

it('oracle: textbook statistics of 1, 2, 3, 4', () => {
  const sample = computeColumnStatistics([1, 2, 3, 4], {
    variance: 'sample',
    fractions: [0, 0.25, 1],
    sumScale: 65536
  });
  expect(sample.count).toBe(4);
  expect(sample.sum).toBe(10n * 65536n);
  expect(sample.mean).toBe(2.5);
  expect(sample.variance).toBeCloseTo(5 / 3, 12);
  expect(sample.skewness).toBeCloseTo(0, 12);
  expect(sample.kurtosis).toBeCloseTo(-1.36, 12);
  expect(sample.median).toBe(2.5);
  expect(sample.percentiles).toEqual([1, 1.75, 4]);
  expect([sample.minimum, sample.maximum, sample.uniqueCount]).toEqual([1, 4, 4]);
  const population = computeColumnStatistics([1, 2, 3, 4], {
    variance: 'population',
    fractions: [],
    sumScale: 65536
  });
  expect(population.variance).toBeCloseTo(1.25, 12);
  expect(population.standardDeviation).toBeCloseTo(Math.sqrt(1.25), 12);
});

it('oracle: non-finite values, -0, ties, empty and single-value groups', () => {
  const options = {
    variance: 'sample' as const,
    fractions: [0.5],
    sumScale: 65536
  };
  // The mode of {3, 1, 1, 3, 2} ties between 1 and 3 and resolves to the smallest value.
  const tie = computeColumnStatistics([3, 1, NaN, 1, 3, Infinity, 2], options);
  expect([tie.count, tie.mode, tie.uniqueCount]).toEqual([5, 1, 3]);
  const zero = computeColumnStatistics([-0, 0, -0], options);
  expect(Object.is(zero.minimum, 0)).toBe(true);
  expect(zero.uniqueCount).toBe(1);
  expect(Object.is(zero.median, 0)).toBe(true);
  const empty = computeColumnStatistics([NaN, -Infinity], options);
  expect(empty.count).toBe(0);
  expect(empty.sum).toBe(0n);
  expect(Number.isNaN(empty.mean) && Number.isNaN(empty.median) && Number.isNaN(empty.mode)).toBe(
    true
  );
  const single = computeColumnStatistics([7], options);
  expect(Number.isNaN(single.variance)).toBe(true);
  expect(single.median).toBe(7);
  expect(Number.isNaN(single.skewness)).toBe(true);
  expect(interpolateQuantile([1, 2], 5)).toBe(2);
  expect(interpolateQuantile([1, 2], -1)).toBe(1);
  expect(Number.isNaN(interpolateQuantile([1, 2], NaN))).toBe(true);
});

it('oracle: keys, masks, reserved keys, capacity and z-scores', () => {
  const result = computeGroupStatisticsOnCPU({
    keys: [5n, 2n, 5n, 0xffffffffn, 2n, 9n, 5n],
    keyBits: 32,
    mask: Uint32Array.from([1, 1, 1, 1, 1, 0, 1]),
    columns: [Float32Array.from([1, 10, 3, 99, 20, 7, NaN])],
    variance: 'sample',
    fractions: [],
    sumScale: 65536,
    capacity: 1
  });
  expect(result.totalCount).toBe(2);
  expect(result.groups.map(group => [group.key, group.count])).toEqual([[2n, 2]]);
  expect(result.groups[0].columns[0].mean).toBe(15);
  // Group 2 has values 10 and 20: sd = sqrt(50), z = +-1/sqrt(2).
  expect(result.zScores[0][1]).toBeCloseTo(-Math.SQRT1_2, 12);
  expect(result.zScores[0][4]).toBeCloseTo(Math.SQRT1_2, 12);
  // Rows of the dropped group 5, the reserved key and the masked row stay NaN.
  expect([0, 2, 3, 5, 6].every(row => Number.isNaN(result.zScores[0][row]))).toBe(true);
  const wide = computeGroupStatisticsOnCPU({
    keys: [(1n << 32n) | 3n, 3n, 0xffffffffffffffffn],
    keyBits: 64,
    columns: [Float32Array.from([1, 2, 3])],
    variance: 'population',
    fractions: [],
    sumScale: 65536,
    capacity: 4
  });
  expect(wide.groups.map(group => group.key)).toEqual([3n, (1n << 32n) | 3n]);
});

it('GPUGroupStatistics validates props', () => {
  const view = <Format extends 'uint32' | 'float32' | 'uint32x2'>(
    graph: GPUCommandGraph,
    format: Format,
    length: number
  ) => createTransientView(graph, `v-${serial++}`, format, length);
  expectThrows(graph => ({keys: view(graph, 'float32', 10) as never}), /keys/);
  expectThrows(graph => ({mask: view(graph, 'uint32', 9)}), /mask must have the same length/);
  expectThrows(() => ({variance: 'bogus' as never}), /variance/);
  expectThrows(() => ({sumScale: 0}), /sumScale/);
  expectThrows(
    graph => ({
      output: {
        keys: view(graph, 'uint32x2', 4),
        counts: view(graph, 'uint32', 4),
        count: view(graph, 'uint32', 1),
        overflow: view(graph, 'uint32', 1)
      }
    }),
    /output.keys/
  );
  expectThrows(
    graph => ({
      output: {
        keys: view(graph, 'uint32', 4),
        counts: view(graph, 'uint32', 3),
        count: view(graph, 'uint32', 1),
        overflow: view(graph, 'uint32', 1)
      }
    }),
    /same length as output.keys/
  );
  expectThrows(
    graph => ({
      output: {
        keys: view(graph, 'uint32', 4),
        counts: view(graph, 'uint32', 4),
        count: view(graph, 'uint32', 0),
        overflow: view(graph, 'uint32', 1)
      }
    }),
    /one uint32 row/
  );
  expectThrows(
    graph => ({
      columns: [
        {
          values: view(graph, 'float32', 9),
          statistics: ['count'],
          output: {counts: view(graph, 'uint32', 4)}
        }
      ]
    }),
    /values must have the same length/
  );
  expectThrows(
    graph => ({
      columns: [
        {
          values: view(graph, 'float32', 10),
          statistics: ['mean'],
          output: {counts: view(graph, 'uint32', 4)}
        }
      ]
    }),
    /output.counts requires statistic 'count'|statistic 'mean' needs output.means/
  );
  expectThrows(
    graph => ({
      columns: [
        {
          values: view(graph, 'float32', 10),
          statistics: ['median'],
          output: {}
        }
      ]
    }),
    /statistic 'median' needs output.medians/
  );
  expectThrows(
    graph => ({
      columns: [
        {
          values: view(graph, 'float32', 10),
          statistics: ['zScore'],
          output: {zScores: view(graph, 'float32', 4)}
        }
      ]
    }),
    /zScores must have the same length as keys/
  );
  expectThrows(
    graph => ({
      columns: [
        {
          values: view(graph, 'float32', 10),
          statistics: ['percentiles'],
          output: {percentiles: view(graph, 'float32', 8)}
        }
      ]
    }),
    /needs the percentiles view/
  );
  expectThrows(
    graph => ({
      percentiles: view(graph, 'float32', 3),
      columns: [
        {
          values: view(graph, 'float32', 10),
          statistics: ['percentiles'],
          output: {percentiles: view(graph, 'float32', 8)}
        }
      ]
    }),
    /capacity|output.keys.length \* percentiles.length/
  );
  expectThrows(graph => ({percentiles: view(graph, 'float32', 3)}), /only used by/);
  expectThrows(
    graph => ({
      columns: Array.from({length: 5}, () => ({
        values: view(graph, 'float32', 10),
        statistics: ['count'] as const,
        output: {counts: view(graph, 'uint32', 4)}
      }))
    }),
    /at most 4 columns/
  );
  // Outputs must not alias inputs.
  expectThrows(graph => {
    const keys = view(graph, 'uint32', 10);
    return {
      keys,
      output: {
        keys: view(graph, 'uint32', 4),
        counts: view(graph, 'uint32', 4),
        count: keys,
        overflow: view(graph, 'uint32', 1)
      }
    };
  }, /must not share buffers/);
});

it('GPUGroupStatistics emits deterministic nodes for every statistic', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const view = <Format extends 'uint32' | 'float32' | 'uint32x2'>(format: Format, length: number) =>
    createTransientView(graph, `all-${serial++}`, format, length);
  const props: GPUGroupStatisticsProps = {
    id: 'stats',
    keys: view('uint32x2', 12),
    mask: view('uint32', 12),
    percentiles: view('float32', 3),
    columns: [
      {
        values: view('float32', 12),
        statistics: [
          'count',
          'sum',
          'mean',
          'minimum',
          'maximum',
          'variance',
          'standardDeviation',
          'skewness',
          'kurtosis',
          'median',
          'percentiles',
          'mode',
          'uniqueCount',
          'zScore'
        ],
        output: {
          counts: view('uint32', 5),
          sums: view('uint32x2', 5),
          sumValues: view('float32', 5),
          means: view('float32', 5),
          minimums: view('float32', 5),
          maximums: view('float32', 5),
          variances: view('float32', 5),
          standardDeviations: view('float32', 5),
          skewness: view('float32', 5),
          kurtosis: view('float32', 5),
          medians: view('float32', 5),
          percentiles: view('float32', 15),
          modes: view('float32', 5),
          uniqueCounts: view('uint32', 5),
          zScores: view('float32', 12)
        }
      },
      {
        values: view('float32', 12),
        statistics: ['mode'],
        output: {modes: view('float32', 5)}
      }
    ],
    output: {
      keys: view('uint32x2', 5),
      counts: view('uint32', 5),
      count: view('uint32', 1),
      overflow: view('uint32', 1),
      totalCount: view('uint32', 1)
    }
  };
  const contributor = new GPUGroupStatistics(props);
  const nodes = contributor.getCommandNodes(graph);
  const ids = nodes.map(node => node.id);
  expect(new Set(ids).size).toBe(ids.length);
  expect(ids.every(id => id.startsWith('stats-'))).toBe(true);
  for (const step of [
    'stats-prepare',
    'stats-publish',
    'stats-c0-moments',
    'stats-c0-z-scores',
    'stats-c0-medians',
    'stats-c0-percentiles',
    'stats-c0-mode-runs',
    'stats-c0-finish-unique',
    'stats-c1-finish-modes'
  ]) {
    expect(ids, step).toContain(step);
  }
  // 64-bit keys use two key sorts per chain.
  expect(ids.filter(id => id.includes('-order-sort-high')).length).toBeGreaterThan(0);
  // A second call on a second graph with the same props gives the same IDs.
  const graph2 = new GPUCommandGraph(device);
  const props2 = createProps(graph2, {id: 'stats'}, ['count', 'mean']);
  const first = new GPUGroupStatistics(props2).getCommandNodes(graph2).map(node => node.id);
  const graph3 = new GPUCommandGraph(device);
  const props3 = createProps(graph3, {id: 'stats'}, ['count', 'mean']);
  const second = new GPUGroupStatistics(props3).getCommandNodes(graph3).map(node => node.id);
  expect(second).toEqual(first);
  // Order statistics are skipped when no column asks for them.
  expect(first.some(id => id.includes('sort-value'))).toBe(false);
  device.destroy();
});
