// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  GPU_RASTER_ZONAL_STATISTICS_NO_ZONE,
  GPURasterZonalStatistics,
  type GPURasterZonalStatisticsProps
} from '../../../src/gpu-raster/raster-zonal-statistics';
import {getRasterZonalCellArea} from '../../../src/gpu-raster/raster-zonal-statistics/gpu-raster-zonal-statistics';
import {computeZonalStatistics} from './raster-zonal-statistics-oracle';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

const WIDTH = 4;
const HEIGHT = 3;
const CAPACITY = 5;

function createProps(
  graph: GPUCommandGraph,
  overrides: Partial<GPURasterZonalStatisticsProps> = {}
): GPURasterZonalStatisticsProps {
  return {
    id: 'zonal',
    width: WIDTH,
    height: HEIGHT,
    zones: createTransientView(graph, 'zones', 'uint32', WIDTH * HEIGHT),
    values: {
      id: 'values',
      format: 'float32',
      storage: {
        kind: 'buffer',
        values: createTransientView(graph, 'values', 'float32', WIDTH * HEIGHT)
      }
    },
    zoneCapacity: CAPACITY,
    output: {means: createTransientView(graph, 'means', 'float32', CAPACITY)},
    ...overrides
  };
}

function getNodeIds(props: GPURasterZonalStatisticsProps, graph: GPUCommandGraph): string[] {
  return new GPURasterZonalStatistics(props).getCommandNodes(graph).map(node => node.id);
}

it('GPURasterZonalStatistics prefixes IDs and schedules only requested columns', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  expect(GPU_RASTER_ZONAL_STATISTICS_NO_ZONE).toBe(0xffffffff);
  const meanIds = getNodeIds(createProps(graph), graph);
  expect(meanIds.every(id => id.startsWith('zonal-'))).toBe(true);
  expect(meanIds).toContain('zonal-mask');
  expect(meanIds.some(id => id.startsWith('zonal-means'))).toBe(true);
  expect(meanIds.some(id => id.startsWith('zonal-overflow'))).toBe(false);

  // cellCounts alone needs no band canonicalization and no value mask.
  const countGraph = new GPUCommandGraph(createNullWebGPUDevice());
  const countIds = getNodeIds(
    createProps(countGraph, {
      output: {
        cellCounts: createTransientView(countGraph, 'cell-counts', 'uint32', CAPACITY)
      }
    }),
    countGraph
  );
  expect(countIds.every(id => id.startsWith('zonal-'))).toBe(true);
  expect(countIds.some(id => id.includes('values-elevation'))).toBe(false);
  expect(countIds.some(id => id.startsWith('zonal-means'))).toBe(false);
  expect(countIds.length).toBeLessThan(meanIds.length);

  const fullGraph = new GPUCommandGraph(createNullWebGPUDevice());
  const fullIds = getNodeIds(
    createProps(fullGraph, {
      output: {
        cellCounts: createTransientView(fullGraph, 'o0', 'uint32', CAPACITY),
        valueCounts: createTransientView(fullGraph, 'o1', 'uint32', CAPACITY),
        sums: createTransientView(fullGraph, 'o2', 'float32', CAPACITY),
        means: createTransientView(fullGraph, 'o3', 'float32', CAPACITY),
        minimums: createTransientView(fullGraph, 'o4', 'float32', CAPACITY),
        maximums: createTransientView(fullGraph, 'o5', 'float32', CAPACITY)
      },
      overflow: createTransientView(fullGraph, 'overflow', 'uint32', 1)
    }),
    fullGraph
  );
  expect(fullIds).toContain('zonal-overflow-reset');
  // Counts, minimums and maximums share one workgroup-private pass (no per-column aggregation).
  expect(fullIds).toContain('zonal-zone-reduce');
  expect(fullIds).toContain('zonal-zone-reduce-decode');
  for (const name of ['cellCounts', 'valueCounts', 'minimums', 'maximums']) {
    expect(fullIds.some(id => id.startsWith(`zonal-${name}`))).toBe(false);
  }
  // The default sorted order computes sums in the shared `zonal-sorted-*` nodes.
  expect(fullIds.some(id => id.startsWith('zonal-sorted-'))).toBe(true);
  expect(fullIds).toContain('zonal-means');
});

function expectConstructorToThrow(
  createOverrides: (graph: GPUCommandGraph) => Partial<GPURasterZonalStatisticsProps>,
  pattern?: RegExp
): void {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  expect(() => new GPURasterZonalStatistics(createProps(graph, createOverrides(graph)))).toThrow(
    pattern
  );
}

it('GPURasterZonalStatistics validates its props', () => {
  expectConstructorToThrow(() => ({width: 0}), /dimensions/);
  expectConstructorToThrow(
    graph => ({
      zones: createTransientView(graph, 'short-zones', 'uint32', 3)
    }),
    /zones/
  );
  expectConstructorToThrow(() => ({output: {}}), /at least one output/);
  expectConstructorToThrow(() => ({zoneCapacity: 0}), /zoneCapacity/);
  expectConstructorToThrow(() => ({zoneCapacity: 2 ** 32}), /zoneCapacity/);
  expectConstructorToThrow(() => ({ignoredZone: -1}), /ignoredZone/);
  expectConstructorToThrow(() => ({ignoredZone: 1.5}), /ignoredZone/);
  expectConstructorToThrow(
    graph => ({
      output: {
        means: createTransientView(graph, 'wrong-length', 'float32', CAPACITY + 1)
      }
    }),
    /zoneCapacity rows/
  );
  expectConstructorToThrow(graph => ({
    output: {
      means: createTransientView(graph, 'wrong-format', 'uint32', CAPACITY) as never
    }
  }));
  expectConstructorToThrow(
    graph => ({
      output: {sums: createTransientView(graph, 'sums', 'float32', CAPACITY)},
      overflow: createTransientView(graph, 'overflow-two', 'uint32', 2)
    }),
    /overflow/
  );
});

it('GPURasterZonalStatistics rejects aliasing and foreign graphs', () => {
  expectConstructorToThrow(graph => {
    const shared = createTransientView(graph, 'shared', 'float32', CAPACITY);
    return {output: {sums: shared, means: shared}};
  }, /share buffers/);
  expectConstructorToThrow(graph => {
    const zones = createTransientView(graph, 'aliased-zones', 'uint32', WIDTH * HEIGHT);
    return {
      zones,
      zoneCapacity: WIDTH * HEIGHT,
      output: {cellCounts: zones}
    };
  }, /share buffers/);
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const otherGraph = new GPUCommandGraph(createNullWebGPUDevice());
  expect(() =>
    new GPURasterZonalStatistics(createProps(graph)).getCommandNodes(otherGraph)
  ).toThrow(/target graph/);
});

it('GPURasterZonalStatistics sumOrder defaults to sorted and atomic skips the sort nodes', () => {
  const outputs = (graph: GPUCommandGraph) => ({
    sums: createTransientView(graph, 'o-sums', 'float32', CAPACITY),
    means: createTransientView(graph, 'o-means', 'float32', CAPACITY)
  });
  const defaultGraph = new GPUCommandGraph(createNullWebGPUDevice());
  const defaultIds = getNodeIds(
    createProps(defaultGraph, {output: outputs(defaultGraph)}),
    defaultGraph
  );
  const sortedGraph = new GPUCommandGraph(createNullWebGPUDevice());
  const sortedIds = getNodeIds(
    createProps(sortedGraph, {
      output: outputs(sortedGraph),
      sumOrder: 'sorted'
    }),
    sortedGraph
  );
  expect(sortedIds).toEqual(defaultIds);
  expect(defaultIds.some(id => id.startsWith('zonal-sorted-sort'))).toBe(true);
  expect(defaultIds).toContain('zonal-sorted-gather-sums');
  expect(defaultIds).toContain('zonal-means');

  const atomicGraph = new GPUCommandGraph(createNullWebGPUDevice());
  const atomicIds = getNodeIds(
    createProps(atomicGraph, {
      output: outputs(atomicGraph),
      sumOrder: 'atomic'
    }),
    atomicGraph
  );
  expect(atomicIds.some(id => id.includes('-sorted-'))).toBe(false);
  expect(atomicIds.length).toBeLessThan(defaultIds.length);

  // Minimum-only requests never need the sort even when sorted is requested.
  const minGraph = new GPUCommandGraph(createNullWebGPUDevice());
  const minIds = getNodeIds(
    createProps(minGraph, {
      output: {
        minimums: createTransientView(minGraph, 'o-min', 'float32', CAPACITY)
      },
      sumOrder: 'sorted'
    }),
    minGraph
  );
  expect(minIds.some(id => id.includes('-sorted-'))).toBe(false);

  expectConstructorToThrow(
    () => ({sumOrder: 'fast' as never}),
    /sumOrder must be atomic or sorted/
  );
});

// Geographic fixture: 2 x 3 cells of 10 degrees longitude, rows 60-40, 40-20 and 20-0 N.
// Expected values from python3 -I (closed form, numpy not installed): cell area
// R^2 dLon |sin(top) - sin(bottom)| with the WGS84 authalic R = 6371007.180918475 m; weighted
// variance sum w (v - m)^2 / (V1 - V2 / V1), the numpy.cov(v, aweights=w, ddof=1) formula.
const GEOGRAPHIC_SETTINGS = [10, 20, 60, 0];
const GEOGRAPHIC_ZONES = [0, 0, 0, 1, 1, 1];
const GEOGRAPHIC_VALUES = [1, 2, 3, 4, 5, 6];
const GEOGRAPHIC_ROW_AREAS = [1581471169984.1543, 2130710343385.227, 2422954403438.46];

it('getRasterZonalCellArea matches closed-form spherical row areas', () => {
  for (const [row, area] of GEOGRAPHIC_ROW_AREAS.entries()) {
    expect(getRasterZonalCellArea(GEOGRAPHIC_SETTINGS, row, 3) / area).toBeCloseTo(1, 12);
    // Rows advancing north give the same areas.
    expect(getRasterZonalCellArea([10, 20, 0, 60], 2 - row, 3) / area).toBeCloseTo(1, 12);
  }
});

it('the zonal oracle reproduces python area-weighted means and reliability variances', () => {
  const cellWeights = GEOGRAPHIC_ZONES.map((_, cell) =>
    getRasterZonalCellArea(GEOGRAPHIC_SETTINGS, Math.floor(cell / 2), 3)
  );
  const weighted = computeZonalStatistics(GEOGRAPHIC_ZONES, GEOGRAPHIC_VALUES, {
    zoneCapacity: 2,
    cellWeights
  });
  expect(weighted.valueCounts).toEqual([3, 3]);
  const expected = [
    {weightSum: 5293652683353.535, sum: 11136544540108.145, mean: 2.1037542895717767},
    {weightSum: 6976619150262.146, sum: 35175339811363.97, mean: 5.041889066001582}
  ];
  const variances = [1.047000628016599, 0.9781397620488637];
  const deviations = [1.0232304862623078, 0.9890094853179435];
  for (const zone of [0, 1]) {
    expect(weighted.weightSums[zone] / expected[zone].weightSum).toBeCloseTo(1, 12);
    expect(weighted.sums[zone] / expected[zone].sum).toBeCloseTo(1, 12);
    expect(weighted.means[zone]).toBeCloseTo(expected[zone].mean, 12);
    expect(weighted.variances[zone]).toBeCloseTo(variances[zone], 12);
    expect(weighted.standardDeviations[zone]).toBeCloseTo(deviations[zone], 12);
  }
  // Uniform cells: plain mean and sum (v - mean)^2 / (n - 1) = 1 for 1,2,3 and 4,5,6.
  const uniform = computeZonalStatistics(GEOGRAPHIC_ZONES, GEOGRAPHIC_VALUES, {zoneCapacity: 2});
  expect(uniform.means).toEqual([2, 5]);
  expect(uniform.variances).toEqual([1, 1]);
  expect(uniform.weightSums).toEqual([3, 3]);
});

it('GPURasterZonalStatistics schedules weighted moments only when requested', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const settings = createTransientView(graph, 'settings', 'float32', 4);
  const ids = getNodeIds(
    createProps(graph, {
      cellSizeMode: 'geographic',
      settings,
      output: {
        means: createTransientView(graph, 'o-means', 'float32', CAPACITY),
        weightSums: createTransientView(graph, 'o-weights', 'float32', CAPACITY),
        variances: createTransientView(graph, 'o-variances', 'float32', CAPACITY),
        standardDeviations: createTransientView(graph, 'o-deviations', 'float32', CAPACITY)
      }
    }),
    graph
  );
  expect(ids).toContain('zonal-means');
  expect(ids).toContain('zonal-deviations');
  expect(ids).toContain('zonal-variances');
  expect(ids).toContain('zonal-sorted-gather-weights');
  expect(ids).toContain('zonal-variance-sorted-gather-weight-squares');

  // Atomic order aggregates the same contributions without the sort.
  const atomicGraph = new GPUCommandGraph(createNullWebGPUDevice());
  const atomicIds = getNodeIds(
    createProps(atomicGraph, {
      sumOrder: 'atomic',
      output: {
        standardDeviations: createTransientView(atomicGraph, 'o-deviations', 'float32', CAPACITY)
      }
    }),
    atomicGraph
  );
  expect(atomicIds.some(id => id.includes('-sorted-'))).toBe(false);
  expect(atomicIds).toContain('zonal-variances');

  // Planar requests without moments keep the previous node set.
  const planarGraph = new GPUCommandGraph(createNullWebGPUDevice());
  const planarIds = getNodeIds(createProps(planarGraph), planarGraph);
  expect(planarIds.some(id => id.includes('deviations') || id.includes('weight'))).toBe(false);
});

it('GPURasterZonalStatistics validates geographic cell weighting', () => {
  expectConstructorToThrow(() => ({cellSizeMode: 'geographic'}), /need settings/);
  expectConstructorToThrow(
    () => ({cellSizeMode: 'web-mercator' as never}),
    /cellSizeMode must be uniform or geographic/
  );
  expectConstructorToThrow(
    graph => ({settings: createTransientView(graph, 'settings', 'float32', 4)}),
    /settings need cellSizeMode geographic/
  );
  expectConstructorToThrow(
    graph => ({
      cellSizeMode: 'geographic',
      settings: createTransientView(graph, 'short-settings', 'float32', 3)
    }),
    /settings must contain 4/
  );
  expectConstructorToThrow(
    graph => ({
      output: {variances: createTransientView(graph, 'short-variances', 'float32', CAPACITY - 1)}
    }),
    /variances must contain zoneCapacity rows/
  );
});
