// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  getGPUFocalStatisticsParameterValues,
  getGPUInverseDistanceWeightingParameterValues,
  GPUFocalStatistics,
  GPUInverseDistanceWeighting,
  type GPUFocalStatisticsProps,
  type GPUInverseDistanceWeightingProps
} from '../../../src/geospatial/spatial-interpolation';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  computeFocalStatisticsOnCPU,
  interpolateInverseDistanceWeightingOnCPU
} from './spatial-interpolation-oracle';

let serial = 0;

function view<Format extends 'uint32' | 'float32' | 'float32x2'>(
  graph: GPUCommandGraph,
  format: Format,
  length: number
) {
  return createTransientView(graph, `view-${serial++}`, format, length);
}

function createInterpolationProps(
  graph: GPUCommandGraph,
  overrides: Partial<GPUInverseDistanceWeightingProps> = {}
): GPUInverseDistanceWeightingProps {
  return {
    positions: view(graph, 'float32x2', 10),
    values: view(graph, 'float32', 10),
    parameters: view(graph, 'float32', 8),
    width: 4,
    height: 3,
    indexGridSize: [2, 2],
    indexBounds: [0, 0, 1, 1],
    output: {
      values: view(graph, 'float32', 12),
      counts: view(graph, 'uint32', 12)
    },
    ...overrides
  };
}

function createFocalProps(
  graph: GPUCommandGraph,
  overrides: Partial<GPUFocalStatisticsProps> = {}
): GPUFocalStatisticsProps {
  return {
    values: view(graph, 'float32', 12),
    parameters: view(graph, 'float32', 4),
    width: 4,
    height: 3,
    maximumRadius: 2,
    output: {mean: view(graph, 'float32', 12)},
    ...overrides
  };
}

function withGraph(callback: (graph: GPUCommandGraph) => void): void {
  const device = createNullWebGPUDevice();
  callback(new GPUCommandGraph(device));
  device.destroy();
}

it('spatial interpolation parameter helpers pack layouts', () => {
  expect(
    Array.from(
      getGPUInverseDistanceWeightingParameterValues({
        extent: [0, 1, 2, 3],
        searchRadius: 5,
        power: 3,
        neighborCount: 4,
        minimumNeighborCount: 2
      })
    )
  ).toEqual([0, 1, 2, 3, 5, 3, 4, 2]);
  expect(
    Array.from(
      getGPUInverseDistanceWeightingParameterValues({
        extent: [0, 0, 1, 1],
        searchRadius: 1
      })
    )
  ).toEqual([0, 0, 1, 1, 1, 2, 0, 1]);
  expect(() =>
    getGPUInverseDistanceWeightingParameterValues({
      extent: [0, 0, 1, 1],
      searchRadius: 1,
      neighborCount: 1.5
    })
  ).toThrow(/neighborCount/);
  expect(() =>
    getGPUInverseDistanceWeightingParameterValues(
      {extent: [0, 0, 1, 1], searchRadius: 1},
      new Float32Array(4)
    )
  ).toThrow(/hold/);
  expect(
    Array.from(
      getGPUFocalStatisticsParameterValues({
        radius: 2.5,
        shape: 'circle',
        minimumCount: 3,
        propagateCenterNoData: true
      })
    )
  ).toEqual([2.5, 1, 3, 1]);
  expect(Array.from(getGPUFocalStatisticsParameterValues({radius: 1}))).toEqual([1, 0, 1, 0]);
  expect(() => getGPUFocalStatisticsParameterValues({radius: 1, minimumCount: -1})).toThrow(
    /minimumCount/
  );
});

it('GPUInverseDistanceWeighting validates its props', () => {
  const cases: [(graph: GPUCommandGraph) => Partial<GPUInverseDistanceWeightingProps>, RegExp][] = [
    [() => ({width: 0}), /width/],
    [() => ({indexGridSize: [2, 1.5]}), /indexGridSize/],
    [() => ({indexBounds: [0, 0, -1, 1]}), /indexBounds/],
    [() => ({indexBounds: [0, 0, Infinity, 1]}), /indexBounds/],
    [() => ({maximumNeighborCount: 65}), /maximumNeighborCount/],
    [graph => ({positions: view(graph, 'float32x2', 9)}), /positions length/],
    [graph => ({mask: view(graph, 'uint32', 9)}), /mask length/],
    [graph => ({parameters: view(graph, 'float32', 7)}), /parameters/],
    [graph => ({output: {values: view(graph, 'float32', 11)}}), /output.values/],
    [
      graph => ({
        output: {
          values: view(graph, 'float32', 12),
          counts: view(graph, 'uint32', 2)
        }
      }),
      /output.counts/
    ],
    [
      graph => ({
        values: view(graph, 'float32', 0),
        positions: view(graph, 'float32x2', 0)
      }),
      /at least one/
    ]
  ];
  for (const [overrides, message] of cases) {
    withGraph(graph => {
      expect(
        () => new GPUInverseDistanceWeighting(createInterpolationProps(graph, overrides(graph)))
      ).toThrow(message);
    });
  }
  withGraph(graph => {
    const values = view(graph, 'float32', 12);
    expect(
      () =>
        new GPUInverseDistanceWeighting(
          createInterpolationProps(graph, {
            values: view(graph, 'float32', 12),
            positions: view(graph, 'float32x2', 12),
            output: {values},
            parameters: values
          })
        )
    ).toThrow(/outputs must not share/);
  });
});

it('GPUInverseDistanceWeighting returns deterministic node IDs', () => {
  for (const maximumNeighborCount of [0, 8]) {
    const ids: string[][] = [];
    for (let run = 0; run < 2; run++) {
      withGraph(graph => {
        const recipe = new GPUInverseDistanceWeighting(
          createInterpolationProps(graph, {id: 'idw', maximumNeighborCount})
        );
        ids.push(recipe.getCommandNodes(graph).map(node => node.id));
      });
    }
    expect(ids[0]).toEqual(ids[1]);
    expect(ids[0].every(id => id.startsWith('idw-'))).toBe(true);
    expect(ids[0].slice(-2)).toEqual(['idw-sort-cells', 'idw-gather']);
  }
});

it('GPUFocalStatistics validates its props and schedules only requested kernels', () => {
  const cases: [(graph: GPUCommandGraph) => Partial<GPUFocalStatisticsProps>, RegExp][] = [
    [() => ({height: -1}), /height/],
    [() => ({maximumRadius: 65}), /maximumRadius/],
    [() => ({maximumRadius: 1.5}), /maximumRadius/],
    [() => ({noDataValue: NaN}), /noDataValue/],
    [() => ({output: {}}), /at least one output/],
    [graph => ({values: view(graph, 'float32', 11)}), /values must hold/],
    [graph => ({validity: view(graph, 'uint32', 11)}), /validity must hold/],
    [graph => ({parameters: view(graph, 'float32', 3)}), /parameters/],
    [graph => ({output: {count: view(graph, 'float32', 12) as never}}), /output.count/],
    [graph => ({output: {min: view(graph, 'float32', 3)}}), /output.min/]
  ];
  for (const [overrides, message] of cases) {
    withGraph(graph => {
      expect(() => new GPUFocalStatistics(createFocalProps(graph, overrides(graph)))).toThrow(
        message
      );
    });
  }
  withGraph(graph => {
    const shared = view(graph, 'float32', 12);
    expect(
      () => new GPUFocalStatistics(createFocalProps(graph, {output: {mean: shared, sum: shared}}))
    ).toThrow(/share buffers/);
  });
  withGraph(graph => {
    const moments = new GPUFocalStatistics(createFocalProps(graph, {id: 'focal'}));
    expect(moments.getCommandNodes(graph).map(node => node.id)).toEqual(['focal-moments']);
    const extremes = new GPUFocalStatistics(
      createFocalProps(graph, {
        id: 'focal2',
        output: {range: view(graph, 'float32', 12)}
      })
    );
    expect(extremes.getCommandNodes(graph).map(node => node.id)).toEqual(['focal2-extremes']);
    const both = new GPUFocalStatistics(
      createFocalProps(graph, {
        id: 'focal3',
        output: {
          mean: view(graph, 'float32', 12),
          sum: view(graph, 'float32', 12),
          min: view(graph, 'float32', 12),
          max: view(graph, 'float32', 12),
          range: view(graph, 'float32', 12),
          standardDeviation: view(graph, 'float32', 12),
          count: view(graph, 'uint32', 12)
        }
      })
    );
    expect(both.getCommandNodes(graph).map(node => node.id)).toEqual([
      'focal3-moments',
      'focal3-extremes'
    ]);
  });
});

it('interpolateInverseDistanceWeightingOnCPU weights, limits, and resolves exact hits', () => {
  // Two samples at distance 1 and 3 from the single cell center (0.5, 0.5).
  const base = {
    positions: Float32Array.from([1.5, 0.5, 0.5, 3.5, 0.5, 0.5, 0.5, 0.5]),
    values: Float32Array.from([10, 20, 30, 40]),
    width: 1,
    height: 1,
    indexBounds: [0, 0, 4, 4] as const,
    extent: [0, 0, 1, 1] as const,
    searchRadius: 5,
    power: 2,
    neighborCount: 0,
    minimumNeighborCount: 1
  };
  // Rows 2 and 3 are exact hits: the smaller row wins and all four samples count.
  let result = interpolateInverseDistanceWeightingOnCPU(base);
  expect(result.values[0]).toBe(30);
  expect(result.counts[0]).toBe(4);
  // Without exact hits: (10 * 1 + 20 / 9) / (1 + 1 / 9) = 11.
  const mask = [1, 1, 0, 0];
  result = interpolateInverseDistanceWeightingOnCPU({...base, mask});
  expect(result.values[0]).toBeCloseTo(11, 5);
  // Power zero is the plain mean; k = 1 keeps only the nearest sample.
  expect(interpolateInverseDistanceWeightingOnCPU({...base, mask, power: 0}).values[0]).toBe(15);
  result = interpolateInverseDistanceWeightingOnCPU({
    ...base,
    mask,
    neighborCount: 1
  });
  expect([result.values[0], result.counts[0]]).toEqual([10, 1]);
  // Radius 2 excludes the far sample; minimumNeighborCount 2 then yields nodata.
  result = interpolateInverseDistanceWeightingOnCPU({
    ...base,
    mask,
    searchRadius: 2,
    minimumNeighborCount: 2
  });
  expect(Number.isNaN(result.values[0])).toBe(true);
  expect(result.counts[0]).toBe(1);
});

it('computeFocalStatisticsOnCPU clips edges and honors nodata', () => {
  const values = Float32Array.from([1, 2, 3, 4, NaN, 6, 7, 8, -9]);
  const result = computeFocalStatisticsOnCPU({
    values,
    noDataValue: -9,
    width: 3,
    height: 3,
    maximumRadius: 4,
    radius: 1,
    shape: 'square',
    minimumCount: 1,
    propagateCenterNoData: false
  });
  // Corner (0, 0) sees 1, 2, 4 (center 4 is NaN).
  expect([result.count[0], result.sum[0], result.min[0], result.max[0]]).toEqual([3, 7, 1, 4]);
  // Center sees 7 valid cells (NaN center and the -9 sentinel are skipped).
  expect(result.count[4]).toBe(7);
  expect(result.mean[4]).toBeCloseTo(31 / 7, 5);
  const propagated = computeFocalStatisticsOnCPU({
    values,
    width: 3,
    height: 3,
    maximumRadius: 4,
    radius: 1,
    shape: 'circle',
    minimumCount: 1,
    propagateCenterNoData: true
  });
  expect(Number.isNaN(propagated.mean[4])).toBe(true);
  // Circle radius 1 excludes diagonals: corner sees 1, 2, 4.
  expect(propagated.count[0]).toBe(3);
});
