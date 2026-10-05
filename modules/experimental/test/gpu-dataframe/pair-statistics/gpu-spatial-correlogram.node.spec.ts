// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  GPUSpatialCorrelogram,
  type GPUSpatialCorrelogramProps
} from '../../../src/gpu-dataframe/pair-statistics/gpu-spatial-correlogram';
import {
  getGPUSpatialCorrelogramParameterValues,
  GPU_SPATIAL_CORRELOGRAM_PARAMETER_LENGTH
} from '../../../src/gpu-dataframe/pair-statistics/spatial-correlogram-parameters';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  computeSpatialCorrelogramOnCPU,
  getTwoSidedNormalPValue
} from './spatial-correlogram-oracle';

let serial = 0;

function createProps(
  graph: GPUCommandGraph,
  overrides: Partial<GPUSpatialCorrelogramProps> = {}
): GPUSpatialCorrelogramProps {
  const view = <Format extends 'uint32' | 'float32' | 'float32x2'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `view-${serial++}`, format, length);
  return {
    positions: view('float32x2', 10),
    values: view('float32', 10),
    parameters: view('float32', GPU_SPATIAL_CORRELOGRAM_PARAMETER_LENGTH),
    gridSize: [4, 4],
    bandCount: 6,
    moransI: view('float32', 6),
    ...overrides
  };
}

function expectThrows(
  overrides: (graph: GPUCommandGraph) => Partial<GPUSpatialCorrelogramProps>,
  message: RegExp
): void {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  expect(() => new GPUSpatialCorrelogram(createProps(graph, overrides(graph)))).toThrow(message);
  device.destroy();
}

it('getGPUSpatialCorrelogramParameterValues packs and validates the layout', () => {
  expect(
    Array.from(getGPUSpatialCorrelogramParameterValues({bounds: [0, 1, 2, 3], maximumDistance: 4}))
  ).toEqual([0, 1, 2, 3, 4, 1, 0, 0]);
  expect(
    Array.from(
      getGPUSpatialCorrelogramParameterValues({
        bounds: [0, 0, 1, 1],
        maximumDistance: 0.5,
        varianceAssumption: 'normality'
      })
    )
  ).toEqual([0, 0, 1, 1, 0.5, 0, 0, 0]);
  const bounds = [0, 0, 1, 1] as const;
  expect(() => getGPUSpatialCorrelogramParameterValues({bounds, maximumDistance: -1})).toThrow(
    /positive/
  );
  expect(() =>
    getGPUSpatialCorrelogramParameterValues({
      bounds,
      maximumDistance: 1,
      varianceAssumption: 'bootstrap' as never
    })
  ).toThrow(/varianceAssumption/);
  expect(() =>
    getGPUSpatialCorrelogramParameterValues({bounds, maximumDistance: 1}, new Float32Array(2))
  ).toThrow(/8/);
});

it('GPUSpatialCorrelogram validates its inputs and outputs', () => {
  const view = <Format extends 'float32' | 'uint32'>(
    graph: GPUCommandGraph,
    format: Format,
    length: number
  ) => createTransientView(graph, `bad-${serial++}`, format, length);
  expectThrows(() => ({bandCount: 0}), /bandCount/);
  expectThrows(graph => ({bandCount: 65, moransI: view(graph, 'float32', 65)}), /bandCount/);
  expectThrows(() => ({bandMode: 'ring' as never}), /bandMode/);
  expectThrows(graph => ({moransI: view(graph, 'float32', 5)}), /moransI/);
  expectThrows(graph => ({zScores: view(graph, 'float32', 5)}), /zScores/);
  expectThrows(graph => ({pairCounts: view(graph, 'uint32', 5)}), /pairCounts/);
  expectThrows(graph => ({peakBands: view(graph, 'uint32', 1)}), /peakBands/);
  expectThrows(graph => ({statistics: view(graph, 'float32', 4)}), /statistics/);
  expectThrows(graph => ({mask: view(graph, 'uint32', 4)}), /mask length/);
});

it('GPUSpatialCorrelogram emits deterministic node IDs with every output', () => {
  const device = createNullWebGPUDevice();
  const ids = () => {
    const graph = new GPUCommandGraph(device);
    const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
      createTransientView(graph, `out-${serial++}`, format, length);
    const contributor = new GPUSpatialCorrelogram({
      ...createProps(graph),
      id: 'c',
      zScores: view('float32', 6),
      pValues: view('float32', 6),
      expectedI: view('float32', 6),
      varianceI: view('float32', 6),
      pairCounts: view('uint32', 6),
      peakBands: view('uint32', 2),
      statistics: view('float32', 5)
    });
    return contributor.getCommandNodes(graph).map(node => node.id);
  };
  const first = ids();
  expect(ids()).toEqual(first);
  expect(first.every(id => id.startsWith('c-'))).toBe(true);
  for (const step of ['c-histogram-pairs', 'c-finish', 'c-publish', 'c-peaks', 'c-statistics']) {
    expect(first).toContain(step);
  }
  device.destroy();
});

it('computeSpatialCorrelogramOnCPU matches a hand-computed transect', () => {
  // Values 1..4 on a line: lag-1 neighbors only. mean 2.5, c = [-1.5, -0.5, 0.5, 1.5], M2 = 5.
  // sum_{i != j} w c_i c_j = 2 (0.75 - 0.25 + 0.75) = 2.5, S0 = 6, I = 4 / 6 * 2.5 / 5 = 1/3.
  // Normality: S1 = 12, k = [1, 2, 2, 1], S2 = 40,
  // Var = (16 * 12 - 4 * 40 + 3 * 36) / (15 * 36) - 1/9 = 4/27.
  const scene = {
    positions: new Float32Array([0, 0, 1, 0, 2, 0, 3, 0]),
    values: new Float32Array([1, 2, 3, 4])
  };
  const frame = {bounds: [0, -1, 3, 1] as const, maximumDistance: 1.5};
  const normality = computeSpatialCorrelogramOnCPU(
    scene,
    {...frame, varianceAssumption: 'normality'},
    1
  );
  expect(normality.moransI[0]).toBeCloseTo(1 / 3, 12);
  expect(normality.expectedI[0]).toBeCloseTo(-1 / 3, 12);
  expect(normality.varianceI[0]).toBeCloseTo(4 / 27, 12);
  expect(normality.pairCounts).toEqual([3]);
  expect(normality.zScores[0]).toBeCloseTo(2 / 3 / Math.sqrt(4 / 27), 12);
  // Randomization: b2 = 4 * (2 * 1.5^4 + 2 * 0.5^4) / 25 = 1.64.
  const randomization = computeSpatialCorrelogramOnCPU(scene, frame, 1);
  expect(randomization.statistics[4]).toBeCloseTo(1.64, 12);
  const n = 4;
  const numerator =
    n * ((n * n - 3 * n + 3) * 12 - n * 40 + 3 * 36) -
    1.64 * ((n * n - n) * 12 - 2 * n * 40 + 6 * 36);
  expect(randomization.varianceI[0]).toBeCloseTo(numerator / (3 * 2 * 1 * 36) - 1 / 9, 12);
});

it('getTwoSidedNormalPValue matches reference normal tail probabilities', () => {
  expect(getTwoSidedNormalPValue(0)).toBeCloseTo(1, 6);
  expect(getTwoSidedNormalPValue(1.959963984540054)).toBeCloseTo(0.05, 6);
  expect(getTwoSidedNormalPValue(-2.5758293035489004)).toBeCloseTo(0.01, 6);
});
