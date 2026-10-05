// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  GPUGeographicallyWeightedRegression,
  type GPUGeographicallyWeightedRegressionProps
} from '../../../src/geospatial/spatial-regression/gpu-geographically-weighted-regression';
import {
  getGPUGeographicallyWeightedRegressionParameterLength,
  getGPUGeographicallyWeightedRegressionParameterValues,
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_STATUS
} from '../../../src/geospatial/spatial-regression/geographically-weighted-regression-parameters';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  computeGeographicallyWeightedRegressionOnCPU,
  getGeographicallyWeightedRegressionWeight
} from './geographically-weighted-regression-oracle';

let serial = 0;
const ROWS = 10;

function createProps(
  graph: GPUCommandGraph,
  overrides: Partial<GPUGeographicallyWeightedRegressionProps> = {}
): GPUGeographicallyWeightedRegressionProps {
  const view = <Format extends 'uint32' | 'float32' | 'float32x2'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `view-${serial++}`, format, length);
  return {
    positions: view('float32x2', ROWS),
    predictors: view('float32', ROWS * 2),
    predictorCount: 2,
    response: view('float32', ROWS),
    parameters: view('float32', getGPUGeographicallyWeightedRegressionParameterLength()),
    output: {coefficients: view('float32', ROWS * 3)},
    ...overrides
  };
}

function expectThrows(
  overrides: (graph: GPUCommandGraph) => Partial<GPUGeographicallyWeightedRegressionProps>,
  message: RegExp
): void {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  expect(
    () => new GPUGeographicallyWeightedRegression(createProps(graph, overrides(graph)))
  ).toThrow(message);
  device.destroy();
}

it('getGPUGeographicallyWeightedRegressionParameterValues packs kernel, mode and ladder', () => {
  const values = getGPUGeographicallyWeightedRegressionParameterValues(
    {kernel: 'gaussian', bandwidthMode: 'adaptive', bandwidths: [10, 20, 40]},
    8
  );
  expect(values.length).toBe(12);
  expect(Array.from(values.subarray(0, 7))).toEqual([0, 1, 3, 0, 10, 20, 40]);
  expect(values[7]).toBe(0);
  const defaults = getGPUGeographicallyWeightedRegressionParameterValues({bandwidths: [2.5]}, 4);
  expect(Array.from(defaults.subarray(0, 5))).toEqual([1, 0, 1, 0, 2.5]);
  expect(() => getGPUGeographicallyWeightedRegressionParameterValues({bandwidths: []})).toThrow(
    /1 to 32/
  );
  expect(() =>
    getGPUGeographicallyWeightedRegressionParameterValues({bandwidths: [1, 2, 3]}, 2)
  ).toThrow(/1 to 2/);
  expect(() =>
    getGPUGeographicallyWeightedRegressionParameterValues({
      bandwidths: [Number.NaN]
    })
  ).toThrow(/finite/);
  expect(() =>
    getGPUGeographicallyWeightedRegressionParameterValues({bandwidths: [1]}, 8, new Float32Array(4))
  ).toThrow(/hold/);
});

it('GPUGeographicallyWeightedRegression validates props', () => {
  expectThrows(() => ({predictorCount: 0}), /predictorCount/);
  expectThrows(() => ({predictorCount: 8}), /predictorCount/);
  expectThrows(() => ({maximumBandwidthCount: 33}), /maximumBandwidthCount/);
  expectThrows(() => ({maximumNeighborCount: 129}), /maximumNeighborCount/);
  expectThrows(
    graph => ({
      predictors: createTransientView(graph, `p-${serial++}`, 'float32', 7)
    }),
    /predictors length/
  );
  expectThrows(
    graph => ({
      response: createTransientView(graph, `r-${serial++}`, 'float32', 9)
    }),
    /response length/
  );
  expectThrows(
    graph => ({
      mask: createTransientView(graph, `m-${serial++}`, 'uint32', 9)
    }),
    /mask length/
  );
  expectThrows(
    graph => ({
      parameters: createTransientView(graph, `q-${serial++}`, 'float32', 8)
    }),
    /parameters/
  );
  expectThrows(
    graph => ({
      output: {
        coefficients: createTransientView(graph, `c-${serial++}`, 'float32', ROWS * 2)
      }
    }),
    /coefficients/
  );
  expectThrows(
    graph => ({
      output: {
        coefficients: createTransientView(graph, `c-${serial++}`, 'float32', ROWS * 3),
        bandwidthScores: createTransientView(graph, `b-${serial++}`, 'float32', 4)
      }
    }),
    /bandwidthScores/
  );
  expectThrows(
    graph => ({
      output: {
        coefficients: createTransientView(graph, `c-${serial++}`, 'float32', ROWS * 3),
        localStatus: createTransientView(graph, `s-${serial++}`, 'float32', ROWS) as never
      }
    }),
    /localStatus/
  );
  expectThrows(
    graph => ({
      positions: createTransientView(graph, `pos-${serial++}`, 'float32x2', 65537),
      predictors: createTransientView(graph, `pr-${serial++}`, 'float32', 65537 * 2),
      response: createTransientView(graph, `re-${serial++}`, 'float32', 65537),
      output: {
        coefficients: createTransientView(graph, `co-${serial++}`, 'float32', 65537 * 3)
      }
    }),
    /at most 65536/
  );
});

it('GPUGeographicallyWeightedRegression emits deterministic node IDs within the binding limit', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
    createTransientView(graph, `out-${serial++}`, format, length);
  const minimal = new GPUGeographicallyWeightedRegression(createProps(graph, {id: 'min'}));
  const minimalIds = minimal.getCommandNodes(graph).map(node => node.id);
  expect(minimalIds).toEqual([
    'min-validate',
    'min-tile-response',
    'min-tile-total',
    'min-candidates',
    'min-tile-candidates',
    'min-select',
    'min-fit'
  ]);
  const full = new GPUGeographicallyWeightedRegression(
    createProps(graph, {
      id: 'full',
      mask: createTransientView(graph, `mask-${serial++}`, 'uint32', ROWS),
      output: {
        coefficients: view('float32', ROWS * 3),
        localR2: view('float32', ROWS),
        fitted: view('float32', ROWS),
        residuals: view('float32', ROWS),
        hatDiagonal: view('float32', ROWS),
        localStatus: view('uint32', ROWS),
        bandwidthScores: view('float32', 32),
        selectedBandwidth: view('float32', 2),
        summary: view('float32', 6)
      }
    })
  );
  const fullIds = full.getCommandNodes(graph).map(node => node.id);
  expect(fullIds).toEqual([...minimalIds.map(name => name.replace('min', 'full')), 'full-publish']);
  expect(new Set(fullIds).size).toBe(fullIds.length);
  device.destroy();
});

it('computeGeographicallyWeightedRegressionOnCPU recovers an exact plane and flags singular fits', () => {
  const side = 8;
  const rows = side * side;
  const positions = new Float32Array(rows * 2);
  const predictors = new Float32Array(rows);
  const response = new Float32Array(rows);
  for (let row = 0; row < rows; row++) {
    positions[2 * row] = row % side;
    positions[2 * row + 1] = Math.floor(row / side);
    predictors[row] = ((row * 7) % 5) - 2;
    response[row] = 3 + 2 * predictors[row];
  }
  const result = computeGeographicallyWeightedRegressionOnCPU({
    positions,
    predictors,
    predictorCount: 1,
    response,
    settings: {kernel: 'gaussian', bandwidths: [3, 5]}
  });
  expect(
    result.localStatus.every(status => status === GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_STATUS.OK)
  ).toBe(true);
  for (let row = 0; row < rows; row++) {
    expect(result.coefficients[2 * row]).toBeCloseTo(3, 6);
    expect(result.coefficients[2 * row + 1]).toBeCloseTo(2, 6);
    expect(result.residuals[row]).toBeCloseTo(0, 6);
  }
  // A bandwidth below the grid spacing leaves a single neighbour: every candidate is singular.
  const tiny = computeGeographicallyWeightedRegressionOnCPU({
    positions,
    predictors,
    predictorCount: 1,
    response,
    settings: {kernel: 'bisquare', bandwidths: [0.5]}
  });
  expect(tiny.hasValidCandidate).toBe(false);
  expect(tiny.localStatus[0]).toBe(GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_STATUS.SINGULAR);
  // Kernels: weight 1 at zero distance, bisquare zero at the bandwidth.
  expect(getGeographicallyWeightedRegressionWeight('gaussian', 0, 2)).toBe(1);
  expect(getGeographicallyWeightedRegressionWeight('bisquare', 2, 2)).toBe(0);
  expect(getGeographicallyWeightedRegressionWeight('bisquare', 1, 2)).toBeCloseTo(0.5625, 12);
});
