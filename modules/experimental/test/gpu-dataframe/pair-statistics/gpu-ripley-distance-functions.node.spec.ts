// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  GPURipleyDistanceFunctions,
  type GPURipleyDistanceFunctionsProps
} from '../../../src/gpu-dataframe/pair-statistics/gpu-ripley-distance-functions';
import {
  getGPURipleyDistanceParameterValues,
  GPU_RIPLEY_DISTANCE_PARAMETER_LENGTH
} from '../../../src/gpu-dataframe/pair-statistics/ripley-distance-parameters';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

let serial = 0;

function createProps(
  graph: GPUCommandGraph,
  overrides: Partial<GPURipleyDistanceFunctionsProps> = {}
): GPURipleyDistanceFunctionsProps {
  const view = <Format extends 'uint32' | 'float32' | 'float32x2'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `view-${serial++}`, format, length);
  return {
    positions: view('float32x2', 10),
    parameters: view('float32', GPU_RIPLEY_DISTANCE_PARAMETER_LENGTH),
    gridSize: [4, 4],
    referenceGrid: [5, 5],
    radiusCount: 8,
    g: view('float32', 8),
    ...overrides
  };
}

it('getGPURipleyDistanceParameterValues packs and validates the layout', () => {
  expect(
    Array.from(getGPURipleyDistanceParameterValues({bounds: [0, 1, 10, 11], maximumDistance: 3}))
  ).toEqual([0, 1, 10, 11, 3, 1, 0, 0]);
  expect(
    getGPURipleyDistanceParameterValues({
      bounds: [0, 1, 10, 11],
      maximumDistance: 3,
      edgeCorrection: 'none'
    })[5]
  ).toBe(0);
  expect(
    ['kaplan-meier', 'hanisch'].map(
      edgeCorrection =>
        getGPURipleyDistanceParameterValues({
          bounds: [0, 1, 10, 11],
          maximumDistance: 3,
          edgeCorrection: edgeCorrection as 'hanisch'
        })[5]
    )
  ).toEqual([2, 3]);
  expect(() =>
    getGPURipleyDistanceParameterValues({
      bounds: [0, 0, 1, 1],
      maximumDistance: 1,
      edgeCorrection: 'isotropic' as 'none'
    })
  ).toThrow(/edgeCorrection/);
  expect(() =>
    getGPURipleyDistanceParameterValues({bounds: [0, 0, 1, 1], maximumDistance: 0})
  ).toThrow(/maximumDistance/);
});

it('GPURipleyDistanceFunctions validates props and declares nodes', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device, {id: 'ripley-distance-nodes'});
  expect(() => new GPURipleyDistanceFunctions(createProps(graph, {g: undefined}))).toThrow(
    /at least one/
  );
  expect(() => new GPURipleyDistanceFunctions(createProps(graph, {radiusCount: 0}))).toThrow(
    /radiusCount/
  );
  expect(() => new GPURipleyDistanceFunctions(createProps(graph, {referenceGrid: [0, 4]}))).toThrow(
    /referenceGrid/
  );
  const ids = new GPURipleyDistanceFunctions(createProps(graph))
    .getCommandNodes(graph)
    .map(node => node.id);
  expect(ids).toContain('ripley-distance-functions-f-reference');
  expect(ids.at(-1)).toBe('ripley-distance-functions-finish');
  device.destroy();
});
