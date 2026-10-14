// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  GPURipleyDistanceFunctions,
  type GPURipleyDistanceFunctionsOutput,
  type GPURipleyDistanceFunctionsProps
} from '../../../src/gpu-dataframe/pair-statistics/gpu-ripley-distance-functions';
import {
  getGPURipleyDistanceFunctionsParameterValues,
  GPU_RIPLEY_DISTANCE_FUNCTIONS_PARAMETER_LENGTH
} from '../../../src/gpu-dataframe/pair-statistics/ripley-distance-parameters';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

let serial = 0;

type Overrides = Omit<Partial<GPURipleyDistanceFunctionsProps>, 'output'> & {
  output?: Partial<GPURipleyDistanceFunctionsOutput>;
};

function createProps(
  graph: GPUCommandGraph,
  overrides: Overrides = {}
): GPURipleyDistanceFunctionsProps {
  const view = <Format extends 'uint32' | 'float32' | 'float32x2'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `view-${serial++}`, format, length);
  return {
    positions: view('float32x2', 10),
    parameters: view('float32', GPU_RIPLEY_DISTANCE_FUNCTIONS_PARAMETER_LENGTH),
    gridSize: [4, 4],
    referenceGrid: [5, 5],
    radiusCount: 8,
    ...overrides,
    output: {g: view('float32', 8), ...overrides.output}
  };
}

it('getGPURipleyDistanceFunctionsParameterValues packs and validates the layout', () => {
  expect(
    Array.from(
      getGPURipleyDistanceFunctionsParameterValues({bounds: [0, 1, 10, 11], maximumDistance: 3})
    )
  ).toEqual([0, 1, 10, 11, 3, 1, 0, 0]);
  expect(
    getGPURipleyDistanceFunctionsParameterValues({
      bounds: [0, 1, 10, 11],
      maximumDistance: 3,
      edgeCorrection: 'none'
    })[5]
  ).toBe(0);
  expect(
    ['kaplan-meier', 'hanisch'].map(
      edgeCorrection =>
        getGPURipleyDistanceFunctionsParameterValues({
          bounds: [0, 1, 10, 11],
          maximumDistance: 3,
          edgeCorrection: edgeCorrection as 'hanisch'
        })[5]
    )
  ).toEqual([2, 3]);
  expect(() =>
    getGPURipleyDistanceFunctionsParameterValues({
      bounds: [0, 0, 1, 1],
      maximumDistance: 1,
      edgeCorrection: 'isotropic' as 'none'
    })
  ).toThrow(/edgeCorrection/);
  expect(() =>
    getGPURipleyDistanceFunctionsParameterValues({bounds: [0, 0, 1, 1], maximumDistance: 0})
  ).toThrow(/maximumDistance/);
});

it('GPURipleyDistanceFunctions validates props and declares nodes', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device, {id: 'ripley-distance-nodes'});
  expect(
    () => new GPURipleyDistanceFunctions(createProps(graph, {output: {g: undefined}}))
  ).toThrow(/at least one/);
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

it('GPURipleyDistanceFunctions declares planar-only spatial support', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  expect(
    () =>
      new GPURipleyDistanceFunctions(
        createProps(graph, {
          spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'great-circle'} as const
        })
      )
  ).toThrow(/project positions to a planar frame first/);
  expect(new GPURipleyDistanceFunctions(createProps(graph)).spatialContext.coordinateSpace).toBe(
    'planar'
  );
  device.destroy();
});
