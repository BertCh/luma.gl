// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {GPULocalOutlierFactor} from '../../../src/gpu-spatial-analysis/outlier-detection/gpu-local-outlier-factor';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {describe, expect, it} from 'vitest';
import {
  getGPULocalOutlierFactorParameterValues,
  GPU_LOCAL_OUTLIER_FACTOR_PARAMETER_LENGTH
} from '../../../src/gpu-spatial-analysis/outlier-detection/index';

describe('getGPULocalOutlierFactorParameterValues', () => {
  it('packs threshold and density floor; an omitted threshold is NaN, not a hidden 1.5', () => {
    const values = getGPULocalOutlierFactorParameterValues();
    expect(values.length).toBe(GPU_LOCAL_OUTLIER_FACTOR_PARAMETER_LENGTH);
    expect(Number.isNaN(values[0])).toBe(true);
    expect(values[1]).toBe(Math.fround(1e-10));
    expect(
      Array.from(getGPULocalOutlierFactorParameterValues({threshold: 2, densityFloor: 0.5}))
    ).toEqual([2, 0.5, 0, 0]);
  });

  it('rejects a non-positive density floor and a NaN threshold', () => {
    expect(() => getGPULocalOutlierFactorParameterValues({densityFloor: 0})).toThrow();
    expect(() => getGPULocalOutlierFactorParameterValues({densityFloor: 1e-50})).toThrow();
    expect(() => getGPULocalOutlierFactorParameterValues({threshold: NaN})).toThrow();
  });
});

it('LOF checks the density floor context against the neighbor distances when both are declared', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `context-${serial++}`, format, length);
  const context = {
    coordinateSpace: 'longitude-latitude',
    metric: 'great-circle',
    units: 'meters'
  } as const;
  const props = {
    neighbors: {
      offsets: view('uint32', 4),
      neighbors: view('uint32', 6),
      weights: view('float32', 6),
      distances: view('float32', 6),
      spatialContext: context
    },
    parameters: view('float32', GPU_LOCAL_OUTLIER_FACTOR_PARAMETER_LENGTH),
    output: {
      kDistance: view('float32', 3),
      localReachabilityDensity: view('float32', 3),
      lof: view('float32', 3)
    }
  };
  expect(
    () =>
      new GPULocalOutlierFactor({
        ...props,
        spatialContext: {coordinateSpace: 'planar', metric: 'native'}
      })
  ).toThrow(/incompatible spatial contexts/);
  expect(
    () => new GPULocalOutlierFactor({...props, spatialContext: {...context, units: 'kilometers'}})
  ).toThrow(/incompatible spatial contexts/);
  expect(() => new GPULocalOutlierFactor({...props, spatialContext: {...context}})).not.toThrow();
  expect(() => new GPULocalOutlierFactor(props)).not.toThrow();
  expect(
    () =>
      new GPULocalOutlierFactor({
        ...props,
        neighbors: {...props.neighbors, spatialContext: undefined},
        spatialContext: context
      })
  ).not.toThrow();
  device.destroy();
});
