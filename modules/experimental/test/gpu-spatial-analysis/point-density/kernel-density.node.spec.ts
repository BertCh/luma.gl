// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  getGPUKernelDensityParameterValues,
  GPUKernelDensity
} from '../../../src/gpu-spatial-analysis/point-density';
import {
  evaluateGPUKernelDensityKernel,
  getGPUKernelDensitySupportRadius,
  getGPUKernelDensityWindowMass,
  getScottBandwidth,
  getSilvermanBandwidth,
  type GPUKernelDensityKernel
} from '../../../src/gpu-spatial-analysis/point-density/kernel-density-math';
import {KERNEL_DENSITY_ORACLE} from './kernel-density-oracle-values';

const KERNELS: GPUKernelDensityKernel[] = ['gaussian', 'quartic', 'epanechnikov'];

it('kernels integrate to one over the plane (Silverman 1986, Table 3.1)', () => {
  for (const kernel of KERNELS) {
    for (const bandwidth of [0.5, 2]) {
      const radius = getGPUKernelDensitySupportRadius(kernel, bandwidth);
      const stepCount = 20000;
      const step = radius / stepCount;
      let mass = 0;
      for (let index = 0; index < stepCount; index++) {
        const distance = (index + 0.5) * step;
        mass +=
          evaluateGPUKernelDensityKernel(kernel, bandwidth, distance) *
          2 *
          Math.PI *
          distance *
          step;
      }
      // The gaussian truncation at 5 bandwidths loses exp(-12.5) = 3.7e-6 of the mass.
      expect(mass).toBeCloseTo(1, kernel === 'gaussian' ? 5 : 7);
    }
  }
});

it('window mass matches scipy.integrate.dblquad of the kernel over the window', () => {
  for (const testCase of KERNEL_DENSITY_ORACLE.massCases) {
    const actual = getGPUKernelDensityWindowMass(
      testCase.kernel,
      testCase.bandwidth,
      testCase.point as unknown as [number, number],
      testCase.window as unknown as [number, number, number, number]
    );
    // The gaussian erfc fit has 1.2e-7 relative error per factor; polynomial kernels are exact.
    const tolerance = testCase.kernel === 'gaussian' ? 5e-7 : 1e-9;
    expect(Math.abs(actual - testCase.mass), `${testCase.kernel} ${testCase.point}`).toBeLessThan(
      tolerance
    );
  }
});

it('bandwidth rules match scipy.stats.gaussian_kde', () => {
  const {points, weights, unweighted, weighted} = KERNEL_DENSITY_ORACLE.bandwidth;
  for (const [rule, getBandwidth] of [
    ['scott', getScottBandwidth],
    ['silverman', getSilvermanBandwidth]
  ] as const) {
    for (const [expected, bandwidth] of [
      [unweighted[rule], getBandwidth(points)],
      [weighted[rule], getBandwidth(points, weights)]
    ] as const) {
      expect(bandwidth.factor).toBeCloseTo(expected.factor, 12);
      expect(bandwidth.x).toBeCloseTo(expected.x, 12);
      expect(bandwidth.y).toBeCloseTo(expected.y, 12);
      expect(bandwidth.isotropic).toBeCloseTo(expected.isotropic, 12);
      expect(bandwidth.effectiveSampleSize).toBeCloseTo(expected.effectiveSampleSize, 10);
    }
  }
});

it('bandwidth rules skip non-finite rows and reject degenerate input', () => {
  const {points} = KERNEL_DENSITY_ORACLE.bandwidth;
  const withMissing = [...points, Number.NaN, 1, 2, Number.POSITIVE_INFINITY];
  expect(getScottBandwidth(withMissing).isotropic).toBeCloseTo(
    getScottBandwidth(points).isotropic,
    12
  );
  expect(() => getScottBandwidth([1, 2])).toThrow();
  expect(() => getScottBandwidth([1, 2, 3, 4], [1, 0])).toThrow();
});

it('GPUKernelDensity schedules index, sort, weights, reductions, and gather', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const ids = new GPUKernelDensity({
    positions: createTransientView(graph, 'positions', 'float32x2', 8),
    weights: createTransientView(graph, 'weights', 'float32', 8),
    parameters: createTransientView(graph, 'parameters', 'float32', 9),
    width: 4,
    height: 3,
    indexGridSize: [2, 2],
    indexBounds: [0, 0, 4, 3],
    kernel: 'quartic',
    edgeCorrection: 'diggle',
    normalization: 'probability',
    output: {
      values: createTransientView(graph, 'values', 'float32', 12),
      invalidCount: createTransientView(graph, 'invalid', 'uint32', 1)
    }
  })
    .getCommandNodes(graph)
    .map(node => node.id);
  expect(ids.some(id => id.startsWith('kernel-density-index'))).toBe(true);
  expect(ids).toContain('kernel-density-sort-cells');
  expect(ids).toContain('kernel-density-point-weights');
  expect(ids.some(id => id.startsWith('kernel-density-weight-sum'))).toBe(true);
  expect(ids.some(id => id.startsWith('kernel-density-invalid-count'))).toBe(true);
  expect(ids.at(-1)).toBe('kernel-density-gather');
  expect(new Set(ids).size).toBe(ids.length);
  device.destroy();
});

it('GPUKernelDensity rejects non-planar contexts and invalid options', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const base = {
    positions: createTransientView(graph, 'positions', 'float32x2', 8),
    parameters: createTransientView(graph, 'parameters', 'float32', 9),
    width: 4,
    height: 3,
    indexGridSize: [2, 2] as const,
    indexBounds: [0, 0, 4, 3] as const,
    output: {values: createTransientView(graph, 'values', 'float32', 12)}
  };
  expect(
    () =>
      new GPUKernelDensity({
        ...base,
        spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'great-circle'}
      })
  ).toThrow(/project positions to a planar frame/);
  expect(() => new GPUKernelDensity({...base, kernel: 'triangle' as never})).toThrow();
  expect(() => new GPUKernelDensity({...base, edgeCorrection: 'reflect' as never})).toThrow();
  expect(() => new GPUKernelDensity({...base, width: 0})).toThrow();
  expect(new GPUKernelDensity(base).edgeCorrection).toBe('uniform');
  expect(() => getGPUKernelDensityParameterValues({extent: [0, 0, 1, 1], bandwidth: 0})).toThrow();
  expect(
    Array.from(getGPUKernelDensityParameterValues({extent: [0, 0, 4, 2], bandwidth: 1}))
  ).toEqual([0, 0, 4, 2, 1, 0, 0, 4, 2]);
  device.destroy();
});
