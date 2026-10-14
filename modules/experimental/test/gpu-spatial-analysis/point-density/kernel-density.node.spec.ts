// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {GPUStudyWindowPreparation} from '../../../src/gpu-spatial-analysis/study-window/gpu-study-window-preparation';
import {
  getGPUCommandNodeKernelSources,
  getStorageBindingCount
} from '../../utils/gpu-contributor-test-utils';
import {auditWGSLModule} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  getGPUKernelDensityParameterValues,
  GPUKernelDensity
} from '../../../src/gpu-spatial-analysis/point-density';
import {
  evaluateGPUKernelDensityKernel,
  getGPUKernelDensitySupportRadius,
  getGPUKernelDensityWindowMass,
  type GPUKernelDensityKernel
} from '../../../src/gpu-spatial-analysis/point-density/kernel-density-math';
import {makeExtentGrid} from '../spatial-interpolation/extent-grid';
import {getBandwidthOracle} from './kernel-density-bandwidth-oracle';
import {getPolygonKernelDensityOracle} from './point-density-oracle';
import {KERNEL_DENSITY_ORACLE} from './kernel-density-oracle-values';

const GRID_4_BY_3 = makeExtentGrid(4, 3, [0, 0, 4, 3]);
const SHEARED_GRID = {...GRID_4_BY_3, affine: [1, 0.5, 0, 0, 1, 0] as const};

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
  const getBandwidth = getBandwidthOracle;
  for (const rule of ['scott', 'silverman'] as const) {
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
  expect(getBandwidthOracle(withMissing).isotropic).toBeCloseTo(
    getBandwidthOracle(points).isotropic,
    12
  );
  expect(() => getBandwidthOracle([1, 2])).toThrow();
  expect(() => getBandwidthOracle([1, 2, 3, 4], [1, 0])).toThrow();
});

it('GPUKernelDensity schedules index, sort, weights, reductions, and gather', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const ids = new GPUKernelDensity({
    positions: createTransientView(graph, 'positions', 'float32x2', 8),
    weights: createTransientView(graph, 'weights', 'float32', 8),
    parameters: createTransientView(graph, 'parameters', 'float32', 9),
    grid: GRID_4_BY_3,
    indexGridSize: [2, 2],
    indexBounds: [0, 0, 4, 3],
    kernel: 'quartic',
    edgeCorrection: 'diggle',
    normalization: 'probability',
    output: {
      values: createTransientView(graph, 'values', 'float32', 12),
      status: {invalidCount: createTransientView(graph, 'invalid', 'uint32', 1)}
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
    grid: GRID_4_BY_3,
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
  expect(() => new GPUKernelDensity({...base, grid: {...GRID_4_BY_3, width: 0}})).toThrow(
    /grid dimensions/
  );
  expect(() => new GPUKernelDensity({...base, grid: SHEARED_GRID})).toThrow(/axis-aligned/);
  expect(new GPUKernelDensity(base).edgeCorrection).toBe('uniform');
  expect(() => getGPUKernelDensityParameterValues({bandwidth: 0}, GRID_4_BY_3)).toThrow();
  expect(
    Array.from(
      getGPUKernelDensityParameterValues({bandwidth: 1}, makeExtentGrid(4, 3, [0, 0, 4, 2]))
    )
  ).toEqual([0, 0, 4, 2, 1, 0, 0, 4, 2]);
  device.destroy();
});

it('getGPUKernelDensityParameterValues reads the corners and window from the grid', () => {
  const extent = [10, 20, 18, 26] as const;
  const [minX, minY, maxX, maxY] = extent;
  const north = makeExtentGrid(4, 3, extent);
  const northUp = makeExtentGrid(4, 3, extent, 'south');
  // North-advancing rows pack the old extent. The north-up grid of the same rectangle starts at
  // the top, but its default window is still the ordered rectangle.
  expect(Array.from(getGPUKernelDensityParameterValues({bandwidth: 1}, north))).toEqual(
    [minX, minY, maxX, maxY, 1, minX, minY, maxX, maxY].map(Math.fround)
  );
  expect(Array.from(getGPUKernelDensityParameterValues({bandwidth: 1}, northUp))).toEqual(
    [minX, maxY, maxX, minY, 1, minX, minY, maxX, maxY].map(Math.fround)
  );
  // Point pixels move the outer corner by half a cell (cell size 2 by 2 here).
  const pointPixels = {...north, pixelInterpretation: 'point' as const};
  expect(
    Array.from(getGPUKernelDensityParameterValues({bandwidth: 1}, pointPixels).slice(0, 4))
  ).toEqual([minX - 1, minY - 1, maxX - 1, maxY - 1].map(Math.fround));
  // An explicit window stays [minX, minY, maxX, maxY].
  expect(
    Array.from(
      getGPUKernelDensityParameterValues({bandwidth: 1, window: [11, 21, 17, 25]}, northUp).slice(5)
    )
  ).toEqual([11, 21, 17, 25]);
  expect(() => getGPUKernelDensityParameterValues({bandwidth: 1}, SHEARED_GRID)).toThrow(
    /axis-aligned/
  );
  expect(() =>
    getGPUKernelDensityParameterValues({bandwidth: 1}, {...north, affine: [0, 0, 0, 0, 1, 0]})
  ).toThrow(/nonzero cell size/);
  expect(() =>
    getGPUKernelDensityParameterValues({bandwidth: 1, window: [5, 5, 4, 6]}, north)
  ).toThrow(/window/);
});

const DISTANCE_FRAME = {
  coordinateSpace: 'planar',
  metric: 'native',
  units: 'meters',
  frame: {id: 'f', kind: 'local-azimuthal-equidistant'}
} as const;
const AREA_FRAME = {
  coordinateSpace: 'planar',
  metric: 'native',
  units: 'meters',
  frame: {id: 'f', kind: 'local-lambert-equal-area'}
} as const;

it('GPUKernelDensity validates its area frame', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const props = {
    positions: createTransientView(graph, 'frame-positions', 'float32x2', 8),
    parameters: createTransientView(graph, 'frame-parameters', 'float32', 9),
    grid: GRID_4_BY_3,
    indexGridSize: [2, 2] as const,
    indexBounds: [0, 0, 4, 3] as const,
    output: {values: createTransientView(graph, 'frame-values', 'float32', 12)}
  };
  expect(() => new GPUKernelDensity({...props, spatialContext: DISTANCE_FRAME})).toThrow(
    /measures area; project into a lambert-equal-area frame/
  );
  expect(() => new GPUKernelDensity({...props, spatialContext: AREA_FRAME})).not.toThrow();
  device.destroy();
});

it('polygon KDE packs NaN rectangle slots, validates corrections, frames and graph ownership', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const otherGraph = new GPUCommandGraph(device);
  const base = {
    id: 'polygon-kde',
    spatialContext: {
      coordinateSpace: 'planar',
      metric: 'native',
      frame: {kind: 'crs', id: 'kde-frame'}
    } as const,
    positions: createTransientView(graph, 'points', 'float32x2', 8),
    parameters: createTransientView(graph, 'parameters', 'float32', 9),
    grid: GRID_4_BY_3,
    indexGridSize: [2, 2] as const,
    indexBounds: [0, 0, 4, 3] as const,
    output: {values: createTransientView(graph, 'values', 'float32', 12)}
  };
  const studyWindow = new GPUStudyWindowPreparation({
    positions: createTransientView(graph, 'vertices', 'float32x2', 4),
    featureOffsets: createTransientView(graph, 'features', 'uint32', 2),
    polygonOffsets: createTransientView(graph, 'polygons', 'uint32', 2),
    ringOffsets: createTransientView(graph, 'rings', 'uint32', 2),
    samplingGrid: [4, 4]
  });
  const packed = getGPUKernelDensityParameterValues(
    {bandwidth: 1, window: 'study-window'},
    GRID_4_BY_3
  );
  expect(Array.from(packed.slice(0, 5))).toEqual([0, 0, 4, 3, 1]);
  expect(Array.from(packed.slice(5)).every(Number.isNaN)).toBe(true);
  expect(() => new GPUKernelDensity({...base, studyWindow, edgeCorrection: 'none'})).toThrow(
    "polygon-kde studyWindow is ignored without edge correction; set edgeCorrection to 'uniform' or 'diggle'"
  );
  for (const spatialContext of [
    {coordinateSpace: 'longitude-latitude', metric: 'great-circle'} as const,
    {
      coordinateSpace: 'planar',
      metric: 'native',
      units: 'meters',
      frame: {kind: 'crs', id: 'other'}
    } as const
  ]) {
    expect(
      () =>
        new GPUKernelDensity({
          ...base,
          studyWindow: {
            kind: 'polygon',
            id: 'wrong-frame',
            spatialContext,
            getViews: () => studyWindow.getViews(graph)
          }
        })
    ).toThrow(/studyWindow incompatible spatial contexts/);
  }
  expect(() => new GPUKernelDensity({...base, studyWindow}).getCommandNodes(graph)).toThrow(
    /needs graph.add/
  );
  graph.add(studyWindow);
  const views = studyWindow.getViews(graph);
  expect(() =>
    new GPUKernelDensity({
      ...base,
      studyWindow: {
        kind: 'polygon',
        id: 'foreign-packed',
        spatialContext: studyWindow.spatialContext,
        getViews: () => ({
          ...views,
          packed: createTransientView(otherGraph, 'foreign', 'uint32', views.packed.length)
        })
      }
    }).getCommandNodes(graph)
  ).toThrow(/graph/);
  device.destroy();
});

it('polygon KDE uses one packed binding and one exact inclusion call per kernel at core limits', () => {
  const device = createNullWebGPUDevice();
  for (const method of ['exact', 'binned', 'auto'] as const) {
    for (const kernel of KERNELS) {
      if (method === 'binned' && kernel !== 'gaussian') {
        continue;
      }
      for (const edgeCorrection of ['uniform', 'diggle'] as const) {
        const graph = new GPUCommandGraph(device);
        const studyWindow = new GPUStudyWindowPreparation({
          positions: createTransientView(graph, 'vertices', 'float32x2', 6),
          featureOffsets: createTransientView(graph, 'features', 'uint32', 2),
          polygonOffsets: createTransientView(graph, 'polygons', 'uint32', 2),
          ringOffsets: createTransientView(graph, 'rings', 'uint32', 2),
          samplingGrid: [4, 4]
        });
        graph.add(studyWindow);
        const nodes = new GPUKernelDensity({
          studyWindow,
          method,
          kernel,
          edgeCorrection,
          bandwidthRule: 'scott',
          normalization: 'probability',
          positions: createTransientView(graph, 'points', 'float32x2', 8),
          weights: createTransientView(graph, 'weights', 'float32', 8),
          parameters: createTransientView(graph, 'parameters', 'float32', 9),
          grid: GRID_4_BY_3,
          indexGridSize: [2, 2],
          indexBounds: [0, 0, 4, 3],
          output: {
            values: createTransientView(graph, 'values', 'float32', 12),
            weightSum: createTransientView(graph, 'weightSum', 'float32', 1),
            status: {invalidCount: createTransientView(graph, 'invalid', 'uint32', 1)}
          }
        }).getCommandNodes(graph);
        const sources = getGPUCommandNodeKernelSources(nodes);
        const windowSources = sources.filter(({source}) =>
          source.includes('fn isInsideStudyWindow')
        );
        expect(windowSources.length).toBeGreaterThanOrEqual(2);
        for (const {source} of sources) {
          expect(getStorageBindingCount(source)).toBeLessThanOrEqual(8);
        }
        for (const {source, constants} of windowSources) {
          expect(source).toMatch(/var<storage, read> studyWindow: array<u32>/);
          expect(source.match(/isInsideStudyWindow\(/g)).toHaveLength(2); // declaration + single call
          expect(source.match(/fn isFiniteBits\(/g)).toHaveLength(1);
          expect(source.match(/fn exactProbeCrossing\(/g)).toHaveLength(1);
          expect(source).not.toContain('isInsideStudyWindow(relative)');
          expect(auditWGSLModule(source, constants)).toEqual([]);
        }
      }
    }
  }
  device.destroy();
});

it('explicit rectangle preserves default KDE nodes and byte-identical WGSL', () => {
  const device = createNullWebGPUDevice();
  for (const method of ['exact', 'binned', 'auto'] as const) {
    for (const edgeCorrection of ['none', 'uniform', 'diggle'] as const) {
      const build = (explicit: boolean) => {
        const graph = new GPUCommandGraph(device);
        const nodes = new GPUKernelDensity({
          ...(explicit ? {studyWindow: 'rectangle' as const} : {}),
          method,
          edgeCorrection,
          positions: createTransientView(graph, 'points', 'float32x2', 8),
          parameters: createTransientView(graph, 'parameters', 'float32', 9),
          grid: GRID_4_BY_3,
          indexGridSize: [2, 2],
          indexBounds: [0, 0, 4, 3],
          output: {values: createTransientView(graph, 'values', 'float32', 12)}
        }).getCommandNodes(graph);
        return {ids: nodes.map(node => node.id), sources: getGPUCommandNodeKernelSources(nodes)};
      };
      expect(build(true)).toEqual(build(false));
    }
  }
  device.destroy();
});

it('polygon exact oracle agrees with the pinned rectangle fields at their documented tolerance', () => {
  const {field} = KERNEL_DENSITY_ORACLE;
  const [minimumX, minimumY, maximumX, maximumY] = field.window;
  const window = {
    positions: new Float64Array([
      minimumX,
      minimumY,
      maximumX,
      minimumY,
      maximumX,
      maximumY,
      minimumX,
      maximumY
    ]),
    polygonOffsets: new Uint32Array([0, 1]),
    ringOffsets: new Uint32Array([0, 4])
  };
  for (const testCase of field.cases) {
    if (testCase.edgeCorrection === 'none') {
      continue;
    }
    const actual = getPolygonKernelDensityOracle({
      window,
      points: field.points,
      weights: testCase.weighted ? field.weights : undefined,
      width: field.columns,
      height: field.rows,
      start: [minimumX, minimumY],
      cellSize: [(maximumX - minimumX) / field.columns, (maximumY - minimumY) / field.rows],
      bandwidth: field.bandwidth,
      kernel: testCase.kernel,
      edgeCorrection: testCase.edgeCorrection
    });
    // The pinned sklearn Gaussian field is untruncated; use its existing field tolerance.
    // Polygon-versus-rectangle GPU parity uses the same truncated sum and keeps 2e-5 relative.
    const tolerance = 2e-4 * Math.max(...testCase.values.map(Math.abs));
    for (const [index, expected] of testCase.values.entries()) {
      expect(Math.abs(actual[index] - expected)).toBeLessThanOrEqual(tolerance);
    }
  }
});
