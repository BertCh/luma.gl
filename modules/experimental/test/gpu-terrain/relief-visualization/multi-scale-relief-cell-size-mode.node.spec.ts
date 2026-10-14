// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {makeGPURasterMetadata} from '../../../src/gpu-raster';
import {
  GPUMultiScaleRelief,
  type GPUMultiScaleReliefProps
} from '../../../src/gpu-terrain/relief-visualization/gpu-multi-scale-relief';
import {
  createReliefMeanFilterSetup,
  GPU_RELIEF_DEFAULT_QUANTUM
} from '../../../src/gpu-terrain/relief-visualization/relief-visualization-utils';
import {
  getRasterGridGeoreference,
  getRasterGroundCellSize
} from '../../../src/analysis-core/raster-grid';
import {
  createNullWebGPUDevice,
  getParsedKernelSources,
  getGPUCommandNodePipelineKeys
} from '../../utils/gpu-contributor-test-utils';
import {createBand} from '../terrain-test-utils';
import {
  computeGroundMultiScaleReliefRVT,
  computeMultiScaleReliefRVT,
  getGroundReliefRowRadii
} from './relief-visualization-oracle';

const GRID = makeGPURasterMetadata({width: 9, height: 7, cellSize: 0.01, origin: [0, 60.035]});

function createRelief(overrides: Partial<GPUMultiScaleReliefProps> = {}, separable = false) {
  const device = createNullWebGPUDevice();
  const grid = overrides.grid ?? GRID;
  const pixelCount = grid.width * grid.height;
  if (separable) {
    // Native WebGPU limits are prototype getters, not enumerable own properties.
    // Also model their receiver requirement so forwarding through a proxy is checked.
    const originalLimits: typeof device.limits = Object.create(
      Object.create(
        null,
        Object.fromEntries(
          Object.entries(device.limits).map(([name, value]) => [
            name,
            {
              get() {
                expect(this).toBe(originalLimits);
                return value;
              }
            }
          ])
        )
      )
    );
    // One scalar plane fits; the summed-area table's three planes do not.
    Object.defineProperty(device, 'limits', {
      value: new Proxy(originalLimits, {
        get(target, property) {
          return property === 'maxStorageBufferBindingSize'
            ? pixelCount * 4
            : Reflect.get(target, property, target);
        }
      })
    });
  }
  const graph = new GPUCommandGraph(device);
  const props: GPUMultiScaleReliefProps = {
    grid,
    resolution: 1000,
    featureMinimum: 3000,
    featureMaximum: 11000,
    scalingFactor: 1,
    elevation: createBand(graph, 'elevation', pixelCount),
    parameters: createTransientView(graph, 'settings', 'float32', 4),
    output: {relief: createTransientView(graph, 'relief', 'float32', pixelCount)},
    ...overrides
  };
  return {graph, contributor: new GPUMultiScaleRelief(props)};
}

it('MSRM rejects invalid modes and rotated ground-scaled grids', () => {
  // @ts-expect-error Exercise runtime validation of untyped input.
  expect(() => createRelief({cellSizeMode: 'degrees'})).toThrow(/cellSizeMode/);
  const grid = {...GRID, affine: [0.01, 0.001, 0, 0, -0.01, 60] as const};
  expect(() => createRelief({cellSizeMode: 'geographic', grid})).toThrow(/axis-aligned/);
  expect(() => createRelief({cellSizeMode: 'web-mercator', grid})).toThrow(/axis-aligned/);
  expect(() => createRelief({cellSizeMode: 'uniform', grid})).not.toThrow();
});

for (const separable of [false, true]) {
  it(`MSRM default and uniform shaders and variants are identical (separable=${separable})`, () => {
    const omitted = createRelief({}, separable);
    const uniform = createRelief({cellSizeMode: 'uniform'}, separable);
    const omittedNodes = omitted.contributor.getCommandNodes(omitted.graph);
    const uniformNodes = uniform.contributor.getCommandNodes(uniform.graph);
    expect(getParsedKernelSources(uniformNodes)).toEqual(getParsedKernelSources(omittedNodes));
    expect(getGPUCommandNodePipelineKeys(uniformNodes)).toEqual(
      getGPUCommandNodePipelineKeys(omittedNodes)
    );
    expect(uniformNodes.map(node => node.workload)).toEqual(
      omittedNodes.map(node => node.workload)
    );
    expect(uniform.contributor.requiredHalo).toBe(5);
  });

  for (const cellSizeMode of ['geographic', 'web-mercator'] as const) {
    it(`MSRM bakes float64 per-axis bounds and parses ${cellSizeMode} kernels (separable=${separable})`, () => {
      const grid =
        cellSizeMode === 'geographic'
          ? GRID
          : makeGPURasterMetadata({width: 9, height: 7, cellSize: 1000, origin: [0, 8399737.89]});
      const {graph, contributor} = createRelief(
        {grid, cellSizeMode, resolution: 1060, featureMinimum: 3180, featureMaximum: 11660},
        separable
      );
      const georeference = getRasterGridGeoreference(grid);
      const nodes = contributor.getCommandNodes(graph);
      const sources = getParsedKernelSources(nodes);
      const filterVariants = nodes
        .filter(node => /-(first|last)-(rows|columns|deviation)$/.test(node.id))
        .map(node => node.workload?.variant);
      expect(filterVariants).toEqual(
        separable
          ? [
              `mean-rows-${cellSizeMode}`,
              `mean-columns-${cellSizeMode}`,
              `mean-rows-${cellSizeMode}`,
              `mean-columns-${cellSizeMode}`
            ]
          : [`mean-summed-area-${cellSizeMode}`, `mean-summed-area-${cellSizeMode}`]
      );
      for (const [filter, pixelRadius] of [
        ['first', 1],
        ['last', 5]
      ] as const) {
        const radii = Array.from({length: grid.height}, (_, row) => {
          const groundCell = getRasterGroundCellSize(
            cellSizeMode,
            georeference,
            row + 0.5,
            grid.height
          );
          return groundCell.map(size => Math.max(1, Math.round((pixelRadius * 1060) / size)));
        });
        const maximumX = Math.max(...radii.map(radius => radius[0]));
        const maximumY = Math.max(...radii.map(radius => radius[1]));
        const source = sources.get(
          `multi-scale-relief-${filter}-${separable ? 'rows' : 'deviation'}`
        )!;
        expect(source).toContain(`const MAXIMUM_RADIUS_X: i32 = ${maximumX};`);
        expect(source).toContain(`const MAXIMUM_RADIUS_Y: i32 = ${maximumY};`);
        expect(source).toContain('fn getReliefGroundCell(row: u32)');
        expect(source).not.toContain('fn getGroundCellSize');
        expect(source).toContain(
          'return clamp(rounded, vec2<i32>(1), vec2<i32>(MAXIMUM_RADIUS_X, MAXIMUM_RADIUS_Y))'
        );
        if (filter === 'last') {
          expect(contributor.requiredHalo).toBe(Math.max(maximumX, maximumY));
          if (cellSizeMode === 'geographic')
            expect(new Set(radii.map(radius => radius[0])).size).toBe(2);
        }
        if (separable) {
          expect(source).toContain('let radius = getReliefPixelRadius(index / WIDTH).x');
          expect(sources.get(`multi-scale-relief-${filter}-columns`)).toContain(
            'let radius = getReliefPixelRadius(u32(row)).y'
          );
        } else {
          expect(source).toContain('column - radiusX');
          expect(source).toContain('row - radiusY');
          expect(source).toContain('max(radiusX - column, 0i)');
          expect(source).toContain('max(radiusY - row, 0i)');
        }
      }
    });
  }

  it(`ground windows reject per-frame radius slots (separable=${separable})`, () => {
    const {graph, contributor} = createRelief({}, separable);
    const setup = createReliefMeanFilterSetup(graph, {
      id: 'filter',
      width: GRID.width,
      height: GRID.height,
      band: {
        ...createBand(graph, 'filter-elevation', GRID.width * GRID.height),
        validity: createTransientView(graph, 'filter-validity', 'uint32', GRID.width * GRID.height)
      },
      quantum: GPU_RELIEF_DEFAULT_QUANTUM
    });
    expect(() =>
      setup.getFilterNodes({
        id: 'conflict',
        operation: 'test',
        radius: 2,
        radiusSlot: {settings: contributor.props.parameters, index: 0},
        groundRadius: {
          mode: 'geographic',
          ...getRasterGridGeoreference(GRID),
          radius: 2000,
          maximumRadius: [4, 2]
        },
        deviation: createTransientView(graph, 'deviation', 'float32', GRID.width * GRID.height)
      })
    ).toThrow(/groundRadius cannot use radiusSlot/);
  });
}

it('MSRM validates the ground maximum instead of the RVT pixel radius', () => {
  const {graph, contributor} = createRelief({
    cellSizeMode: 'geographic',
    resolution: 3000000,
    featureMinimum: 3000000,
    featureMaximum: 33000000
  });
  expect(contributor.radii.lastRadius).toBe(5);
  expect(contributor.requiredHalo).toBeGreaterThan(23170);
  expect(() => contributor.getCommandNodes(graph)).toThrow(/radius must be at most/);
});

it('ground MSRM agrees exactly with square-window RVT at the equator, including radius zero', () => {
  const grid = makeGPURasterMetadata({width: 9, height: 7, cellSize: 0.001, origin: [0, 0.0035]});
  const elevation = Float32Array.from({length: 63}, (_, index) => 100 + (index % 13) / 8);
  elevation[0] = NaN;
  elevation[31] = NaN;
  const scales = {resolution: 111, featureMinimum: 111, featureMaximum: 1221, scalingFactor: 1};
  for (let radius = 0; radius <= 5; radius++) {
    expect(getGroundReliefRowRadii(grid, 'geographic', radius * scales.resolution)).toEqual(
      Array.from({length: grid.height}, () => [radius, radius])
    );
  }
  const options = {
    ...scales,
    width: grid.width,
    height: grid.height,
    grid,
    elevation,
    cellSizeMode: 'geographic' as const
  };
  expect(computeGroundMultiScaleReliefRVT(options)).toEqual(computeMultiScaleReliefRVT(options));
  const {graph, contributor} = createRelief({...scales, grid, cellSizeMode: 'geographic'});
  expect(contributor.getCommandNodes(graph).some(node => node.id.includes('-first-'))).toBe(false);
});
