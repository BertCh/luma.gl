// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createHash} from 'node:crypto';
import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it, vi} from 'vitest';
import {WgslReflect} from 'wgsl_reflect';
import {getRasterPipelines} from '../../gpu-raster/raster-pipeline-count';
import {makeGPURasterMetadata} from '../../../src/gpu-raster';
import {
  GPUTerrainViewshed,
  GPUTerrainLineOfSight,
  GPUTerrainCumulativeViewshed,
  GPU_TERRAIN_VIEWSHED_PARAMETER_LENGTH,
  getGPUTerrainViewshedParameterValues,
  getGPUTerrainLineOfSightParameterValues
} from '../../../src/gpu-terrain/terrain-analysis';
import {
  getRasterGridGeoreference,
  getRasterGroundScale
} from '../../../src/analysis-core/raster-grid';
import {
  computeTerrainViewshed,
  computeTerrainSightLine,
  computeTerrainViewshedWithTolerance
} from './terrain-analysis-oracle';
import * as kernelNodes from '../../../src/utils/wgsl-kernel-nodes';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {createBand} from '../terrain-test-utils';

const KERNEL_CASES = [
  ['viewshed', 'march', false],
  ['viewshed', 'march', true],
  ['viewshed', 'pyramid', false],
  ['viewshed', 'pyramid', true],
  ['line-of-sight', 'march', false],
  ['line-of-sight', 'pyramid', false],
  ['cumulative', 'march', false],
  ['cumulative', 'pyramid', false]
] as const;

function captureKernel(
  [kind, traversal, tolerance]: (typeof KERNEL_CASES)[number],
  overrides: Record<string, unknown> = {}
) {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const capture = vi.spyOn(kernelNodes, 'createWGSLKernelNode');
  try {
    const common = {
      grid: makeGPURasterMetadata({width: 9, height: 7, cellSize: 30}),
      elevation: createBand(graph, 'elevation', 63),
      parameters: createTransientView(graph, 'parameters', 'float32', 12),
      traversal,
      ...overrides
    };
    const contributor =
      kind === 'viewshed'
        ? new GPUTerrainViewshed({
            ...common,
            toleranceParameters: tolerance
              ? createTransientView(graph, 'tolerance', 'float32', 4)
              : undefined,
            output: {visibility: createTransientView(graph, 'visibility', 'uint32', 63)}
          })
        : kind === 'line-of-sight'
          ? new GPUTerrainLineOfSight({
              ...common,
              pairs: createTransientView(graph, 'pairs', 'float32x4', 3),
              output: {
                visibility: createTransientView(graph, 'visibility', 'uint32', 3),
                clearance: createTransientView(graph, 'clearance', 'float32', 3)
              }
            })
          : new GPUTerrainCumulativeViewshed({
              ...common,
              observers: createTransientView(graph, 'observers', 'float32x2', 3),
              output: {
                visibleCount: createTransientView(graph, 'visible', 'uint32', 63),
                marginalCount: createTransientView(graph, 'marginal', 'uint32', 63)
              }
            });
    const nodes = contributor.getCommandNodes(graph);
    const {declarations, body, variant} = capture.mock.calls.at(-1)![1];
    const source = getRasterPipelines(nodes).at(-1)!.source;
    return {declarations, body, variant, source};
  } finally {
    capture.mockRestore();
    device.destroy();
  }
}

it('sight-line uniform declarations, bodies and variants retain the pre-cellSizeMode snapshot', () => {
  const hash = (source: string | undefined) =>
    createHash('sha256')
      .update(source ?? '')
      .digest('hex');
  const snapshots = KERNEL_CASES.map(kernelCase => {
    const kernel = captureKernel(kernelCase);
    expect(kernel).toEqual(captureKernel(kernelCase, {cellSizeMode: 'uniform'}));
    expect(kernel.declarations).not.toContain('getRayGroundCell');
    expect(kernel.body).not.toContain('getRayGroundCell');
    return {
      variant: kernel.variant,
      declarations: hash(kernel.declarations),
      body: hash(kernel.body)
    };
  });
  expect(snapshots).toMatchInlineSnapshot(`
    [
      {
        "body": "d1363442f69b2d11dc61ad2ad4821c2382b996c9f1b8788f8a8f326b854d6f3c",
        "declarations": "d4f51fb51f4fc5479a7b44a523d566ceacc54da947d7f847d8f1b6effe7f2b6e",
        "variant": "line-of-sight",
      },
      {
        "body": "a9d5700f6cdeebe7f3be9d4fe5b497e4df1b32dda98634b7e7b4234fd8b65363",
        "declarations": "1258582b39d0f20f09045b876493b6663fe42fc0a13ab1c12c20e0750113d883",
        "variant": "tolerance",
      },
      {
        "body": "070e70062226699927161edf927a95217e807a7f266909f36cef071d81cc73ce",
        "declarations": "b969f1312cb22a57d7fbecd1448b2e619e41eed68809c7956790be74a496ed5d",
        "variant": "pyramid",
      },
      {
        "body": "a9d5700f6cdeebe7f3be9d4fe5b497e4df1b32dda98634b7e7b4234fd8b65363",
        "declarations": "b969f1312cb22a57d7fbecd1448b2e619e41eed68809c7956790be74a496ed5d",
        "variant": "pyramid",
      },
      {
        "body": "79e8d4029a7f0ff1b1f0619d46413b04cfc5a3a94b539ad222710a2cc79cbadc",
        "declarations": "648a7e56682fe09d98830604ffa89e2d87f4e02bef759851568b8a305c250bfb",
        "variant": "march",
      },
      {
        "body": "79e8d4029a7f0ff1b1f0619d46413b04cfc5a3a94b539ad222710a2cc79cbadc",
        "declarations": "75f31b3e1f1b67b55dfc9bcfd400be951c460931fa8927d125cd2c6b2f1f3428",
        "variant": "pyramid",
      },
      {
        "body": "3e8c6c5cde8bb54ee2e1445b2d6e22c9195340c436f5c7b029249325c9d667c9",
        "declarations": "cde29dc57bcabc6184522e01c520f6d432145a9ef13991968d5b916670e42b7d",
        "variant": "march",
      },
      {
        "body": "3e8c6c5cde8bb54ee2e1445b2d6e22c9195340c436f5c7b029249325c9d667c9",
        "declarations": "7fedf362880a40129bea37ce1ac5c4cb3c4ebb821c750336e916e85ddcd7317b",
        "variant": "pyramid",
      },
    ]
  `);
});

it.each([
  'viewshed',
  'line-of-sight',
  'cumulative'
] as const)('%s validates cellSizeMode and rejects rotated non-uniform grids', kind => {
  const kernelCase = [kind, 'march', false] as const;
  expect(() => captureKernel(kernelCase, {cellSizeMode: 'polar'})).toThrow(
    'cellSizeMode must be uniform, web-mercator, or geographic'
  );
  const grid = makeGPURasterMetadata({width: 9, height: 7});
  grid.affine = [1, 0.1, 0, 0.1, -1, 60];
  for (const cellSizeMode of ['geographic', 'web-mercator'] as const) {
    expect(() => captureKernel(kernelCase, {grid, cellSizeMode})).toThrow(
      `${cellSizeMode} cell sizes need an axis-aligned grid`
    );
  }
  expect(() => captureKernel(kernelCase, {grid})).not.toThrow();
});

it('sight-line packers always write area and point grid row edges in world units', () => {
  expect(GPU_TERRAIN_VIEWSHED_PARAMETER_LENGTH).toBe(10);
  for (const pixelInterpretation of ['area', 'point'] as const) {
    for (const rowStep of [-0.25, 0.25, -30]) {
      const grid = makeGPURasterMetadata({width: 9, height: 7});
      grid.pixelInterpretation = pixelInterpretation;
      grid.affine = [0.25, 0, 10, 0, rowStep, 60];
      const northEdge = 60 + (pixelInterpretation === 'point' ? -0.5 * rowStep : 0);
      const edges = [northEdge, northEdge + 7 * rowStep];
      const viewshed = getGPUTerrainViewshedParameterValues({observer: [1, 2]}, grid);
      const sightLine = getGPUTerrainLineOfSightParameterValues({}, grid);
      expect(viewshed).toHaveLength(10);
      expect(sightLine).toHaveLength(12);
      expect(Array.from(viewshed.slice(8))).toEqual(edges);
      expect(Array.from(sightLine.slice(10))).toEqual(edges);
      expect(() =>
        getGPUTerrainViewshedParameterValues({observer: [1, 2]}, grid, new Float32Array(8))
      ).toThrow(/must hold 10/);
    }
  }
});

it('every non-uniform sight-line kernel uses the middle row and mode-specific variant', () => {
  for (const kernelCase of KERNEL_CASES) {
    const uniform = captureKernel(kernelCase);
    for (const cellSizeMode of ['geographic', 'web-mercator'] as const) {
      const kernel = captureKernel(kernelCase, {cellSizeMode});
      const edgeSlot = kernelCase[0] === 'viewshed' ? 8 : 10;
      const targetRow = kernelCase[0] === 'line-of-sight' ? 'targetPosition.y' : 'f32(row)';
      expect(kernel.variant).toBe(`${uniform.variant}-${cellSizeMode}`);
      expect(() => new WgslReflect(kernel.source)).not.toThrow();
      expect(kernel.declarations).toContain('fn getRayGroundCell(');
      expect(kernel.declarations).toContain(
        'mix(northEdge, southEdge, rowCoordinate / f32(HEIGHT))'
      );
      expect(kernel.declarations).toContain(
        cellSizeMode === 'geographic' ? 'sin(latitude)' : 'tanh(mercatorY)'
      );
      expect(kernel.body).toContain(
        `settings[settingsOffset + ${edgeSlot}u], settings[settingsOffset + ${edgeSlot + 1}u], 0.5 * (observer.y + ${targetRow}) + 0.5)`
      );
      expect(kernel.declarations).not.toContain('fn getPixelDirection');
      expect(kernel.declarations).not.toContain('fn getGroundCellSize');
    }
  }
});

it('the geographic sight-line oracle matches WGS84 meters per degree at the equator', () => {
  const grid = makeGPURasterMetadata({width: 7, height: 3});
  grid.affine = [1 / 3600, 0, 0, 0, -1 / 3600, 1.5 / 3600];
  const georeference = getRasterGridGeoreference(grid);
  const terrain = new Float64Array(21);
  const settings = getGPUTerrainLineOfSightParameterValues({observerHeight: 1}, grid);
  const result = computeTerrainSightLine(
    terrain,
    undefined,
    7,
    3,
    [0, 1],
    [4, 1],
    settings,
    undefined,
    {cellSizeMode: 'geographic', georeference}
  );
  // Independent WGS84 equatorial meters/degree: 2*pi*a/360, and that times (1-e^2).
  const metersPerDegree = [111319.49079327358, 110574.27582159436];
  expect(getRasterGroundScale('geographic', 0)[0]).toBeCloseTo(metersPerDegree[0], 8);
  expect(getRasterGroundScale('geographic', 0)[1]).toBeCloseTo(metersPerDegree[1], 8);
  expect(-1 / result.targetSlope).toBeCloseTo((4 * metersPerDegree[0]) / 3600, 10);
  const uniformSettings = Array.from(settings);
  uniformSettings[3] = metersPerDegree[0] / 3600;
  uniformSettings[4] = metersPerDegree[1] / 3600;
  const uniform = computeTerrainSightLine(
    terrain,
    undefined,
    7,
    3,
    [0, 1],
    [4, 1],
    uniformSettings
  );
  expect(result).toEqual(uniform);
});

it('the geographic oracle range ring at 60 degrees is elliptical in pixels', () => {
  const width = 257;
  const height = 257;
  const center = 128;
  const grid = makeGPURasterMetadata({width, height});
  grid.affine = [1 / 3600, 0, 0, 0, -1 / 3600, 60 + (center + 0.5) / 3600];
  const terrain = new Float64Array(width * height);
  const settings = getGPUTerrainViewshedParameterValues(
    {observer: [center, center], observerHeight: 1, maximumDistance: 1500},
    grid
  );
  const options = {
    cellSizeMode: 'geographic' as const,
    georeference: getRasterGridGeoreference(grid)
  };
  const codes = computeTerrainViewshed(terrain, undefined, width, height, settings, options);
  let columnRadius = 0;
  let rowRadius = 0;
  for (let offset = 1; offset <= center; offset++) {
    if (codes[center * width + center + offset] === 1) columnRadius = offset;
    if (codes[(center + offset) * width + center] === 1) rowRadius = offset;
  }
  const expectedRatio = 1 / Math.cos(Math.PI / 3);
  expect(Math.abs(columnRadius / rowRadius / expectedRatio - 1)).toBeLessThan(0.01);
  expect(codes[center * width + center + columnRadius + 1]).toBe(2);
  expect(codes[(center + rowRadius + 1) * width + center]).toBe(2);
  const tolerant = computeTerrainViewshedWithTolerance(
    terrain,
    undefined,
    width,
    height,
    settings,
    [0, 0, 0, 0],
    options
  );
  expect(tolerant.map(result => result.code)).toEqual(codes);
});
