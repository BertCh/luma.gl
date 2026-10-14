// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  getGPUTerrainCastShadowParameterValues,
  GPU_TERRAIN_CAST_SHADOW_PARAMETER_LENGTH,
  GPUTerrainCastShadow,
  type GPUTerrainCastShadowOutput,
  type GPUTerrainCastShadowProps
} from '../../../src/gpu-terrain/terrain-illumination/gpu-terrain-cast-shadow';
import {
  getTerrainSweepGroundLineGeometry,
  getTerrainSweepLineGeometry,
  getTerrainSweepMirroredLineGeometry
} from '../../../src/gpu-terrain/terrain-illumination/terrain-horizon-sweep';
import {
  getRasterGridGeoreference,
  getRasterGroundCellSize
} from '../../../src/analysis-core/raster-grid';
import {makeGPURasterMetadata} from '../../../src/gpu-raster';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

it('getGPUTerrainCastShadowParameterValues packs the sun line geometry', () => {
  expect(GPU_TERRAIN_CAST_SHADOW_PARAMETER_LENGTH).toBe(16);
  // Cell size 10 x 12 on a 48 x 40 grid: the row edges are y = 0 and y = -480.
  const grid = makeGPURasterMetadata({width: 48, height: 40, cellSize: [10, 12]});
  const pack = (azimuthDegrees: number) =>
    Array.from(
      getGPUTerrainCastShadowParameterValues(
        {azimuthDegrees, altitudeDegrees: 20, zFactor: 2},
        grid
      )
    );
  // East: x-major, travel +1, no slope; the last two slots hold the ground direction (1, 0).
  expect(pack(90)).toEqual([
    10,
    12,
    2,
    0,
    0,
    -480,
    0,
    90,
    20,
    expect.any(Number),
    1,
    0,
    1,
    1,
    1,
    0
  ]);
  // The packer writes the grid georeference into the cell size and row edge slots.
  const georeference = getRasterGridGeoreference(grid);
  expect(pack(90).slice(0, 2)).toEqual([...georeference.cellSize]);
  expect(pack(90).slice(4, 6)).toEqual([georeference.northEdge, georeference.southEdge]);
  // North on a south-increasing grid goes toward row 0: y-major, travel -1.
  expect(pack(0).slice(10, 14)).toEqual([0, 0, -1, 1]);
  // The packer always uses south-advancing rows (the contributor mirrors the line for
  // north-advancing rows).
  // North-east: the packer stores the square-cell line (slope 1, step sqrt(2)) and the ground
  // direction (sin 45, -cos 45). On 10 x 12 m cells the kernel converts that direction (CPU twin
  // getTerrainSweepGroundLineGeometry) to the ground diagonal, not the pixel diagonal: the pixel
  // step is normalize((sin 45 / 10, cos 45 / 12)), slope 10 / 12, slopeFixed
  // round(65536 * 10 / 12) = 54613 and step hypot(1, 54613 / 65536) = 1.3017050231774017
  // (python3 -I).
  const diagonal = pack(45);
  expect(Math.abs(diagonal[11])).toBe(65536);
  expect(diagonal.slice(14)).toEqual([Math.fround(Math.SQRT1_2), Math.fround(-Math.SQRT1_2)]);
  const converted = getTerrainSweepGroundLineGeometry(
    [diagonal[14], diagonal[15]],
    [10, 12],
    48,
    40
  );
  expect(Math.abs(converted.slopeFixed)).toBe(54613);
  expect(converted.stepPixelLength).toBe(Math.fround(1.3017050231774017));
  // On square cells north-east is the exact pixel diagonal: slope 1, step sqrt(2).
  const square = Array.from(
    getGPUTerrainCastShadowParameterValues(
      {azimuthDegrees: 45, altitudeDegrees: 20},
      makeGPURasterMetadata({width: 48, height: 40, cellSize: 10})
    )
  );
  expect(Math.abs(square[11])).toBe(65536);
  expect(square[13]).toBe(Math.fround(Math.SQRT2));
  // A 22.5 degree sun is steep in one axis only.
  const steep = pack(22.5);
  expect(steep[10]).toBe(0); // y-major
  expect(Math.abs(steep[11])).toBeLessThan(65536);
  expect(Math.abs(steep[11])).toBeGreaterThan(0);
  expect(
    Array.from(
      getGPUTerrainCastShadowParameterValues(
        {azimuthDegrees: 5, altitudeDegrees: 5},
        makeGPURasterMetadata({width: 48, height: 40})
      )
    )[9]
  ).toBeCloseTo(0.2666, 6);
});

it('cast shadow takes the cell size mode from the contributor only', () => {
  // The packer has no cell size mode, so the settings cannot disagree with the contributor: the
  // same packed values serve a uniform and a geographic contributor, and the kernel converts the
  // packed ground direction with the contributor's mode. At 60 N (0.001 degree cells, python3 -I
  // WGS84: [55.80000157243614, 111.41228745777943] m) a 45 degree sun converts to slopeFixed
  // round(-32823.209957312734) = -32823, the true 45 degree ground line; the packed square-cell
  // line (slopeFixed -65536) is a 26.60365263545291 degree ground ray.
  const values = getGPUTerrainCastShadowParameterValues(
    {azimuthDegrees: 45, altitudeDegrees: 20},
    makeGPURasterMetadata({width: 48, height: 24, cellSize: 0.001, origin: [0, 60]})
  );
  expect(values[11]).toBe(-65536);
  const groundCell = getRasterGroundCellSize(
    'geographic',
    {cellSize: [0.001, 0.001], northEdge: 60, southEdge: 60},
    0.5,
    1
  );
  const converted = getTerrainSweepGroundLineGeometry([values[14], values[15]], groundCell, 48, 24);
  expect(converted).toMatchObject({xMajor: true, slopeFixed: -32823, travelSign: 1});
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  for (const cellSizeMode of ['uniform', 'web-mercator', 'geographic'] as const) {
    const contributor = new GPUTerrainCastShadow({
      id: `cast-${cellSizeMode}`,
      grid: makeGPURasterMetadata({width: 6, height: 5}),
      cellSizeMode,
      elevation: {
        id: `elevation-${cellSizeMode}`,
        format: 'float32',
        storage: {
          kind: 'buffer',
          values: createTransientView(graph, `values-${cellSizeMode}`, 'float32', 30)
        }
      },
      parameters: createTransientView(graph, `parameters-${cellSizeMode}`, 'float32', 16),
      output: {
        horizonAngle: createTransientView(graph, `horizon-${cellSizeMode}`, 'float32', 30)
      }
    });
    expect(contributor.getCommandNodes(graph).some(node => node.id.endsWith('-sweep'))).toBe(true);
  }
  device.destroy();
});

it('cast shadow mirrors the packed line for north rows exactly like the old north packing', () => {
  // The kernel negates slopeFixed (and the travel sign of y-major lines) of the south-packed line
  // for rowDirection 'north'. That equals packing (sin az, +cos az) directly except at exact
  // half-unit rounding ties of slopeFixed; none occurs on a 0.1 degree azimuth grid.
  const snap = (value: number) => (Math.abs(value) < 1e-9 ? 0 : value);
  for (let tenth = 0; tenth < 3600; tenth++) {
    const azimuthDegrees = tenth / 10;
    const values = getGPUTerrainCastShadowParameterValues(
      {azimuthDegrees, altitudeDegrees: 10},
      makeGPURasterMetadata({width: 48, height: 40, cellSize: 10})
    );
    const packed = {
      xMajor: values[10] !== 0,
      slopeFixed: values[11],
      travelSign: values[12] > 0 ? (1 as const) : (-1 as const),
      firstLine: 0,
      lineCount: 0,
      stepPixelLength: values[13]
    };
    const mirrored = getTerrainSweepMirroredLineGeometry(packed, 1, 1);
    const azimuth = (azimuthDegrees * Math.PI) / 180;
    const north = getTerrainSweepLineGeometry(
      [snap(Math.sin(azimuth)), snap(Math.cos(azimuth))],
      1,
      1
    );
    expect(
      [mirrored.xMajor, mirrored.slopeFixed, mirrored.travelSign, mirrored.stepPixelLength],
      `${azimuthDegrees}`
    ).toEqual([north.xMajor, north.slopeFixed, north.travelSign, north.stepPixelLength]);
  }
});

it('getGPUTerrainCastShadowParameterValues validates its input', () => {
  const valid = {azimuthDegrees: 1, altitudeDegrees: 1};
  const grid = makeGPURasterMetadata({width: 48, height: 40});
  expect(() =>
    getGPUTerrainCastShadowParameterValues(
      valid,
      makeGPURasterMetadata({width: 48, height: 40, cellSize: [0, 1]})
    )
  ).toThrow(/cell size/);
  expect(() =>
    getGPUTerrainCastShadowParameterValues({...valid, angularRadiusDegrees: -1}, grid)
  ).toThrow(/angular radius/);
  expect(() =>
    getGPUTerrainCastShadowParameterValues({...valid, azimuthDegrees: Number.NaN}, grid)
  ).toThrow(/finite/);
  expect(() => getGPUTerrainCastShadowParameterValues(valid, grid, new Float32Array(15))).toThrow(
    /16 values/
  );
});

it('GPUTerrainCastShadow validates props and schedules one sweep node', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let instance = 0;
  const create = (
    overrides: Partial<Omit<GPUTerrainCastShadowProps, 'output'>> = {},
    outputOverrides?: GPUTerrainCastShadowOutput
  ) => {
    instance++;
    const length = 30;
    return new GPUTerrainCastShadow({
      grid: makeGPURasterMetadata({width: 6, height: 5}),
      elevation: {
        id: `elevation-${instance}`,
        format: 'float32',
        storage: {
          kind: 'buffer',
          values: createTransientView(graph, `elevation-${instance}`, 'float32', length)
        }
      },
      parameters: createTransientView(graph, `parameters-${instance}`, 'float32', 16),
      output: outputOverrides ?? {
        sunVisibility: createTransientView(graph, `visibility-${instance}`, 'float32', length)
      },
      ...overrides
    });
  };
  const contributor = create();
  expect(contributor.id).toBe('terrain-cast-shadow');
  // Whole tile by default: the diagonal covers every azimuth.
  expect(contributor.maximumRadius).toBe(Math.ceil(Math.hypot(6, 5)));
  expect(contributor.requiredHalo).toBe(contributor.maximumRadius);
  expect(create({maximumRadius: 4}).requiredHalo).toBe(4);
  const nodes = contributor.getCommandNodes(graph);
  expect(nodes.some(node => node.id === 'terrain-cast-shadow-sweep')).toBe(true);
  expect(nodes.filter(node => node.id.includes('sweep'))).toHaveLength(1);

  expect(() => create({}, {})).toThrow(/at least one output/);
  expect(() => create({maximumRadius: 0})).toThrow(/maximumRadius/);
  expect(() => create({maximumRadius: 2.5})).toThrow(/maximumRadius/);
  expect(() => create({grid: makeGPURasterMetadata({width: 40000, height: 5})})).toThrow(/32767/);
  expect(() =>
    create({parameters: createTransientView(graph, 'short-parameters', 'float32', 8)})
  ).toThrow(/at least 16/);
  expect(() =>
    create({}, {sunVisibility: createTransientView(graph, 'short-visibility', 'float32', 3)})
  ).toThrow(/30 float32/);
  expect(() => create({cellSizeMode: 'bogus' as 'uniform'})).toThrow(/cellSizeMode/);
  expect(() => create({rowDirection: 'up' as 'south'})).toThrow(/rowDirection/);
  const other = new GPUCommandGraph(device);
  expect(() => create().getCommandNodes(other)).toThrow(/belong/);
});
