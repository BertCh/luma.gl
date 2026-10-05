// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Texture, type Buffer, type Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPURasterTextureToBuffer} from '../../../src/gpu-raster';
import {GPUParameterBuffer, importGraphBuffer, submitGraph} from '../../../src/utils/gpu-contributor-utils';
import {
  GPUTerrainDerivatives,
  getGPUTerrainDerivativesParameterValues,
  type GPUTerrainDerivativesProps
} from '../../../src/gpu-terrain/terrain-analysis';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32
} from '../../utils/gpu-contributor-test-utils';
import {computeTerrainDerivatives, type TerrainDerivativeOptions} from './terrain-analysis-oracle';

const WIDTH = 6;
const HEIGHT = 5;
const PIXEL_COUNT = WIDTH * HEIGHT;
const PLANE = Float32Array.from(
  {length: PIXEL_COUNT},
  (_, index) => 2 * (index % WIDTH) + 3 * Math.floor(index / WIDTH)
);

function expectCloseArray(actual: number[], expected: number[], angular: boolean = false): void {
  expect(actual.length).toBe(expected.length);
  for (const [index, value] of expected.entries()) {
    if (Number.isNaN(value)) {
      expect(Number.isNaN(actual[index])).toBe(true);
      continue;
    }
    let difference = Math.abs(actual[index] - value);
    if (angular) {
      difference = Math.min(difference, 360 - difference);
    }
    expect(difference).toBeLessThan(Math.max(5e-4, Math.abs(value) * 1e-4));
  }
}

type DerivativeFixture = {
  graph: GPUCommandGraph;
  settings: GPUParameterBuffer<'float32'>;
  buffers: {slope: Buffer; aspect: Buffer; hillshade: Buffer; validity: Buffer};
  owned: Buffer[];
};

function createDerivativeFixture(
  device: Device,
  elevation: Float32Array,
  settingsValues: Float32Array,
  overrides: (graph: GPUCommandGraph) => Partial<GPUTerrainDerivativesProps> = () => ({}),
  validityMask?: Uint32Array
): DerivativeFixture {
  const graph = new GPUCommandGraph(device, {id: 'terrain-derivatives-test'});
  const elevationBuffer = createInputBuffer(device, elevation);
  const maskBuffer = validityMask ? createInputBuffer(device, validityMask) : undefined;
  const buffers = {
    slope: createOutputBuffer(device, PIXEL_COUNT),
    aspect: createOutputBuffer(device, PIXEL_COUNT),
    hillshade: createOutputBuffer(device, PIXEL_COUNT),
    validity: createOutputBuffer(device, PIXEL_COUNT)
  };
  const settings = new GPUParameterBuffer(device, {
    id: 'derivative-settings',
    format: 'float32',
    length: 8,
    values: settingsValues
  });
  graph.add(
    new GPUTerrainDerivatives({
      width: WIDTH,
      height: HEIGHT,
      elevation: {
        id: 'elevation',
        format: 'float32',
        storage: {
          kind: 'buffer',
          values: importGraphBuffer(graph, 'elevation', elevationBuffer, 'float32', PIXEL_COUNT)
        },
        validity: maskBuffer
          ? importGraphBuffer(graph, 'mask', maskBuffer, 'uint32', PIXEL_COUNT)
          : undefined
      },
      settings: settings.importToGraph(graph),
      slope: importGraphBuffer(graph, 'slope', buffers.slope, 'float32', PIXEL_COUNT),
      aspect: importGraphBuffer(graph, 'aspect', buffers.aspect, 'float32', PIXEL_COUNT),
      hillshade: importGraphBuffer(graph, 'hillshade', buffers.hillshade, 'float32', PIXEL_COUNT),
      validity: importGraphBuffer(graph, 'validity', buffers.validity, 'uint32', PIXEL_COUNT),
      ...overrides(graph)
    })
  );
  return {
    graph,
    settings,
    buffers,
    owned: [elevationBuffer, ...(maskBuffer ? [maskBuffer] : []), ...Object.values(buffers)]
  };
}

async function expectDerivatives(
  fixture: DerivativeFixture,
  elevation: Float32Array,
  settingsValues: Float32Array,
  options: TerrainDerivativeOptions = {},
  mask?: Uint32Array
): Promise<void> {
  const oracle = computeTerrainDerivatives(elevation, mask, WIDTH, HEIGHT, settingsValues, options);
  expectCloseArray(await readFloat32(fixture.buffers.slope, PIXEL_COUNT), oracle.slope);
  expectCloseArray(await readFloat32(fixture.buffers.aspect, PIXEL_COUNT), oracle.aspect, true);
  expectCloseArray(await readFloat32(fixture.buffers.hillshade, PIXEL_COUNT), oracle.hillshade);
  expect(await readUint32(fixture.buffers.validity, PIXEL_COUNT)).toEqual(oracle.validity);
}

function destroyFixture(fixture: DerivativeFixture): void {
  fixture.settings.destroy();
  for (const buffer of fixture.owned) buffer.destroy();
}

it('GPUTerrainDerivatives shades a plane and updates settings per frame', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const settingsValues = getGPUTerrainDerivativesParameterValues({cellSize: [10, 10]});
  const fixture = createDerivativeFixture(device, PLANE, settingsValues);
  const compiled = fixture.graph.compile();
  submitGraph(device, compiled, undefined);
  await expectDerivatives(fixture, PLANE, settingsValues);
  const interior = 2 * WIDTH + 2;
  expect((await readFloat32(fixture.buffers.slope, PIXEL_COUNT))[interior]).toBeCloseTo(
    (Math.atan(Math.sqrt(0.13)) * 180) / Math.PI,
    3
  );
  expect((await readFloat32(fixture.buffers.aspect, PIXEL_COUNT))[interior]).toBeCloseTo(
    326.3099,
    3
  );

  const updated = getGPUTerrainDerivativesParameterValues({
    cellSize: [10, 10],
    zFactor: 2,
    azimuthDegrees: 90,
    altitudeDegrees: 30
  });
  fixture.settings.write(updated);
  submitGraph(device, compiled, undefined);
  await expectDerivatives(fixture, PLANE, updated);
  compiled.destroy();
  destroyFixture(fixture);
});

it('GPUTerrainDerivatives handles flat terrain and nodata', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const settingsValues = getGPUTerrainDerivativesParameterValues({cellSize: [10, 10]});
  const flat = new Float32Array(PIXEL_COUNT).fill(100);
  const flatFixture = createDerivativeFixture(device, flat, settingsValues);
  let compiled = flatFixture.graph.compile();
  submitGraph(device, compiled, undefined);
  expect(
    (await readFloat32(flatFixture.buffers.slope, PIXEL_COUNT)).every(value => value === 0)
  ).toBe(true);
  expect(
    (await readFloat32(flatFixture.buffers.aspect, PIXEL_COUNT)).every(value => value === -1)
  ).toBe(true);
  for (const value of await readFloat32(flatFixture.buffers.hillshade, PIXEL_COUNT)) {
    expect(value).toBeCloseTo(Math.SQRT1_2, 4);
  }
  compiled.destroy();
  destroyFixture(flatFixture);

  const mask = new Uint32Array(PIXEL_COUNT).fill(1);
  mask[2 * WIDTH + 2] = 0;
  const maskFixture = createDerivativeFixture(device, PLANE, settingsValues, () => ({}), mask);
  compiled = maskFixture.graph.compile();
  submitGraph(device, compiled, undefined);
  await expectDerivatives(maskFixture, PLANE, settingsValues, {}, mask);
  const validity = await readUint32(maskFixture.buffers.validity, PIXEL_COUNT);
  expect(validity[1 * WIDTH + 1]).toBe(0);
  expect(validity[3 * WIDTH + 3]).toBe(0);
  expect(validity[4 * WIDTH + 5]).toBe(1);
  compiled.destroy();
  destroyFixture(maskFixture);
});

it('GPUTerrainDerivatives scales Web Mercator and geographic rows', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const mercatorSettings = getGPUTerrainDerivativesParameterValues({
    cellSize: [10, 10],
    northEdge: 0.25,
    southEdge: 0.375
  });
  const mercator = createDerivativeFixture(device, PLANE, mercatorSettings, () => ({
    cellSizeMode: 'web-mercator'
  }));
  let compiled = mercator.graph.compile();
  submitGraph(device, compiled, undefined);
  await expectDerivatives(mercator, PLANE, mercatorSettings, {cellSizeMode: 'web-mercator'});
  const slope = await readFloat32(mercator.buffers.slope, PIXEL_COUNT);
  expect(slope[2]).toBeGreaterThan(slope[(HEIGHT - 1) * WIDTH + 2]);
  compiled.destroy();
  destroyFixture(mercator);

  const geographicSettings = getGPUTerrainDerivativesParameterValues({
    cellSize: [0.001, 0.001],
    northEdge: 60,
    southEdge: 59.9
  });
  const geographic = createDerivativeFixture(device, PLANE, geographicSettings, () => ({
    cellSizeMode: 'geographic'
  }));
  compiled = geographic.graph.compile();
  submitGraph(device, compiled, undefined);
  await expectDerivatives(geographic, PLANE, geographicSettings, {cellSizeMode: 'geographic'});
  compiled.destroy();
  destroyFixture(geographic);
});

it('GPUTerrainDerivatives supports north row order, percent slope, and textures', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const settingsValues = getGPUTerrainDerivativesParameterValues({cellSize: [10, 10]});
  const north = createDerivativeFixture(device, PLANE, settingsValues, () => ({
    rowDirection: 'north',
    slopeUnits: 'percent'
  }));
  let compiled = north.graph.compile();
  submitGraph(device, compiled, undefined);
  await expectDerivatives(north, PLANE, settingsValues, {
    rowDirection: 'north',
    slopeUnits: 'percent'
  });
  const interior = 2 * WIDTH + 2;
  expect((await readFloat32(north.buffers.aspect, PIXEL_COUNT))[interior]).toBeCloseTo(213.69, 1);
  expect((await readFloat32(north.buffers.slope, PIXEL_COUNT))[interior]).toBeCloseTo(36.0555, 3);
  compiled.destroy();
  destroyFixture(north);

  if (!device.getTextureFormatCapabilities('r32float').store) {
    return;
  }
  const inputTexture = device.createTexture({
    format: 'r32float',
    width: WIDTH,
    height: HEIGHT,
    usage: Texture.SAMPLE | Texture.COPY_DST
  });
  inputTexture.writeData(PLANE);
  const outputTexture = device.createTexture({
    format: 'r32float',
    width: WIDTH,
    height: HEIGHT,
    usage: Texture.STORAGE | Texture.SAMPLE | Texture.COPY_DST
  });
  const readbackBuffer = createOutputBuffer(device, PIXEL_COUNT);
  const readbackValidityBuffer = createOutputBuffer(device, PIXEL_COUNT);
  const settings = new GPUParameterBuffer(device, {
    id: 'texture-settings',
    format: 'float32',
    length: 8,
    values: settingsValues
  });
  const graph = new GPUCommandGraph(device, {id: 'terrain-texture'});
  const importTexture = (id: string, texture: Texture) =>
    graph.createTextureView(
      graph.importTexture(
        {id, format: 'r32float', width: WIDTH, height: HEIGHT, usage: texture.props.usage},
        texture
      ),
      {mipLevelCount: 1}
    ) as never;
  const outputView = importTexture('hillshade-texture', outputTexture);
  graph.add(
    new GPUTerrainDerivatives({
      width: WIDTH,
      height: HEIGHT,
      elevation: {
        id: 'elevation',
        format: 'float32',
        storage: {kind: 'texture', view: importTexture('elevation-texture', inputTexture)}
      },
      settings: settings.importToGraph(graph),
      hillshadeTexture: outputView
    })
  );
  new GPURasterTextureToBuffer({
    id: 'hillshade-readback',
    input: {id: 'hillshade-band', format: 'float32', storage: {kind: 'texture', view: outputView}},
    output: importGraphBuffer(graph, 'readback', readbackBuffer, 'float32', PIXEL_COUNT),
    outputValidity: importGraphBuffer(
      graph,
      'readback-validity',
      readbackValidityBuffer,
      'uint32',
      PIXEL_COUNT
    )
  }).addToGraph(graph);
  compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  expectCloseArray(
    await readFloat32(readbackBuffer, PIXEL_COUNT),
    computeTerrainDerivatives(PLANE, undefined, WIDTH, HEIGHT, settingsValues).hillshade
  );
  compiled.destroy();
  settings.destroy();
  for (const resource of [inputTexture, outputTexture, readbackBuffer, readbackValidityBuffer]) {
    resource.destroy();
  }
});
